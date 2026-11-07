import csv
import datetime
import os
import sys
import time
import numpy as np
import sounddevice as sd
import preprocess

# Import modern ai-edge-litert first, then fallback to tensorflow or tflite_runtime
tflite = None
for engine_name in ("ai_edge_litert.interpreter", "tensorflow.lite", "tflite_runtime.interpreter"):
    try:
        tflite = __import__(engine_name, fromlist=["Interpreter"])
        print(f"Using runtime engine: {engine_name}")
        break
    except ImportError:
        continue

if tflite is None:
    print("Error: Could not import ai_edge_litert, tensorflow.lite, or tflite_runtime.")
    sys.exit(1)

# Configuration
MODEL_TFLITE_PATH = "baby_cry_model.tflite"
SCALER_PATH = "scaler_stats.npz"
DURATION = 3.0  # seconds
CLASSES = ["discomfort", "hunger", "other"]

CRY_THRESHOLD = 0.80   # Combined P(discomfort) + P(hunger) threshold
SMOOTHING_WINDOW = 5    # number of frames for temporal smoothing (~2.5s of history)
NOISE_FLOOR = 0.02      # minimum RMS amplitude to run model (filters silence & quiet background)
CONSECUTIVE_FRAMES_REQUIRED = 2  # legacy debounce (kept for dashboard display)

# --- State-machine constants ---
T_CRY        = 0.60   # cry_prob threshold for state transitions
DETECT_SEC   = 3.0    # seconds to collect windows in DETECTING
MIN_HITS     = 2      # minimum windows with cry_prob >= T_CRY to confirm
MIN_AVG      = 0.50   # minimum mean cry_prob over ALL detecting windows to confirm
MAX_GAP_SEC  = 1.5    # cancel DETECTING if cry_prob < T_CRY for this many seconds straight
LOUD_RATIO   = 0      # 0 disables the loudness gate (keep gate code so it can be re-enabled)
MARGIN       = 0.10   # if |p_hunger - p_discomfort| < MARGIN -> label = "uncertain"
COOLDOWN_SEC = 2.0    # seconds to remain in DECIDE before returning to IDLE

# Sample rate is set dynamically at runtime to match the USB mic's native rate
# (USB mics on Raspberry Pi typically only support 44100 Hz)
MIC_SAMPLE_RATE = None   # resolved in main() from device default
BUFFER_SAMPLES = None    # resolved after MIC_SAMPLE_RATE is set

# Global audio buffer (initialized in main after sample rate is known)
audio_buffer = None

def audio_callback(indata, frames, time_info, status):
    """
    Callback function called for each audio block from sounddevice.
    Rolls the global audio buffer and appends the new input data.
    """
    global audio_buffer
    # Keep callback ultra-fast: no print/stderr calls to prevent buffer overflow
    new_samples = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -len(new_samples))
    audio_buffer[-len(new_samples):] = new_samples

def make_ascii_bar(val, length=20):
    """
    Creates an ASCII progress bar.
    """
    filled_len = int(round(length * val))
    bar = "█" * filled_len + "░" * (length - filled_len)
    return bar

def clear_console():
    """
    Instant ANSI terminal clear: zero subprocess forks, zero latency, no freeze.
    """
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------
class CryStateMachine:
    """
    3-state machine: IDLE -> DETECTING -> DECIDE.

    Loudness gate: background RMS is tracked with a slow EMA updated only
    while in IDLE (i.e. not during an active detection).  A window must have
    RMS >= LOUD_RATIO * background_rms to be eligible for state transitions.

    DETECTING cancels (false alarm) if:
      - cry_prob stays below T_CRY for MAX_GAP_SEC continuously, OR
      - at end of DETECT_SEC: hits < MIN_HITS OR mean_cry_prob < MIN_AVG.

    DECIDE fires only on confirmed cries, holds for COOLDOWN_SEC, then IDLE.

    update() returns:
        state        : current state string
        alert_active : True only in DECIDE with a non-uncertain label
        label        : 'hunger' | 'discomfort' | 'uncertain' | None
        hits         : number of hit windows (cry_prob >= T_CRY) or None
        mean_cry     : mean cry_prob over ALL detecting windows or None
        ph_norm      : normalised p_hunger (DECIDE only) or None
        pd_norm      : normalised p_discomfort (DECIDE only) or None
        bg_rms       : current background RMS estimate
    """

    IDLE      = "IDLE"
    DETECTING = "DETECTING"
    DECIDE    = "DECIDE"

    # EMA coefficient for background RMS (small = slow adaptation)
    _BG_ALPHA = 0.05

    def __init__(self):
        self._state          = self.IDLE
        self._buf            = []    # list of (p_hunger, p_discomfort, cry_prob)
        self._detect_start   = None  # monotonic time DETECTING began
        self._detect_max_cry = 0.0   # max cry_prob seen during DETECTING
        self._gap_start      = None  # monotonic time last gap (cry_prob < T_CRY) started
        self._decide_start   = None  # monotonic time DECIDE began
        self._label          = None
        self._hits           = None
        self._mean_cry       = None
        self._ph_norm        = None
        self._pd_norm        = None
        self._bg_rms         = None  # lazily initialised on first window

    # ------------------------------------------------------------------
    @property
    def bg_rms(self):
        return self._bg_rms if self._bg_rms is not None else 0.0

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[SM {ts}] {msg}", flush=True)

    # ------------------------------------------------------------------
    def _reset_decide_fields(self):
        self._label    = None
        self._hits     = None
        self._mean_cry = None
        self._ph_norm  = None
        self._pd_norm  = None

    def _enter_idle(self, reason=""):
        self._log(f"-> IDLE  {reason}")
        self._state          = self.IDLE
        self._buf            = []
        self._detect_start   = None
        self._detect_max_cry = 0.0
        self._gap_start      = None
        self._decide_start   = None
        self._reset_decide_fields()

    def _enter_detecting(self, p_hunger, p_discomfort, cry_prob):
        self._log("-> DETECTING  (loudness gate passed, cry_prob >= T_CRY)")
        self._state          = self.DETECTING
        self._detect_start   = time.monotonic()
        self._detect_max_cry = cry_prob
        self._gap_start      = None  # no gap yet
        self._buf            = [(p_hunger, p_discomfort, cry_prob)]

    def _enter_decide(self, hits, mean_cry):
        """Compute label from hit windows only, then enter DECIDE."""
        hit_windows = [(ph, pd) for ph, pd, cp in self._buf if cp >= T_CRY]
        avg_ph = sum(ph for ph, pd in hit_windows) / len(hit_windows)
        avg_pd = sum(pd for ph, pd in hit_windows) / len(hit_windows)
        total  = avg_ph + avg_pd
        ph_norm = (avg_ph / total) if total > 0 else 0.5
        pd_norm = (avg_pd / total) if total > 0 else 0.5

        if abs(ph_norm - pd_norm) < MARGIN:
            label = "uncertain"
        elif ph_norm > pd_norm:
            label = "hunger"
        else:
            label = "discomfort"

        self._log(
            f"-> DECIDE  label={label}  hits={hits}  mean_cry={mean_cry:.3f}  "
            f"ph_norm={ph_norm:.3f}  pd_norm={pd_norm:.3f}"
        )
        self._state        = self.DECIDE
        self._decide_start = time.monotonic()
        self._label        = label
        self._hits         = hits
        self._mean_cry     = mean_cry
        self._ph_norm      = ph_norm
        self._pd_norm      = pd_norm

    # ------------------------------------------------------------------
    def update(self, cry_prob, p_hunger, p_discomfort, rms_energy):
        """
        Advance the machine by one prediction window.

        rms_energy : RMS of the raw audio window (for loudness gate).
        """
        now = time.monotonic()

        # --- Update background RMS estimate (IDLE only, slow EMA) ----
        if self._state == self.IDLE:
            if self._bg_rms is None:
                self._bg_rms = rms_energy          # first ever window
            else:
                self._bg_rms = (
                    (1 - self._BG_ALPHA) * self._bg_rms
                    + self._BG_ALPHA * rms_energy
                )

        # --- Loudness gate -------------------------------------------
        loud_threshold = LOUD_RATIO * self._bg_rms if self._bg_rms else 0.0
        passes_loudness = rms_energy >= loud_threshold

        # =================================================================
        if self._state == self.IDLE:
            if passes_loudness and cry_prob >= T_CRY:
                self._enter_detecting(p_hunger, p_discomfort, cry_prob)

        # =================================================================
        elif self._state == self.DETECTING:
            self._buf.append((p_hunger, p_discomfort, cry_prob))
            self._detect_max_cry = max(self._detect_max_cry, cry_prob)  # track peak
            elapsed = now - self._detect_start

            # --- Gap cancellation: abort if cry vanishes > MAX_GAP_SEC ---
            if cry_prob < T_CRY:
                if self._gap_start is None:
                    self._gap_start = now          # start timing the gap
                elif (now - self._gap_start) > MAX_GAP_SEC:
                    _gap_dur = now - self._gap_start
                    _hits_so_far = sum(1 for _, _, cp in self._buf if cp >= T_CRY)
                    _mean_so_far = sum(cp for _, _, cp in self._buf) / len(self._buf)
                    self._log(
                        f"DETECTING end  hits={_hits_so_far}  "
                        f"mean={_mean_so_far:.3f}  max={self._detect_max_cry:.3f}  "
                        f"result=FALSE_ALARM(gap={_gap_dur:.2f}s > MAX_GAP_SEC={MAX_GAP_SEC})"
                    )
                    self._enter_idle("(false alarm \u2014 gap too long)")
            else:
                self._gap_start = None             # gap reset; cry is back

            # --- End of detection window ---------------------------------
            if self._state == self.DETECTING and elapsed >= DETECT_SEC:
                hits     = sum(1 for _, _, cp in self._buf if cp >= T_CRY)
                mean_cry = sum(cp for _, _, cp in self._buf) / len(self._buf)
                max_cry  = self._detect_max_cry

                ok_hits = hits >= MIN_HITS
                ok_avg  = mean_cry >= MIN_AVG

                if ok_hits and ok_avg:
                    self._log(
                        f"DETECTING end  hits={hits}  mean={mean_cry:.3f}  "
                        f"max={max_cry:.3f}  result=CONFIRMED"
                    )
                    self._enter_decide(hits, mean_cry)
                else:
                    reasons = []
                    if not ok_hits:
                        reasons.append(f"hits={hits} < MIN_HITS={MIN_HITS}")
                    if not ok_avg:
                        reasons.append(f"mean={mean_cry:.3f} < MIN_AVG={MIN_AVG}")
                    self._log(
                        f"DETECTING end  hits={hits}  mean={mean_cry:.3f}  "
                        f"max={max_cry:.3f}  "
                        f"result=FALSE_ALARM({', '.join(reasons)})"
                    )
                    self._enter_idle("(false alarm \u2014 criteria not met)")

        # =================================================================
        elif self._state == self.DECIDE:
            if now - self._decide_start >= COOLDOWN_SEC:
                self._enter_idle("(cooldown done)")

        alert_active = (self._state == self.DECIDE and self._label != "uncertain")
        return {
            "state":        self._state,
            "alert_active": alert_active,
            "label":        self._label,
            "hits":         self._hits,
            "mean_cry":     self._mean_cry,
            "ph_norm":      self._ph_norm,
            "pd_norm":      self._pd_norm,
            "bg_rms":       self.bg_rms,
        }

def main():
    # Reconfigure standard output to UTF-8 to handle unicode progress bars on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # Load model (and optional scaler stats)
    if not os.path.exists(MODEL_TFLITE_PATH):
        print(f"Error: Missing model '{MODEL_TFLITE_PATH}'.")
        sys.exit(1)

    if os.path.exists(SCALER_PATH):
        print("Loading scaler statistics...")
        scaler = np.load(SCALER_PATH)

    print("Initializing TFLite Interpreter...")
    interpreter = None
    engines = [
        ("ai-edge-litert", "ai_edge_litert.interpreter"),
        ("tensorflow", "tensorflow.lite"),
        ("tflite-runtime", "tflite_runtime.interpreter")
    ]
    for engine_label, module_path in engines:
        try:
            mod = __import__(module_path, fromlist=["Interpreter"])
            candidate = mod.Interpreter(model_path=MODEL_TFLITE_PATH)
            candidate.allocate_tensors()
            interpreter = candidate
            print(f"Successfully loaded model using {engine_label}!")
            break
        except Exception as e:
            # Opcode version mismatch or missing engine
            continue

    if interpreter is None:
        print("\nError: Could not load the model with installed runtime.")
        print("Please install ai-edge-litert: pip install --index-url https://pypi.org/simple/ --user --break-system-packages ai-edge-litert")
        sys.exit(1)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # History list for temporal smoothing
    prediction_history = []
    consecutive_cry_frames = 0  # legacy debounce counter (kept for dashboard)

    # 3-State machine instance
    sm = CryStateMachine()

    # -----------------------------------------------------------------------
    # Logging setup: CSV file + per-window terminal log + IDLE summary
    # -----------------------------------------------------------------------
    os.makedirs("logs", exist_ok=True)
    run_ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join("logs", f"run_{run_ts}.csv")
    CSV_COLUMNS = [
        "timestamp", "rms", "bg_rms", "loud_gate",
        "p_hunger", "p_discomfort", "cry_prob",
        "state", "alert_active",
    ]
    _csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    _csv_writer = csv.DictWriter(_csv_file, fieldnames=CSV_COLUMNS)
    _csv_writer.writeheader()
    _csv_file.flush()
    print(f"[LOG] Writing per-window CSV to: {csv_path}")

    # IDLE-period summary tracker
    _idle_period_start   = time.monotonic()
    _idle_period_max_cry = 0.0
    _IDLE_SUMMARY_INTERVAL = 10.0  # seconds

    # Get available audio input devices & pick best microphone
    devices = sd.query_devices()
    print("\n--- Available Audio Input Devices ---")
    input_device_idx = None
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"[{idx}] {dev['name']} (Inputs: {dev['max_input_channels']}, Default SR: {dev['default_samplerate']}Hz)")
            if input_device_idx is None or "usb" in dev['name'].lower() or "mic" in dev['name'].lower():
                input_device_idx = idx

    # Use the device's NATIVE sample rate to avoid PortAudioError -9997
    # (USB mics on Raspberry Pi typically only support 44100 Hz, not 16000 Hz)
    global MIC_SAMPLE_RATE, BUFFER_SAMPLES, audio_buffer
    native_sr = int(devices[input_device_idx]['default_samplerate'])
    MIC_SAMPLE_RATE = native_sr
    BUFFER_SAMPLES = int(MIC_SAMPLE_RATE * DURATION)
    audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
    print(f"\nUsing native device sample rate: {MIC_SAMPLE_RATE} Hz")

    print(f"\nStarting real-time audio stream using device index [{input_device_idx}] @ {MIC_SAMPLE_RATE} Hz. Press Ctrl+C to stop.")
    
    # We read in blocks of 0.2s for smooth low-CPU processing
    blocksize = int(0.2 * MIC_SAMPLE_RATE)
    stream = sd.InputStream(
        device=input_device_idx,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=blocksize
    )
    
    with stream:
        # Give buffer a moment to fill with initial sound
        time.sleep(1.0)
        
        while True:
            # Copy local snapshot of the shared audio buffer
            y = np.copy(audio_buffer)
            
            # Compute RMS energy (more robust than max-amplitude for noise gating)
            rms_energy = np.sqrt(np.mean(y ** 2))
            max_amp = np.max(np.abs(y))
            
            # Noise floor gate: skip model if signal is too quiet (silence / very faint background)
            if rms_energy < NOISE_FLOOR:
                probs = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # default to "other" (silence)
                consecutive_cry_frames = 0  # reset debounce on silence
            else:
                # Preprocess the 3.0s audio buffer
                features = preprocess.extract_features_from_audio(y, sr=MIC_SAMPLE_RATE, duration=DURATION)
                
                # Apply CMS normalization
                features_scaled = preprocess.normalize_sample(features)
                
                # Prepare tensor for TFLite (Shape: (1, 40, 200, 3))
                input_tensor = np.expand_dims(features_scaled, axis=0).astype(np.float32)
                
                # Run inference
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                probs = interpreter.get_tensor(output_details[0]['index'])[0]

            # Apply temporal smoothing
            prediction_history.append(probs)
            if len(prediction_history) > SMOOTHING_WINDOW:
                prediction_history.pop(0)
            
            smoothed_probs = np.mean(prediction_history, axis=0)

            p_discomfort = smoothed_probs[0]
            p_hunger = smoothed_probs[1]
            p_other = smoothed_probs[2]
            
            # Combined cry probability
            cry_prob = p_discomfort + p_hunger
            
            # Determine cry type (which subclass is dominant)
            if p_discomfort > p_hunger:
                cry_type = "discomfort"
            else:
                cry_type = "hunger"
            
            # Legacy debounce counter (kept for dashboard display)
            if cry_prob >= CRY_THRESHOLD and rms_energy >= NOISE_FLOOR:
                consecutive_cry_frames += 1
            else:
                consecutive_cry_frames = 0

            # --- State machine update ---
            sm_out       = sm.update(cry_prob, p_hunger, p_discomfort, rms_energy)
            alert_active = sm_out["alert_active"]
            sm_state     = sm_out["state"]
            sm_label     = sm_out["label"]
            sm_hits      = sm_out["hits"]
            sm_mean_cry  = sm_out["mean_cry"]
            sm_ph_norm   = sm_out["ph_norm"]
            sm_pd_norm   = sm_out["pd_norm"]
            sm_bg_rms    = sm_out["bg_rms"]

            # ---------------------------------------------------------------
            # Per-window terminal log
            # ---------------------------------------------------------------
            _loud_thresh    = LOUD_RATIO * sm_bg_rms if LOUD_RATIO > 0 else 0.0
            _passes_loud    = (rms_energy >= _loud_thresh) if LOUD_RATIO > 0 else True
            _loud_gate_str  = "PASS" if _passes_loud else "FAIL"
            _win_ts         = time.strftime("%H:%M:%S")
            print(
                f"[WIN {_win_ts}] rms={rms_energy:.4f} bg={sm_bg_rms:.4f} "
                f"loud={_loud_gate_str} "
                f"ph={p_hunger:.3f} pd={p_discomfort:.3f} cry={cry_prob:.3f} "
                f"state={sm_state}",
                flush=True,
            )

            # ---------------------------------------------------------------
            # CSV row
            # ---------------------------------------------------------------
            _csv_writer.writerow({
                "timestamp":   _win_ts,
                "rms":         f"{rms_energy:.5f}",
                "bg_rms":      f"{sm_bg_rms:.5f}",
                "loud_gate":   _loud_gate_str,
                "p_hunger":    f"{p_hunger:.5f}",
                "p_discomfort": f"{p_discomfort:.5f}",
                "cry_prob":    f"{cry_prob:.5f}",
                "state":       sm_state,
                "alert_active": int(alert_active),
            })
            _csv_file.flush()

            # ---------------------------------------------------------------
            # IDLE-period max cry_prob summary (every 10 s)
            # ---------------------------------------------------------------
            if sm_state == CryStateMachine.IDLE:
                _idle_period_max_cry = max(_idle_period_max_cry, cry_prob)
                if time.monotonic() - _idle_period_start >= _IDLE_SUMMARY_INTERVAL:
                    print(
                        f"[IDLE {time.strftime('%H:%M:%S')}] "
                        f"last {_IDLE_SUMMARY_INTERVAL:.0f}s: max_cry={_idle_period_max_cry:.3f}",
                        flush=True,
                    )
                    _idle_period_start   = time.monotonic()
                    _idle_period_max_cry = 0.0
            else:
                # Reset the IDLE period clock whenever we leave IDLE
                _idle_period_start   = time.monotonic()
                _idle_period_max_cry = 0.0

            # Determine alert status string
            if alert_active:
                if sm_label == "hunger":
                    status_str = (
                        f"\033[92mALERT: CRY DETECTED \u2014 HUNGER"
                        f" (ph={sm_ph_norm*100:.0f}%  pd={sm_pd_norm*100:.0f}%"
                        f"  hits={sm_hits}  avg={sm_mean_cry*100:.0f}%)\033[0m"
                    )
                else:
                    status_str = (
                        f"\033[91mALERT: CRY DETECTED \u2014 DISCOMFORT"
                        f" (ph={sm_ph_norm*100:.0f}%  pd={sm_pd_norm*100:.0f}%"
                        f"  hits={sm_hits}  avg={sm_mean_cry*100:.0f}%)\033[0m"
                    )
            elif sm_state == CryStateMachine.DECIDE and sm_label == "uncertain":
                status_str = (
                    f"\033[93mALERT: CRY DETECTED \u2014 UNCERTAIN TYPE"
                    f" (ph={sm_ph_norm*100:.0f}%  pd={sm_pd_norm*100:.0f}%"
                    f"  hits={sm_hits}  avg={sm_mean_cry*100:.0f}%)\033[0m"
                )
            elif sm_state == CryStateMachine.DETECTING:
                elapsed_detect = time.monotonic() - sm._detect_start
                hits_so_far = sum(1 for _, _, cp in sm._buf if cp >= T_CRY)
                status_str = (
                    f"\033[94mDETECTING... "
                    f"{elapsed_detect:.1f}s/{DETECT_SEC:.0f}s  "
                    f"hits={hits_so_far}  cry_prob={cry_prob*100:.0f}%\033[0m"
                )
            elif p_other >= CRY_THRESHOLD:
                status_str = "\033[90mStatus: Ambient Noise / Environment sound\033[0m"
            else:
                status_str = f"\033[93mStatus: Listening... (Cry confidence: {cry_prob*100:.0f}%)\033[0m"

            # Draw CLI Dashboard
            clear_console()
            print("==================================================")
            print("      BABY CRY DETECTOR — TERMINAL DASHBOARD      ")
            print("==================================================")
            print(f"Capture Rate  : {MIC_SAMPLE_RATE} Hz (native) -> 8000 Hz (model) | Window: {DURATION}s")
            loud_thresh = LOUD_RATIO * sm_bg_rms
            print(f"RMS Energy    : {rms_energy:.4f}  BG RMS: {sm_bg_rms:.4f}  LoudGate: {loud_thresh:.4f}")
            print(f"SM State      : {sm_state:<10}  Consec.Frames: {consecutive_cry_frames}/{CONSECUTIVE_FRAMES_REQUIRED}")
            hits_str     = str(sm_hits)         if sm_hits     is not None else "--"
            mean_cry_str = f"{sm_mean_cry*100:.0f}%" if sm_mean_cry is not None else "--"
            print(f"Hits/Avg      : {hits_str}/{mean_cry_str}  (need >={MIN_HITS} hits, >={int(MIN_AVG*100)}% avg over {DETECT_SEC:.0f}s)")
            print("--------------------------------------------------")
            
            # Cry detection bar (combined)
            cry_color = "\033[91m" if alert_active else ("\033[93m" if cry_prob >= CRY_THRESHOLD else "")
            cry_reset = "\033[0m" if cry_color else ""
            cry_bar = make_ascii_bar(min(cry_prob, 1.0), length=20)
            print(f"{'Cry Detect':<12} : [{cry_bar}] {cry_color}{cry_prob*100:5.1f}%{cry_reset} (Threshold: {int(CRY_THRESHOLD*100)}%)")
            
            print("--------------------------------------------------")
            
            # Individual class probabilities
            for idx, cls in enumerate(CLASSES):
                prob = smoothed_probs[idx]
                raw_prob = probs[idx]
                bar = make_ascii_bar(prob, length=20)
                
                if cls == "hunger":
                    color_code = "\033[92m" if (alert_active and sm_label == "hunger") else ""
                elif cls == "discomfort":
                    color_code = "\033[91m" if (alert_active and sm_label == "discomfort") else ""
                else:
                    color_code = ""
                reset_code = "\033[0m" if color_code else ""
                
                print(f"{cls.capitalize():<12} : [{bar}] {color_code}{prob*100:5.1f}%{reset_code} (Raw: {raw_prob*100:4.1f}%)")
                
            print("--------------------------------------------------")
            print(f"Classification: {status_str}")
            print("==================================================")
            
            # Run classification twice per second (500ms sleep)
            time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping baby cry detector stream. Goodbye!")
        # Best-effort CSV close (file may not exist if startup failed early)
        try:
            _csv_file.close()
        except Exception:
            pass
        sys.exit(0)
