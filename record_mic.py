import sounddevice as sd
import soundfile as sf
import time
import numpy as np

# Query default input device to find native hardware sample rate
try:
    default_device = sd.query_devices(kind='input')
    MIC_SAMPLE_RATE = int(default_device['default_samplerate'])
except Exception:
    MIC_SAMPLE_RATE = 16000  # fallback

print(f"Recording at {MIC_SAMPLE_RATE} Hz for 3 seconds...")
print("Please start playing the baby cry now...")
print("Recording in 3...")
time.sleep(1.0)
print("2...")
time.sleep(1.0)
print("1...")
time.sleep(1.0)
print("RECORDING STARTED...")

# Record
duration = 3.0  # seconds
recording = sd.rec(int(duration * MIC_SAMPLE_RATE), samplerate=MIC_SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()  # Wait for recording to finish
print("RECORDING FINISHED!")

# Check max amplitude
max_amp = np.max(np.abs(recording))
print(f"Max Amplitude Captured: {max_amp:.4f}")

# Normalize exactly like the preprocessing pipeline
if max_amp > 0:
    recording = recording / max_amp

# Save
output_filename = "mic_test.wav"
sf.write(output_filename, recording, MIC_SAMPLE_RATE)
print(f"Saved recording to '{output_filename}' in the current directory.")
print("Please transfer this file back to your laptop and listen to it to check for hiss, whine, or low volume!")
