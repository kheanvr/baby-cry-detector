# Baby Cry Detector (Dunstan Baby Language Recognition System)

This repository contains a deep learning system designed for a Raspberry Pi 5 to classify infant pre-cry sounds into two critical states from the Dunstan Baby Language (DBL) system: **Hunger** (vocal reflex "Neh") and **Discomfort** (vocal reflex "Heh"), alongside a baseline class for environmental noises or other sounds.

The system is split into two phases:
1. **Model Training & Conversion** (performed on a Windows PC to leverage high CPU/GPU compute).
2. **Real-Time Inference Dashboard** (deployed to a Raspberry Pi 5 with a USB microphone).

---

## 1. Theoretical Background & Signal Processing Pipeline

### Mel-Frequency Cepstral Coefficients (MFCCs)
Raw audio signals are rapidly changing in time and are difficult for standard neural networks to classify directly. In speech and audio recognition, **MFCCs** are the gold standard because they represent the power spectrum of an audio signal based on the **Mel Scale**, which mimics the non-linear human perception of pitch.

The feature extraction pipeline operates as follows:
1. **Preprocessing & Standardization**: Audio is resampled to **16,000 Hz, mono**, and volume is amplitude-normalized to a range of $[-1.0, 1.0]$.
2. **Framing**: The audio is divided into overlapping short frames to capture transient sounds. We use a window size of **25ms** (400 samples at 16kHz) and an overlap of **10ms** (hop size of 15ms / 240 samples).
3. **Short-Time Fourier Transform (STFT)**: Converts each frame from the time domain to the frequency domain.
4. **Mel Filterbank Processing**: The spectrum is passed through 40 triangular bandpass filters spaced along the Mel frequency scale.
5. **Log-Amplitude & DCT**: Taking the logarithm of the Mel spectrum followed by the Discrete Cosine Transform (DCT) yields **40 static MFCC features** per frame.
6. **Temporal Dynamics ($\Delta$ and $\Delta\Delta$)**:
   - First-order derivatives ($\Delta$) represent the velocity of change of MFCCs.
   - Second-order derivatives ($\Delta\Delta$) represent the acceleration of change.
   - Concatenating MFCC, $\Delta$, and $\Delta\Delta$ creates a **120-coefficient feature vector** per time frame.
7. **Fixed Feature Tensor**: For a 3.0-second clip, this outputs a shape of exactly **(120, 200)**.

---

## 2. Model Architecture

The model is a lightweight **2D Convolutional Neural Network (CNN)**. Spectrogram-like features are treated as single-channel images of shape `(120, 200, 1)`.

```
                  ┌───────────────────────────┐
                  │   Input: (120, 200, 1)    │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Conv2D (16 filters, 3x3)  │
                  ├───────────────────────────┤
                  │    Batch Normalization    │
                  ├───────────────────────────┤
                  │           ReLU            │
                  ├───────────────────────────┤
                  │    MaxPooling2D (2x2)     │
                  ├───────────────────────────┤
                  │       Dropout (25%)       │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Conv2D (32 filters, 3x3)  │
                  ├───────────────────────────┤
                  │    Batch Normalization    │
                  ├───────────────────────────┤
                  │           ReLU            │
                  ├───────────────────────────┤
                  │    MaxPooling2D (2x2)     │
                  ├───────────────────────────┤
                  │       Dropout (25%)       │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ Conv2D (64 filters, 3x3)  │
                  ├───────────────────────────┤
                  │    Batch Normalization    │
                  ├───────────────────────────┤
                  │           ReLU            │
                  ├───────────────────────────┤
                  │ GlobalAveragePooling2D    │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │      Dense (64 units)     │
                  ├───────────────────────────┤
                  │    Batch Normalization    │
                  ├───────────────────────────┤
                  │           ReLU            │
                  ├───────────────────────────┤
                  │       Dropout (50%)       │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │    Dense (3 units, Soft)  │
                  └───────────────────────────┘
```

---

## 3. Real-Time Inference & Privacy Architecture

The inference pipeline on the Raspberry Pi 5 runs inside `inference.py` using a strictly **in-memory rolling buffer**:

- **No Disk Logging**: Incoming microphone bytes are written directly to a rolling circular NumPy array in RAM. Old audio samples are shifted out as new ones arrive. No audio clips are saved to disk, preserving privacy.
- **Noise Gate**: A noise gate evaluates the maximum amplitude of the buffer. If the room is silent (amplitude $< 0.005$), feature extraction is bypassed, outputting 100% "other" class (silence) to prevent the amplification of noise.
- **Classification Threshold**: An **80% (0.80)** softmax probability is required to trigger a warning for hunger or discomfort.
- **Temporal Smoothing**: A rolling average of the last 3 prediction frames is computed before displaying probabilities, eliminating high-frequency UI flickering.

---

## 4. Setup & Running Instructions

### Stage 1: Training on Windows PC
1. Put your audio clips in `C:\THESIS\MONO DATASET\dataset\` in the respective folders:
   - `discomfort/`
   - `hunger/`
   - `other/`
2. Open a command terminal inside the project directory `C:\THESIS\baby-cry-detector\`.
3. Install dependencies:
   ```bash
   pip install -r requirements_train.txt
   ```
4. Run the training script:
   ```bash
   python train.py
   ```
   This will:
   - Load the audio dataset, extract features, and cache them to `X_features.npy` / `y_labels.npy`.
   - Compute and save standard scaling coefficients as `scaler_stats.npz`.
   - Train the CNN, save training performance figures, and output the model files: `baby_cry_model.h5` and `baby_cry_model.tflite`.

### Stage 2: Deployment on Raspberry Pi 5
1. Copy the following files from your PC to the Raspberry Pi 5:
   - `inference.py`
   - `preprocess.py`
   - `baby_cry_model.tflite`
   - `scaler_stats.npz`
   - `requirements_rpi.txt`
2. In the Raspberry Pi terminal, install OS-level sound dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y portaudio19-dev libsndfile1
   ```
3. Install Python requirements:
   ```bash
   pip install -r requirements_rpi.txt
   ```
4. Plug in your USB microphone and run the real-time detector:
   ```bash
   python inference.py
   ```
