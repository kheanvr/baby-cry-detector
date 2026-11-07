import os
import numpy as np
import librosa
from scipy import signal
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess
import model as baby_model

DATASET_PATH = r"C:\THESIS\MONO DATASET\dataset"
CLASSES = ["discomfort", "hunger", "other"]
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}

def highpass_filter(y, sr, cutoff):
    try:
        sos = signal.butter(4, cutoff, 'hp', fs=sr, output='sos')
        return signal.sosfilt(sos, y)
    except Exception:
        return y

def add_random_noise(y, snr_db):
    sig_power = np.mean(y**2)
    noise_power = sig_power / (10**(snr_db / 10.0) + 1e-6)
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def extract_features_robust(y, sr, target_sr=8000, duration=3.0, n_mfcc=40, target_frames=200, augment=False):
    # Ensure 8000Hz
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
        
    # Waveform-level Augmentations (Only during training)
    if augment:
        # 1. Random Gain (volume scaling)
        y = y * np.random.uniform(0.3, 1.0)
        
        # 2. Random High-pass Filter (simulates speaker/mic low-frequency roll-off)
        if np.random.random() < 0.5:
            cutoff = np.random.uniform(100, 350)
            y = highpass_filter(y, sr, cutoff)
            
        # 3. Random Microphone Hiss (Noise SNR 15 to 35 dB)
        if np.random.random() < 0.5:
            snr = np.random.uniform(15, 35)
            y = add_random_noise(y, snr)

    # Pad or truncate
    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
        
    # Amplitude normalization (must happen after noise/gain/filter to ensure standard range)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    # Window settings (25ms window, 15ms hop)
    win_length = int(0.025 * target_sr) # 200 samples
    hop_length = int(0.015 * target_sr) # 120 samples
    n_fft = 512
    
    # MFCC extraction (base 40)
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                                win_length=win_length, hop_length=hop_length, n_mels=40)
    
    # Delta and Delta-Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Concatenate features
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    
    # Force exact frame dimension
    if features.shape[1] > target_frames:
        features = features[:, :target_frames]
    elif features.shape[1] < target_frames:
        features = np.pad(features, ((0, 0), (0, target_frames - features.shape[1])), mode='constant')
        
    # Apply CMS to base MFCCs (0-39)
    mean = np.mean(features[:40], axis=1, keepdims=True)
    features[:40] = features[:40] - mean
    
    return features

# Pre-load audio to RAM to make generator super fast
def load_raw_dataset():
    raw_audio_list = []
    labels_list = []
    print("Loading raw audio files into memory...")
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_PATH, class_name)
        file_names = [f for f in os.listdir(class_dir) if f.lower().endswith(".wav")]
        print(f"Loading {len(file_names)} files for class '{class_name}'...")
        for file_name in file_names:
            file_path = os.path.join(class_dir, file_name)
            try:
                # Load raw audio natively
                y, sr = librosa.load(file_path, sr=8000)
                raw_audio_list.append((y, sr))
                labels_list.append(CLASS_MAP[class_name])
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return raw_audio_list, np.array(labels_list, dtype=np.int32)

class RobustWaveformGenerator(tf.keras.utils.Sequence):
    def __init__(self, raw_audio, labels, batch_size=32, augment=True, shuffle=True):
        self.raw_audio = raw_audio
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.raw_audio))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.raw_audio) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        for i in batch_indices:
            y, sr = self.raw_audio[i]
            feat = extract_features_robust(y, sr, augment=self.augment)
            batch_x.append(feat)
        
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = self.labels[batch_indices]
        
        # Shape for Conv2D: (batch_size, 120, 200, 1)
        batch_x = np.expand_dims(batch_x, axis=-1)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    raw_audio, labels = load_raw_dataset()
    
    # Split
    indices = np.arange(len(raw_audio))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=labels
    )
    
    raw_train = [raw_audio[i] for i in train_idx]
    labels_train = labels[train_idx]
    
    raw_val = [raw_audio[i] for i in val_idx]
    labels_val = labels[val_idx]
    
    print(f"Train samples: {len(raw_train)}, Val samples: {len(raw_val)}")
    
    train_gen = RobustWaveformGenerator(raw_train, labels_train, batch_size=32, augment=True, shuffle=True)
    val_gen = RobustWaveformGenerator(raw_val, labels_val, batch_size=32, augment=False, shuffle=False)
    
    # Initialize model
    model = baby_model.get_model(input_shape=(120, 200, 1), num_classes=3)
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training robust model with waveform augmentations...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop]
    )
    
    # Evaluate Validation
    val_gen_no_aug = RobustWaveformGenerator(raw_val, labels_val, batch_size=32, augment=False, shuffle=False)
    X_val = []
    for y, sr in raw_val:
        X_val.append(extract_features_robust(y, sr, augment=False))
    X_val = np.expand_dims(np.array(X_val, dtype=np.float32), axis=-1)
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n--- CLASSIFICATION REPORT (Waveform Robust Model) ---")
    print(classification_report(labels_val, y_pred_classes, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(labels_val, y_pred_classes))
    
    # Save the model
    model.save("baby_cry_model_robust.h5")
    print("Saved model to 'baby_cry_model_robust.h5'")

if __name__ == "__main__":
    main()
