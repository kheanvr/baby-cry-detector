import os
import numpy as np
import tensorflow as tf
import librosa
from scipy import signal
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
                y, sr = librosa.load(file_path, sr=8000)
                raw_audio_list.append((y, sr))
                labels_list.append(CLASS_MAP[class_name])
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return raw_audio_list, np.array(labels_list, dtype=np.int32)

class ConfidentRobustGenerator(tf.keras.utils.Sequence):
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
            y = y.copy()
            
            # GENTLER waveform-level augmentations to prevent flat probabilities (underfitting)
            if self.augment:
                # 1. Random Gain (0.5 to 1.0)
                y = y * np.random.uniform(0.5, 1.0)
                
                # 2. Very mild highpass filter (cutoff 50 to 120 Hz) to keep vocal pitch intact
                if np.random.random() < 0.3:
                    cutoff = np.random.uniform(50, 120)
                    y = highpass_filter(y, sr, cutoff)
                    
                # 3. Mild noise injection (SNR 25 to 45 dB)
                if np.random.random() < 0.3:
                    snr = np.random.uniform(25, 45)
                    y = add_random_noise(y, snr)

            # Feature extraction (with zero-centering)
            feat = preprocess.extract_features_from_audio(y, sr)
            feat_norm = preprocess.normalize_sample(feat)
            batch_x.append(feat_norm)
        
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = self.labels[batch_indices]
        batch_x = np.expand_dims(batch_x, axis=-1)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    raw_audio, labels = load_raw_dataset()
    
    indices = np.arange(len(raw_audio))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=labels
    )
    
    raw_train = [raw_audio[i] for i in train_idx]
    labels_train = labels[train_idx]
    
    raw_val = [raw_audio[i] for i in val_idx]
    labels_val = labels[val_idx]
    
    train_gen = ConfidentRobustGenerator(raw_train, labels_train, batch_size=32, augment=True, shuffle=True)
    val_gen = ConfidentRobustGenerator(raw_val, labels_val, batch_size=32, augment=False, shuffle=False)
    
    model = baby_model.get_model(input_shape=(120, 200, 1), num_classes=3)
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training confident robust model...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop]
    )
    
    # Evaluate
    X_val = []
    for y, sr in raw_val:
        feat = preprocess.extract_features_from_audio(y, sr)
        feat_norm = preprocess.normalize_sample(feat)
        X_val.append(feat_norm)
    X_val = np.expand_dims(np.array(X_val, dtype=np.float32), axis=-1)
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n--- CLASSIFICATION REPORT (Confident Robust Model) ---")
    print(classification_report(labels_val, y_pred_classes, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(labels_val, y_pred_classes))
    
    # Check predictions above 80% threshold
    confidences = np.max(y_pred, axis=1)
    high_conf_mask = confidences >= 0.80
    high_conf_acc = np.mean(y_pred_classes[high_conf_mask] == labels_val[high_conf_mask])
    pct_high_conf = np.mean(high_conf_mask)
    
    print(f"\nExceeds 80% Threshold: {pct_high_conf*100:.1f}% of validation samples")
    print(f"Accuracy of >=80% predictions: {high_conf_acc*100:.1f}%")
    
    model.save("baby_cry_model_confident.h5")
    print("Saved model to 'baby_cry_model_confident.h5'")

if __name__ == "__main__":
    main()
