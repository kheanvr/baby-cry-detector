import os
import numpy as np
import tensorflow as tf
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

def extract_features_3ch(y, sr, target_sr=8000, duration=3.0, n_mfcc=40, target_frames=200):
    # Ensure mono and native sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
        
    # Remove DC offset (zero-center)
    y = y - np.mean(y)
        
    # Pad/truncate
    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
        
    # Normalize volume
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    # Window settings
    win_length = int(0.025 * target_sr) # 200 samples
    hop_length = int(0.015 * target_sr) # 120 samples
    n_fft = 512
    
    # MFCC extraction (base 40)
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                                win_length=win_length, hop_length=hop_length, n_mels=40)
    
    # Delta and Delta-Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Apply CMS to base MFCCs
    mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc = mfcc - mean
    
    # Force exact frame dimension
    if mfcc.shape[1] > target_frames:
        mfcc = mfcc[:, :target_frames]
        delta = delta[:, :target_frames]
        delta2 = delta2[:, :target_frames]
    elif mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
        delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        
    # Stack as 3 channels
    features_3ch = np.stack([mfcc, delta, delta2], axis=-1) # (40, 200, 3)
    return features_3ch

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

class ConfidentRobustGenerator3Ch(tf.keras.utils.Sequence):
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
            
            # Gentle augmentations
            if self.augment:
                # 1. Random Gain (0.5 to 1.0)
                y = y * np.random.uniform(0.5, 1.0)
                
                # 2. Very mild highpass filter (cutoff 50 to 120 Hz)
                if np.random.random() < 0.3:
                    cutoff = np.random.uniform(50, 120)
                    y = highpass_filter(y, sr, cutoff)
                    
                # 3. Mild noise injection (SNR 25 to 45 dB)
                if np.random.random() < 0.3:
                    snr = np.random.uniform(25, 45)
                    y = add_random_noise(y, snr)

            # Feature extraction
            feat_3ch = extract_features_3ch(y, sr)
            batch_x.append(feat_3ch)
        
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = self.labels[batch_indices]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def get_3ch_flatten_model(input_shape=(40, 200, 3), num_classes=3):
    model = tf.keras.models.Sequential()
    
    # Block 1
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Block 2
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Block 3
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Flatten
    model.add(tf.keras.layers.Flatten())
    
    # Classifier Dense Layer
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    
    # Output Layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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
    
    train_gen = ConfidentRobustGenerator3Ch(raw_train, labels_train, batch_size=32, augment=True, shuffle=True)
    val_gen = ConfidentRobustGenerator3Ch(raw_val, labels_val, batch_size=32, augment=False, shuffle=False)
    
    model = get_3ch_flatten_model(input_shape=(40, 200, 3), num_classes=3)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training 3-channel CNN with Flatten...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop]
    )
    
    # Evaluate
    X_val = []
    for y, sr in raw_val:
        feat_3ch = extract_features_3ch(y, sr)
        X_val.append(feat_3ch)
    X_val = np.array(X_val, dtype=np.float32)
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n--- CLASSIFICATION REPORT (3-Channel Stacked Flatten Model) ---")
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
    
    # Calculate recall at 80% threshold for each class
    print("\nRecall per class at 80% Threshold:")
    for idx, cls in enumerate(CLASSES):
        class_mask = labels_val == idx
        class_high_conf_correct = np.sum((y_pred_classes == idx) & (confidences >= 0.80) & class_mask)
        class_total = np.sum(class_mask)
        print(f"  {cls.capitalize():<12}: {class_high_conf_correct / class_total * 100:.1f}% ({class_high_conf_correct}/{class_total})")

if __name__ == "__main__":
    main()
