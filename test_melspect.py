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

def extract_features_melspect_3ch(y, sr, target_sr=8000, duration=3.0, n_mels=40, target_frames=200):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
        
    y = y - np.mean(y)
    
    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
        
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    win_length = int(0.025 * target_sr) # 200 samples
    hop_length = int(0.015 * target_sr) # 120 samples
    n_fft = 512
    
    # Extract Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=n_mels, n_fft=n_fft,
                                         win_length=win_length, hop_length=hop_length)
    # Convert to power-to-db (Log-Mel)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Delta and Delta2
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)
    
    # CMS on log_mel
    mean = np.mean(log_mel, axis=1, keepdims=True)
    log_mel_norm = log_mel - mean
    
    if log_mel_norm.shape[1] > target_frames:
        log_mel_norm = log_mel_norm[:, :target_frames]
        delta = delta[:, :target_frames]
        delta2 = delta2[:, :target_frames]
    elif log_mel_norm.shape[1] < target_frames:
        pad_width = target_frames - log_mel_norm.shape[1]
        log_mel_norm = np.pad(log_mel_norm, ((0, 0), (0, pad_width)), mode='constant')
        delta = np.pad(delta, ((0, 0), (0, pad_width)), mode='constant')
        delta2 = np.pad(delta2, ((0, 0), (0, pad_width)), mode='constant')
        
    return np.stack([log_mel_norm, delta, delta2], axis=-1)

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

class FastMelGenerator(tf.keras.utils.Sequence):
    def __init__(self, raw_audio, labels, batch_size=32, shuffle=True):
        self.raw_audio = raw_audio
        self.labels = labels
        self.batch_size = batch_size
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
            feat = extract_features_melspect_3ch(y, sr)
            batch_x.append(feat)
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
    model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
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
    
    train_gen = FastMelGenerator(raw_train, labels_train, batch_size=32, shuffle=True)
    val_gen = FastMelGenerator(raw_val, labels_val, batch_size=32, shuffle=False)
    
    model = get_3ch_flatten_model(input_shape=(40, 200, 3), num_classes=3)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training Log-Mel Stacked model...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop]
    )
    
    # Evaluate
    X_val = []
    for y, sr in raw_val:
        feat = extract_features_melspect_3ch(y, sr)
        X_val.append(feat)
    X_val = np.array(X_val, dtype=np.float32)
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n--- CLASSIFICATION REPORT (Log-Mel Stacked Model) ---")
    print(classification_report(labels_val, y_pred_classes, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(labels_val, y_pred_classes))
    
    confidences = np.max(y_pred, axis=1)
    high_conf_mask = confidences >= 0.80
    high_conf_acc = np.mean(y_pred_classes[high_conf_mask] == labels_val[high_conf_mask])
    pct_high_conf = np.mean(high_conf_mask)
    
    print(f"\nExceeds 80% Threshold: {pct_high_conf*100:.1f}% of validation samples")
    print(f"Accuracy of >=80% predictions: {high_conf_acc*100:.1f}%")
    
    print("\nRecall per class at 80% Threshold:")
    for idx, cls in enumerate(CLASSES):
        class_mask = labels_val == idx
        class_high_conf_correct = np.sum((y_pred_classes == idx) & (confidences >= 0.80) & class_mask)
        class_total = np.sum(class_mask)
        print(f"  {cls.capitalize():<12}: {class_high_conf_correct / class_total * 100:.1f}% ({class_high_conf_correct}/{class_total})")

if __name__ == "__main__":
    main()
