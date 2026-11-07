import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

DATASET_PATH = r"C:\THESIS\MONO DATASET\dataset"
CLASSES = ["discomfort", "hunger", "other"]
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}

def extract_features_new(y, sr, target_sr=8000, duration=3.0, n_mfcc=40, target_frames=200):
    # Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    # Ensure mono
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
        
    # Pad or truncate
    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
        
    # Normalize volume
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    # Window settings (25ms window, 15ms hop, but 512 FFT size for better frequency resolution)
    win_length = int(0.025 * target_sr) # 200 samples
    hop_length = int(0.015 * target_sr) # 120 samples
    n_fft = 512
    
    # MFCC extraction (base 40)
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                                win_length=win_length, hop_length=hop_length, n_mels=40)
    
    # Delta and Delta-Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Apply CMS to base MFCCs BEFORE padding/truncating so padding remains exactly zero
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc = mfcc - mfcc_mean
    
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

def load_dataset():
    X = []
    y = []
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_PATH, class_name)
        file_names = [f for f in os.listdir(class_dir) if f.lower().endswith(".wav")]
        print(f"Loading {len(file_names)} files in class '{class_name}'...")
        for file_name in file_names:
            file_path = os.path.join(class_dir, file_name)
            try:
                # Load with librosa
                y_audio, sr = librosa.load(file_path, sr=8000)
                feat = extract_features_new(y_audio, sr=8000)
                X.append(feat)
                y.append(CLASS_MAP[class_name])
            except Exception as e:
                print(f"Error {file_name}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def get_3ch_model(input_shape=(40, 200, 3), num_classes=3):
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.GlobalAveragePooling2D())
    
    # Classifier Dense Layer
    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    X, y = load_dataset()
    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Global Coefficient-wise scaling (disabled to check performance)
    X_train_scaled = X_train
    X_val_scaled = X_val
    
    print(f"Global scaling disabled (raw CMS features used).")
    
    # Train
    model = get_3ch_model(input_shape=(40, 200, 3), num_classes=3)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training 3-channel model...")
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop]
    )
    
    # Evaluate
    y_pred_train = np.argmax(model.predict(X_train_scaled), axis=1)
    print("\n--- TRAINING CLASSIFICATION REPORT (3-Channel) ---")
    print(classification_report(y_train, y_pred_train, target_names=CLASSES))
    
    y_pred_val = np.argmax(model.predict(X_val_scaled), axis=1)
    print("\n--- VALIDATION CLASSIFICATION REPORT (3-Channel) ---")
    print(classification_report(y_val, y_pred_val, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_val))

if __name__ == "__main__":
    main()
