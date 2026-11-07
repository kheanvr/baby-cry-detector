import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import preprocess
import model as baby_model

# Configuration
DATASET_PATH = r"C:\THESIS\MONO DATASET\dataset"
SCALER_PATH = "scaler_stats.npz"
MODEL_H5_PATH = "baby_cry_model.h5"
MODEL_TFLITE_PATH = "baby_cry_model.tflite"

CLASSES = ["discomfort", "hunger", "other"]
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}

def load_raw_dataset():
    """
    Loads raw audio files natively resampled to 8000Hz into memory.
    """
    raw_audio_list = []
    labels_list = []
    print("Loading raw audio files into memory...")
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
            
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

class CleanFeatureDataGenerator(tf.keras.utils.Sequence):
    """
    Clean data generator — NO augmentation.
    Previous experiments showed augmentation reduces confidence for cry classes
    because it makes the already-similar discomfort/hunger features harder
    to distinguish.
    """
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
            feat = preprocess.extract_features_from_audio(y, sr)
            feat_norm = preprocess.normalize_sample(feat)
            batch_x.append(feat_norm)
            
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = self.labels[batch_indices]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'.")

def main():
    # Load dataset
    try:
        raw_audio, labels = load_raw_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Train / Val Split
    indices = np.arange(len(raw_audio))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=labels
    )
    
    raw_train = [raw_audio[i] for i in train_idx]
    labels_train = labels[train_idx]
    raw_val = [raw_audio[i] for i in val_idx]
    labels_val = labels[val_idx]
    
    print(f"Train size: {len(raw_train)}, Validation size: {len(raw_val)}")

    # Save scaler stats for compatibility
    np.savez(SCALER_PATH, mean=np.zeros((40, 200, 3), dtype=np.float32), 
             std=np.ones((40, 200, 3), dtype=np.float32))
    print(f"Scaler statistics saved to '{SCALER_PATH}' (CMS Mode).")

    # Data Generators — NO augmentation
    train_gen = CleanFeatureDataGenerator(raw_train, labels_train, batch_size=32, shuffle=True)
    val_gen = CleanFeatureDataGenerator(raw_val, labels_val, batch_size=32, shuffle=False)

    # Build model with standard softmax output
    model = baby_model.get_model(input_shape=(40, 200, 3), num_classes=len(CLASSES), output_logits=False)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
    )
    
    # Train
    print("\n" + "="*60)
    print("  TRAINING (Standard Cross-Entropy, No Augmentation)")
    print("  LR=0.0005 | 4-Block CNN | Softmax Output")
    print("="*60 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=60,
        callbacks=[early_stop, reduce_lr]
    )

    # Evaluate on clean validation data
    print("\nEvaluating model...")
    X_val = []
    for y, sr in raw_val:
        feat = preprocess.extract_features_from_audio(y, sr)
        feat_norm = preprocess.normalize_sample(feat)
        X_val.append(feat_norm)
    X_val = np.array(X_val, dtype=np.float32)
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n--- CLASSIFICATION REPORT (Per-Class) ---")
    print(classification_report(labels_val, y_pred_classes, target_names=CLASSES))

    print("--- CONFUSION MATRIX ---")
    cm = confusion_matrix(labels_val, y_pred_classes)
    print(cm)

    # ===================================================================
    # PER-CLASS THRESHOLD ANALYSIS (Traditional)
    # ===================================================================
    confidences = np.max(y_pred, axis=1)
    high_conf_mask = confidences >= 0.80
    
    if np.sum(high_conf_mask) > 0:
        high_conf_acc = np.mean(y_pred_classes[high_conf_mask] == labels_val[high_conf_mask])
    else:
        high_conf_acc = 0.0
    pct_high_conf = np.mean(high_conf_mask)
    
    print(f"\n{'='*60}")
    print(f"  PER-CLASS 80% THRESHOLD ANALYSIS")
    print(f"{'='*60}")
    print(f"Exceeds 80% Threshold: {pct_high_conf*100:.1f}% ({np.sum(high_conf_mask)}/{len(labels_val)})")
    print(f"Accuracy of >=80% predictions: {high_conf_acc*100:.1f}%")
    
    print(f"\nRecall per class at 80% Threshold:")
    for idx, cls in enumerate(CLASSES):
        class_mask = labels_val == idx
        class_high_conf_correct = np.sum((y_pred_classes == idx) & (confidences >= 0.80) & class_mask)
        class_total = np.sum(class_mask)
        class_false_pos = np.sum((y_pred_classes == idx) & (confidences >= 0.80) & ~class_mask)
        print(f"  {cls.capitalize():<12}: {class_high_conf_correct / class_total * 100:.1f}% ({class_high_conf_correct}/{class_total}) | False Pos: {class_false_pos}")

    # ===================================================================
    # TWO-STAGE CRY DETECTION ANALYSIS
    # P(cry) = P(discomfort) + P(hunger)
    # If P(cry) >= 0.80 → CRY DETECTED, classify by max(P(d), P(h))
    # ===================================================================
    cry_probs = y_pred[:, 0] + y_pred[:, 1]  # P(discomfort) + P(hunger)
    other_probs = y_pred[:, 2]
    
    # Binary cry labels: 0 = discomfort (cry), 1 = hunger (cry), 2 = other (not cry)
    is_true_cry = labels_val < 2  # True if discomfort or hunger
    
    print(f"\n{'='*60}")
    print(f"  TWO-STAGE CRY DETECTION ANALYSIS")
    print(f"  P(cry) = P(discomfort) + P(hunger)")
    print(f"{'='*60}")
    
    for threshold in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        cry_detected = cry_probs >= threshold
        
        # True cry samples detected as cry
        true_pos = np.sum(cry_detected & is_true_cry)
        # Non-cry samples detected as cry (false alarm)
        false_pos = np.sum(cry_detected & ~is_true_cry)
        # True cry samples missed
        false_neg = np.sum(~cry_detected & is_true_cry)
        # Non-cry correctly not detected
        true_neg = np.sum(~cry_detected & ~is_true_cry)
        
        total_cry = np.sum(is_true_cry)
        total_other = np.sum(~is_true_cry)
        
        cry_recall = true_pos / total_cry * 100 if total_cry > 0 else 0
        cry_precision = true_pos / (true_pos + false_pos) * 100 if (true_pos + false_pos) > 0 else 0
        other_recall = true_neg / total_other * 100 if total_other > 0 else 0
        
        # Among detected cries, check type accuracy
        cry_type_mask = cry_detected & is_true_cry
        if np.sum(cry_type_mask) > 0:
            cry_type_pred = np.where(y_pred[cry_type_mask, 0] > y_pred[cry_type_mask, 1], 0, 1)
            cry_type_acc = np.mean(cry_type_pred == labels_val[cry_type_mask]) * 100
        else:
            cry_type_acc = 0
        
        print(f"\n  Threshold >= {int(threshold*100)}%:")
        print(f"    Cry Recall    : {cry_recall:.1f}% ({true_pos}/{total_cry})")
        print(f"    Cry Precision : {cry_precision:.1f}%")
        print(f"    False Alarms  : {false_pos}/{total_other} other samples")
        print(f"    Other Recall  : {other_recall:.1f}% ({true_neg}/{total_other})")
        print(f"    Type Accuracy : {cry_type_acc:.1f}% (hunger vs discomfort)")

    # Save training curves
    plot_history(history)

    # Save model
    model.save(MODEL_H5_PATH)
    print(f"\nSaved full model to '{MODEL_H5_PATH}'.")

    # Convert to TFLite
    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved optimized TFLite model to '{MODEL_TFLITE_PATH}' (Size: {os.path.getsize(MODEL_TFLITE_PATH) / 1024:.2f} KB).")

if __name__ == "__main__":
    main()
