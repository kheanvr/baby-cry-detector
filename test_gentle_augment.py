import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess
import model as baby_model

DATASET_PATH = r"C:\THESIS\MONO DATASET\dataset"
CLASSES = ["discomfort", "hunger", "other"]
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}

def load_and_preprocess_dataset():
    print("Extracting features using updated preprocess module (with zero-centering)...")
    X = []
    y = []
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_PATH, class_name)
        file_names = [f for f in os.listdir(class_dir) if f.lower().endswith(".wav")]
        print(f"Processing {len(file_names)} files in '{class_name}'...")
        for file_name in file_names:
            file_path = os.path.join(class_dir, file_name)
            features = preprocess.extract_features_from_file(file_path)
            if features is not None:
                # Apply CMS
                features_norm = preprocess.normalize_sample(features)
                X.append(features_norm)
                y.append(CLASS_MAP[class_name])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

class GentleFeatureGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment=False, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.X[batch_indices].copy()
        batch_y = self.y[batch_indices]

        if self.augment:
            # Gentle feature noise
            noise = np.random.normal(0, 0.03, size=batch_x.shape)
            batch_x += noise
            # Temporal rolling
            for i in range(len(batch_x)):
                shift = np.random.randint(-8, 9)
                batch_x[i] = np.roll(batch_x[i], shift, axis=1)

        batch_x = np.expand_dims(batch_x, axis=-1)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    X, y = load_and_preprocess_dataset()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    train_gen = GentleFeatureGenerator(X_train, y_train, batch_size=32, augment=True, shuffle=True)
    val_gen = GentleFeatureGenerator(X_val, y_val, batch_size=32, augment=False, shuffle=False)
    
    model = baby_model.get_model(input_shape=(120, 200, 1), num_classes=3)
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training model with gentle augmentations...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop]
    )
    
    # Evaluate
    X_val_exp = np.expand_dims(X_val, axis=-1)
    y_pred = model.predict(X_val_exp)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n--- CLASSIFICATION REPORT (Gentle Augment Model) ---")
    print(classification_report(y_val, y_pred_classes, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_classes))
    
    # Check what % of predictions exceed the 80% threshold
    confidences = np.max(y_pred, axis=1)
    high_conf_mask = confidences >= 0.80
    high_conf_acc = np.mean(y_pred_classes[high_conf_mask] == y_val[high_conf_mask])
    pct_high_conf = np.mean(high_conf_mask)
    
    print(f"\nExceeds 80% Threshold: {pct_high_conf*100:.1f}% of validation samples")
    print(f"Accuracy of >=80% predictions: {high_conf_acc*100:.1f}%")

if __name__ == "__main__":
    main()
