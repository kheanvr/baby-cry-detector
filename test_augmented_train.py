import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess
import model as baby_model

class RobustAudioFeatureGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment=True, shuffle=True):
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
            # 1. Feature noise addition
            noise = np.random.normal(0, 0.05, size=batch_x.shape)
            batch_x += noise

            # 2. Random temporal shifts (rolling)
            for i in range(len(batch_x)):
                shift = np.random.randint(-10, 11)
                batch_x[i] = np.roll(batch_x[i], shift, axis=1)

            # 3. Channel distortion & low-frequency cutoff simulation
            for i in range(len(batch_x)):
                # Random spectral tilt: add a slope across the first 20 MFCCs (0-19)
                slope = np.random.uniform(-1.0, 1.0)
                tilt = np.linspace(slope, -slope, 20)
                batch_x[i, :20, :] += tilt[:, np.newaxis]
                
                # Random low-frequency attenuation/masking:
                # With 40% probability, zero out the first 3 MFCCs to simulate speaker cut-off
                if np.random.random() < 0.4:
                    batch_x[i, 0:3, :] = 0.0

        # Expand dimensions for Conv2D channel: (batch_size, 120, 200, 1)
        batch_x = np.expand_dims(batch_x, axis=-1)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    X = np.load("X_features.npy")
    y = np.load("y_labels.npy")
    print(f"Loaded features shape: {X.shape}, labels: {y.shape}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Apply CMS to base MFCCs
    print("Applying CMS...")
    X_train_scaled = np.array([preprocess.normalize_sample(x) for x in X_train], dtype=np.float32)
    X_val_scaled = np.array([preprocess.normalize_sample(x) for x in X_val], dtype=np.float32)

    # Data Generators
    train_gen = RobustAudioFeatureGenerator(X_train_scaled, y_train, batch_size=32, augment=True, shuffle=True)
    val_gen = RobustAudioFeatureGenerator(X_val_scaled, y_val, batch_size=32, augment=False, shuffle=False)

    # Initialize model
    model = baby_model.get_model(input_shape=(120, 200, 1), num_classes=3)
    
    # Train
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )

    print("Training robust model...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=60,
        callbacks=[early_stop]
    )

    # Evaluate validation
    X_val_exp = np.expand_dims(X_val_scaled, axis=-1)
    y_pred = model.predict(X_val_exp)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n--- CLASSIFICATION REPORT (Robust Training) ---")
    print(classification_report(y_val, y_pred_classes, target_names=["discomfort", "hunger", "other"]))

    print("--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_val, y_pred_classes))

    # Test with simulated low-frequency cutoff on the validation set!
    print("\n--- EVALUATING VALIDATION SET WITH SIMULATED SPEAKER CUTOFF (MFCC 0-3 zeroed) ---")
    X_val_cutoff = X_val_scaled.copy()
    X_val_cutoff[:, 0:4, :] = 0.0
    X_val_cutoff_exp = np.expand_dims(X_val_cutoff, axis=-1)
    
    y_pred_cutoff = model.predict(X_val_cutoff_exp)
    y_pred_cutoff_classes = np.argmax(y_pred_cutoff, axis=1)
    print(classification_report(y_val, y_pred_cutoff_classes, target_names=["discomfort", "hunger", "other"]))
    print(confusion_matrix(y_val, y_pred_cutoff_classes))

if __name__ == "__main__":
    main()
