import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess
import model as baby_model

# We will crop the features to only use MFCC 4-39, Delta 4-39, and Delta-Delta 4-39.
# This corresponds to rows 4-39, 44-79, and 84-119 from the 120-row feature matrix.
def crop_features(X):
    # X shape: (N, 120, 200)
    indices = []
    # MFCC 4-39
    indices.extend(range(4, 40))
    # Delta 4-39
    indices.extend(range(44, 80))
    # Delta-Delta 4-39
    indices.extend(range(84, 120))
    return X[:, indices, :]

def main():
    X = np.load("X_features.npy")
    y = np.load("y_labels.npy")
    print(f"Original features shape: {X.shape}, labels: {y.shape}")

    # Crop to discard first 4 coefficients of MFCC, Delta, Delta-Delta (new height: 108)
    X_cropped = crop_features(X)
    print(f"Cropped features shape: {X_cropped.shape}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_cropped, y, test_size=0.20, random_state=42, stratify=y
    )

    # Apply CMS to base MFCCs (first 36 rows of the cropped features)
    print("Applying CMS...")
    def apply_cms_cropped(feat):
        feat_normalized = feat.copy()
        # feat has shape (108, 200)
        # first 36 rows are MFCC 4-39
        mean = np.mean(feat[:36], axis=1, keepdims=True)
        feat_normalized[:36] = feat[:36] - mean
        return feat_normalized

    X_train_scaled = np.array([apply_cms_cropped(x) for x in X_train], dtype=np.float32)
    X_val_scaled = np.array([apply_cms_cropped(x) for x in X_val], dtype=np.float32)

    # Define model with input shape (108, 200, 1)
    model = baby_model.get_model(input_shape=(108, 200, 1), num_classes=3)
    
    # Train
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )

    print("Training model without low-frequency MFCCs...")
    # Expand dimensions for Conv2D
    X_train_exp = np.expand_dims(X_train_scaled, axis=-1)
    X_val_exp = np.expand_dims(X_val_scaled, axis=-1)

    model.fit(
        X_train_exp, y_train,
        validation_data=(X_val_exp, y_val),
        epochs=60,
        batch_size=32,
        callbacks=[early_stop]
    )

    # Evaluate validation
    y_pred = model.predict(X_val_exp)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n--- CLASSIFICATION REPORT (No Low MFCCs) ---")
    print(classification_report(y_val, y_pred_classes, target_names=["discomfort", "hunger", "other"]))

    print("--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_val, y_pred_classes))

if __name__ == "__main__":
    main()
