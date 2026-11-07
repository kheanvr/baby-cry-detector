import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess
import model as baby_model

def main():
    X = np.load("X_features.npy")
    y = np.load("y_labels.npy")
    print(f"Loaded features shape: {X.shape}, labels: {y.shape}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Apply CMS to base MFCCs
    print("Applying CMS...")
    X_train_cms = np.array([preprocess.normalize_sample(x) for x in X_train], dtype=np.float32)
    X_val_cms = np.array([preprocess.normalize_sample(x) for x in X_val], dtype=np.float32)

    # 2. Compute global channel-wise mean and std (over samples and time frames)
    # Shape of X_train_cms: (1005, 120, 200)
    # We want mean and std per coefficient (120 values)
    mean = np.mean(X_train_cms, axis=(0, 2), keepdims=True) # (1, 120, 1)
    std = np.std(X_train_cms, axis=(0, 2), keepdims=True)   # (1, 120, 1)
    
    # Avoid division by zero
    std[std < 1e-6] = 1.0

    X_train_scaled = (X_train_cms - mean) / std
    X_val_scaled = (X_val_cms - mean) / std

    print(f"Mean shape: {mean.shape}, min val: {mean.min():.4f}, max val: {mean.max():.4f}")
    print(f"Std shape: {std.shape}, min val: {std.min():.4f}, max val: {std.max():.4f}")

    # Train a model
    model = baby_model.get_model(input_shape=(120, 200, 1), num_classes=3)
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )

    print("Training with CMS + Global Standard Scaling...")
    # Expand dimensions for Conv2D
    X_train_exp = np.expand_dims(X_train_scaled, axis=-1)
    X_val_exp = np.expand_dims(X_val_scaled, axis=-1)

    model.fit(
        X_train_exp, y_train,
        validation_data=(X_val_exp, y_val),
        epochs=40,
        batch_size=32,
        callbacks=[early_stop]
    )

    # Evaluate train
    y_pred_train = np.argmax(model.predict(X_train_exp), axis=1)
    print("\n--- TRAINING CLASSIFICATION REPORT (CMS + Global Scale) ---")
    print(classification_report(y_train, y_pred_train, target_names=["discomfort", "hunger", "other"]))

    # Evaluate val
    y_pred_val = np.argmax(model.predict(X_val_exp), axis=1)
    print("\n--- VALIDATION CLASSIFICATION REPORT (CMS + Global Scale) ---")
    print(classification_report(y_val, y_pred_val, target_names=["discomfort", "hunger", "other"]))
    print(confusion_matrix(y_val, y_pred_val))

if __name__ == "__main__":
    main()
