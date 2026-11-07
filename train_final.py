import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess
import model as baby_model

CLASSES = ["discomfort", "hunger", "other"]
SCALER_PATH = "scaler_stats.npz"
MODEL_H5_PATH = "baby_cry_model.h5"
MODEL_TFLITE_PATH = "baby_cry_model.tflite"

def main():
    print("Loading pre-extracted features...")
    X = np.load("X_features.npy")
    y = np.load("y_labels.npy")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # Apply CMS
    print("Applying CMS...")
    X_cms = np.array([preprocess.normalize_sample(x) for x in X], dtype=np.float32)
    
    # Split (using the exact same split logic for consistency)
    X_train, X_val, y_train, y_val = train_test_split(
        X_cms, y, test_size=0.20, random_state=42, stratify=y
    )
    
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    
    # Initialize the 2D CNN with Flatten model
    model = baby_model.get_model(input_shape=(120, 200, 1), num_classes=3)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training final model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=[early_stop]
    )
    
    # Evaluate
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_val, y_pred_classes, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_classes))
    
    # Check predictions above 80% threshold
    confidences = np.max(y_pred, axis=1)
    high_conf_mask = confidences >= 0.80
    high_conf_acc = np.mean(y_pred_classes[high_conf_mask] == y_val[high_conf_mask])
    pct_high_conf = np.mean(high_conf_mask)
    
    print(f"\nExceeds 80% Threshold: {pct_high_conf*100:.1f}% of validation samples")
    print(f"Accuracy of >=80% predictions: {high_conf_acc*100:.1f}%")
    
    print("\nRecall per class at 80% Threshold:")
    for idx, cls in enumerate(CLASSES):
        class_mask = y_val == idx
        class_high_conf_correct = np.sum((y_pred_classes == idx) & (confidences >= 0.80) & class_mask)
        class_total = np.sum(class_mask)
        print(f"  {cls.capitalize():<12}: {class_high_conf_correct / class_total * 100:.1f}% ({class_high_conf_correct}/{class_total})")

    # Save scaling stats
    np.savez(SCALER_PATH, mean=np.zeros((120, 200), dtype=np.float32), std=np.ones((120, 200), dtype=np.float32))
    print(f"Scaler statistics saved to '{SCALER_PATH}' (CMS Mode).")

    # Save full model
    model.save(MODEL_H5_PATH)
    print(f"Saved full model to '{MODEL_H5_PATH}'.")

    # Convert to TFLite with Dynamic Range Quantization
    print("Converting model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved optimized TFLite model to '{MODEL_TFLITE_PATH}' (Size: {os.path.getsize(MODEL_TFLITE_PATH) / 1024:.2f} KB).")

if __name__ == "__main__":
    main()
