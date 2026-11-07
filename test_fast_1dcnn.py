import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import preprocess

CLASSES = ["discomfort", "hunger", "other"]

def get_1dcnn_model(input_shape=(200, 120), num_classes=3):
    model = tf.keras.models.Sequential()
    
    # Block 1
    model.add(tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Block 2
    model.add(tf.keras.layers.Conv1D(128, kernel_size=5, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Block 3
    model.add(tf.keras.layers.Conv1D(128, kernel_size=5, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    
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
    print("Loading pre-extracted features...")
    X = np.load("X_features.npy")
    y = np.load("y_labels.npy")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # Apply CMS
    print("Applying CMS...")
    X_cms = np.array([preprocess.normalize_sample(x) for x in X], dtype=np.float32)
    
    # Reshape to (N, 200, 120) for 1D CNN (time steps, features)
    X_1d = np.transpose(X_cms, (0, 2, 1))
    print(f"Reshaped for 1D CNN: {X_1d.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_1d, y, test_size=0.20, random_state=42, stratify=y
    )
    
    model = get_1dcnn_model(input_shape=(200, 120), num_classes=3)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training 1D CNN model fast...")
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
    
    print("\n--- CLASSIFICATION REPORT (1D CNN Model) ---")
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

if __name__ == "__main__":
    main()
