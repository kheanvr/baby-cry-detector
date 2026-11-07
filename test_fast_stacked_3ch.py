import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["discomfort", "hunger", "other"]

def get_3ch_flatten_model(input_shape=(40, 200, 3), num_classes=3):
    model = tf.keras.models.Sequential()
    
    # Block 1
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Block 2
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Block 3
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
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
    
    # Stack features into 3 channels in-memory and apply CMS to MFCCs
    print("Stacking features into 3 channels and applying CMS...")
    mfcc = X[:, 0:40, :]
    delta = X[:, 40:80, :]
    delta2 = X[:, 80:120, :]
    
    # CMS on base MFCCs
    mean = np.mean(mfcc, axis=2, keepdims=True)
    mfcc_norm = mfcc - mean
    
    X_3ch = np.stack([mfcc_norm, delta, delta2], axis=-1).astype(np.float32)
    print(f"Stacked 3D shape: {X_3ch.shape}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_3ch, y, test_size=0.20, random_state=42, stratify=y
    )
    
    model = get_3ch_flatten_model(input_shape=(40, 200, 3), num_classes=3)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print("Training 3-channel Flatten model fast...")
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
    
    print("\n--- CLASSIFICATION REPORT (Fast 3-Channel Flatten Model) ---")
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
