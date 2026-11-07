import tensorflow as tf
from tensorflow.keras import layers, models

def get_model(input_shape=(40, 200, 3), num_classes=3, output_logits=False):
    """
    Builds a deeper 2D CNN model for spectrogram audio classification.
    Optimized for confident predictions (80%+ threshold) on Raspberry Pi 5.
    
    Args:
        input_shape: Input spectrogram shape (n_mels, n_frames, n_channels)
        num_classes: Number of output classes
        output_logits: If True, outputs raw logits (no softmax). 
                       Used for training with from_logits=True loss and 
                       post-hoc temperature scaling.
    """
    model = models.Sequential()
    
    # Block 1: 40x200x3 -> 20x100x16
    model.add(layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Block 2: 20x100x16 -> 10x50x32
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Block 3: 10x50x32 -> 5x25x64
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Block 4: 5x25x64 -> 2x12x64 (reduces flatten dimension for efficiency)
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Flatten: 2x12x64 = 1,536 features
    model.add(layers.Flatten())
    
    # Classifier Dense Layer
    model.add(layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))
    
    # Output Layer
    if output_logits:
        # Raw logits output (for temperature scaling workflow)
        model.add(layers.Dense(num_classes))
    else:
        # Softmax probabilities (standard workflow)
        model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
