import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Advanced Sequential CNN Architecture
model = models.Sequential([
    # First Block: Feature Extraction
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Block: Capturing complex textures (dimples vs. waxiness)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Block: High-level pattern recognition
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # Dense Layers with Dropout to prevent overfitting
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(1, activation='sigmoid') # Output: 0 for Apple, 1 for Orange
])

# 2. Compile with Adam Optimizer and Binary Crossentropy
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()