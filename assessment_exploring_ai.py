import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Define the CNN Architecture
model = models.Sequential([
    # Convolutional layer to detect fruit features (edges, textures)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening 2D feature maps into a 1D vector
    layers.Flatten(),
    
    # Fully connected layer for classification
    layers.Dense(64, activation='relu'),
    
    # Output layer: 0 (Apple) or 1 (Orange) using Sigmoid
    layers.Dense(1, activation='sigmoid')
])

# 2. Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'loss']
)

# 3. View the model structure
model.summary()