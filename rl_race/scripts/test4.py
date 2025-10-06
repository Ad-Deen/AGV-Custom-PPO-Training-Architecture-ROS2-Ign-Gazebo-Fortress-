import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Build the CNN model using Sequential
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(72, 72, 1)),  # Define input shape explicitly
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# 2. Pass a dummy input to call the model and define its input shape
dummy_input = np.random.random((1, 72, 72, 1)).astype(np.float32)
_ = model(dummy_input)  # Call the model once to establish input/output shapes

# 3. Create a feature extractor model (extracting from the third MaxPooling2D layer)
feature_extractor = tf.keras.Model(inputs=model.inputs,
                                   outputs=model.get_layer(index=5).output)  # MaxPooling2D (2nd pool layer)

# 4. Function to visualize the feature maps
def visualize_feature_maps(feature_maps, num_cols=8):
    """
    Visualizes feature maps from a CNN layer.
    
    Parameters:
    - feature_maps: NumPy array of shape (1, height, width, channels)
    - num_cols: Number of columns in the grid for plotting
    """
    feature_maps = feature_maps[0]  # Remove batch dimension
    num_features = feature_maps.shape[-1]  # Number of feature maps (channels)
    
    # Determine grid size for plotting
    num_rows = num_features // num_cols
    if num_features % num_cols != 0:
        num_rows += 1

    # Create subplots for feature maps
    plt.figure(figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_features):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(feature_maps[:, :, i], cmap='viridis')  # Display each feature map
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    plt.tight_layout()
    plt.show()

# 5. Function to visualize the input image
def visualize_input(input_data):
    """
    Visualizes the input image.
    
    Parameters:
    - input_data: NumPy array of shape (1, height, width, channels)
    """
    input_image = input_data[0, :, :, 0]  # Remove batch and channel dimensions
    plt.figure(figsize=(6, 6))
    plt.imshow(input_image, cmap='gray')  # Display input image
    plt.axis('off')
    plt.title('Input Image')
    plt.show()

# 6. Continuously feed random inputs and visualize high-level features
while True:
    # Generate random input data: a single 72x72 occupancy grid
    input_data = np.random.randint(0, 2, size=(1, 72, 72, 1)).astype('float32')

    # Visualize the input image
    visualize_input(input_data)

    # Extract features
    features = feature_extractor.predict(input_data)

    # Visualize the extracted high-level feature maps
    visualize_feature_maps(features, num_cols=8)
    
    # Wait for 1 second before feeding the next random input
    time.sleep(1)
