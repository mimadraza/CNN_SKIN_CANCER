# main.py
import numpy as np
from ConvulationalLayer import ConvulationalLayer
import matplotlib.pyplot as plt

# Instantiate the ConvulationalLayer class
layer_one = ConvulationalLayer(num_filter=5, filter_size=3, stride=2, num_channels=3)

# Generate input data (32x32x3)
input_data = np.random.random((32, 32, 3))

# Perform forward propagation (result will be a 3D array: n x n x c)
result = layer_one.forward_prop(input_data)  # Shape: (n, n, c)

# Print the shape of the result
print("Result Shape:", result.shape)  # Example: (30, 30, 5)

# Extract dimensions
height, width, num_filters = result.shape

# Create subplots to visualize each filter's output
fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))

for i in range(num_filters):
    # Plot each channel (slice) as a heatmap
    cax = axes[i].imshow(result[:, :, i], cmap='viridis', interpolation='nearest')
    axes[i].set_title(f"Filter {i + 1}")
    axes[i].axis('off')  # Turn off axis

# Add a colorbar to the last subplot
fig.colorbar(cax, ax=axes, orientation='horizontal', fraction=0.05)

# Display the plots
plt.show()