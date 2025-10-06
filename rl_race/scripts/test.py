#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

def generate_and_save_image():
    # Generate a random array
    data = np.random.rand(100, 100)  # 100x100 array of random numbers

    # Define the path to save the image
    save_path = 'random_image.png'

    # Save the array as a PNG image
    plt.imshow(data, cmap='gray')  # Use a grayscale colormap
    plt.colorbar()  # Add a colorbar for reference
    plt.title('Random Image')
    plt.savefig(save_path)

    # Show the image
    plt.show()

    print(f"Image saved as {save_path}")

if __name__ == "__main__":
    generate_and_save_image()
