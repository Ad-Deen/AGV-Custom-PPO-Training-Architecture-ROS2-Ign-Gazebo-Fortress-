import numpy as np
import os

# Create a random NumPy array
random_array = np.random.rand(10, 10)

# Define the path to save the NumPy array
save_path = os.path.expanduser('~/ros2_ws/src/rl_race/scripts/random_array.npy')

# Save the NumPy array
np.save(save_path, random_array)

print(f"Array saved as {save_path}")

# Verify that the file was saved
if os.path.exists(save_path):
    print(f"File {save_path} exists.")
else:
    print(f"File {save_path} does not exist.")
