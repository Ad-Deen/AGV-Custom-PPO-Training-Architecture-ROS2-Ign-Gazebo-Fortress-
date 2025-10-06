import numpy as np

def load_and_print_npy(file_path):
    # Load the numpy array from the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Print the data
    print(data)
    print(data[101])

# Specify the path to your .npy file
file_path = '/home/deen/ros2_ws/src/rl_race/scripts/center_points.npy'

load_and_print_npy(file_path)
