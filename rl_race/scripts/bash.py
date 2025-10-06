import subprocess

def run_command(command):
    try:
        # Run the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Print the output and error (if any)
        print("Output:\n", result.stdout.decode())
        if result.stderr:
            print("Error:\n", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr.decode()}")

if __name__ == "__main__":
    # Use single quotes for the entire string and escape inner double quotes
    command = 'ign service -s /world/shapes/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 300 --req \'name: "box", position: {z: 5.0}\''
    run_command(command)
