import PyInstaller.__main__
import os
import platform

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths
model_trainer_gui_path = os.path.join(current_dir, 'model_trainer_gui.py')
metrics_weights_path = os.path.join(current_dir, 'metrics_weights.json')

# Determine the correct separator based on OS
separator = ';' if platform.system() == 'Windows' else ':'

# Define PyInstaller arguments
pyinstaller_args = [
    model_trainer_gui_path,  # Main script
    '--name=ModelTrainer',  # Name of the executable
    '--onefile',  # Create a single executable file
    '--windowed',  # Don't show console window
    f'--add-data={metrics_weights_path}{separator}.',  # Include metrics_weights.json
    '--icon=NONE',  # No icon for now
    '--clean',  # Clean PyInstaller cache
    '--noconfirm',  # Replace existing spec file
]

# Run PyInstaller
PyInstaller.__main__.run(pyinstaller_args) 