import PyInstaller.__main__
import os
import platform

def create_working_executable():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the paths
    model_trainer_gui_path = os.path.join(current_dir, 'model_trainer_gui.py')
    metrics_weights_path = os.path.join(current_dir, 'metrics_weights.json')
    
    # Determine the correct separator based on OS
    separator = ';' if platform.system() == 'Windows' else ':'
    
    # Define PyInstaller arguments for a working build
    pyinstaller_args = [
        model_trainer_gui_path,  # Main script
        '--name=ModelTrainer',  # Name of the executable
        '--onefile',  # Create a single executable file
        '--windowed',  # Don't show console window
        f'--add-data={metrics_weights_path}{separator}.',  # Include metrics_weights.json
        '--clean',  # Clean PyInstaller cache
        '--noconfirm',  # Replace existing spec file
        # Include necessary packages
        '--hidden-import=PyQt5',
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=sklearn',
        '--hidden-import=sklearn.ensemble',
        '--hidden-import=sklearn.svm',
        '--hidden-import=sklearn.preprocessing',
        '--hidden-import=sklearn.metrics',
        '--hidden-import=sklearn.model_selection',
        '--hidden-import=xgboost',
        '--hidden-import=matplotlib',
        '--hidden-import=matplotlib.pyplot',
        '--hidden-import=seaborn',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=reportlab',
        '--hidden-import=reportlab.lib',
        '--hidden-import=reportlab.platypus',
        '--hidden-import=reportlab.lib.styles',
        '--hidden-import=reportlab.lib.pagesizes',
        '--hidden-import=reportlab.lib.units',
        '--hidden-import=reportlab.lib.colors',
        # Exclude unnecessary modules to reduce size
        '--exclude-module=matplotlib.tests',
        '--exclude-module=numpy.random.tests',
        '--exclude-module=scipy.tests',
        '--exclude-module=scipy.linalg.tests',
        '--exclude-module=scipy.sparse.tests',
        '--exclude-module=scipy.special.tests',
        '--exclude-module=scipy.stats.tests',
        '--exclude-module=scipy.optimize.tests',
        '--exclude-module=scipy.integrate.tests',
        '--exclude-module=scipy.signal.tests',
        '--exclude-module=scipy.spatial.tests',
        '--exclude-module=scipy.fft.tests',
        '--exclude-module=scipy.io.tests',
        '--exclude-module=scipy.interpolate.tests',
        '--exclude-module=scipy.ndimage.tests',
        '--exclude-module=scipy.cluster.tests',
        '--exclude-module=scipy.constants.tests',
        '--exclude-module=scipy.misc.tests',
        '--exclude-module=scipy.odr.tests',
        # Exclude unnecessary packages
        '--exclude-module=unittest',
        '--exclude-module=email',
        '--exclude-module=html',
        '--exclude-module=http',
        '--exclude-module=xml',
        '--exclude-module=pydoc',
        # Optimize imports
        '--strip',  # Strip symbols from binary
        '--noupx',  # Don't use UPX compression (can cause issues)
    ]
    
    # Run PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)
    
    print(f"\nBuild completed successfully!")
    print(f"Executable created in the 'dist' directory")
    print("\nTo run the application:")
    if platform.system() == "Windows":
        print("1. Double-click on ModelTrainer.exe in the 'dist' directory")
    else:
        print("1. Open terminal in the 'dist' directory")
        print("2. Run: ./ModelTrainer")

if __name__ == "__main__":
    create_working_executable() 