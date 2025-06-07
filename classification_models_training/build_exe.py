import PyInstaller.__main__
import os
import platform
import sys
import subprocess

def create_executable():
    if os.path.basename(os.getcwd()) == 'numpy':
        sys.exit("🚨 ERROR: No ejecutes este script desde el directorio fuente de numpy.")
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the paths
    model_trainer_gui_path = os.path.join(current_dir, 'model_trainer_gui.py')
    metrics_weights_path = os.path.join(current_dir, 'metrics_weights.json')
    
    # Determine the correct separator based on OS
    separator = ';' if platform.system() == 'Windows' else ':'
    
    # First, ensure we're in a clean environment
    print("Cleaning PyInstaller cache...")
    if os.path.exists('build'):
        import shutil
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    # Create spec file first
    spec_args = [
        '--name=ModelTrainer',
        '--onefile',
        '--windowed',
        f'--add-data={metrics_weights_path}{separator}.',
        '--clean',
        '--noconfirm',
        '--collect-submodules=numpy',
        '--collect-data=numpy',
        '--collect-submodules=pandas',
        '--collect-data=pandas',
        '--collect-submodules=xgboost',
        '--collect-data=xgboost',
        '--collect-binaries=xgboost',
        model_trainer_gui_path
    ]
    
    print("Creating spec file...")
    PyInstaller.__main__.run(spec_args)
    
    # Now modify the spec file to include all necessary imports
    spec_path = os.path.join(current_dir, 'ModelTrainer.spec')
    with open(spec_path, 'r') as f:
        spec_content = f.read()
    
    # Add hidden imports to the spec file
    hidden_imports = [
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        'numpy.linalg',
        'numpy.random',
        'numpy.random.mtrand',
        'pandas',
        'pandas._libs',
        'pandas.core',
        'pandas.core.algorithms',
        'pandas.core.arrays',
        'pandas.core.indexers',
        'pandas.core.indexes',
        'pandas.core.internals',
        'pandas.core.ops',
        'pandas.core.series',
        'pandas.core.tools',
        'pandas.core.util',
        'sklearn',
        'sklearn.ensemble',
        'sklearn.svm',
        'sklearn.preprocessing',
        'sklearn.metrics',
        'sklearn.model_selection',
        'xgboost',
        'matplotlib',
        'matplotlib.pyplot',
        'seaborn',
        'reportlab',
        'reportlab.lib',
        'reportlab.platypus',
        'reportlab.lib.styles',
        'reportlab.lib.pagesizes',
        'reportlab.lib.units',
        'reportlab.lib.colors',
        'imblearn',
        'imblearn.over_sampling',
    ]
    
    # Modify the spec file to include hidden imports
    hidden_imports_str = "hiddenimports=[" + ", ".join(f"'{imp}'" for imp in hidden_imports) + "],"
    spec_content = spec_content.replace("hiddenimports=[],", hidden_imports_str)
    
    # Write the modified spec file
    with open(spec_path, 'w') as f:
        f.write(spec_content)
    
    # Build using the spec file
    print("Building executable from spec file...")
    PyInstaller.__main__.run([spec_path, '--clean', '--noconfirm'])
    
    print(f"\nBuild completed successfully!")
    print(f"Executable created in the 'dist' directory")
    
    # Make the executable executable on Linux
    if platform.system() != "Windows":
        exe_path = os.path.join('dist', 'ModelTrainer')
        os.chmod(exe_path, 0o755)
        print(f"Made {exe_path} executable")

if __name__ == "__main__":
    create_executable()
