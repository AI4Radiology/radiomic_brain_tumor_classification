import tkinter as tk
from tkinter import filedialog, scrolledtext
import os
import logging
import sys
import pandas as pd

from flair_reformatter import FlairReformatter
from model_trainer import ModelTrainer

class TextRedirector(object):
    """A class to redirect stdout and stderr to a tkinter Text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str_to_write):
        self.widget.configure(state='normal')  # Enable writing
        self.widget.insert(tk.END, str_to_write, self.tag)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')  # Disable writing again

    def flush(self):
        pass

class ModelTrainerApp:
    """GUI application for model training and evaluation."""
    def __init__(self, master):
        self.master = master
        master.title("Model Trainer")

        self.data_dir = tk.StringVar()
        self.output_dir = "data"  # Directory for prepared_data.csv
        self.resources_dir = "resources"  # Directory for zip archive
        self.report_dir = "report"  # Directory for PDF reports
        self.last_report_path = None # To store the path of the last generated report

        # Initialize the reformatter and trainer
        self.reformatter = FlairReformatter("", self.output_dir)
        self.trainer = ModelTrainer(self.output_dir, self.resources_dir, self.report_dir)

        # GUI elements
        self.label = tk.Label(master, text="FLAIR Data Directory:")
        self.label.pack()

        self.entry = tk.Entry(master, textvariable=self.data_dir, width=50)
        self.entry.pack()

        self.browse_button = tk.Button(master, text="Browse", command=self.browse_data_directory)
        self.browse_button.pack()

        self.train_button = tk.Button(master, text="Train Models", command=self.start_training)
        self.train_button.pack()

        self.view_report_button = tk.Button(master, text="Ver Reporte", command=self.open_report, state=tk.DISABLED)
        self.view_report_button.pack()

        self.log_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, height=20, width=80, state='disabled')
        self.log_text.pack()

        # Redirect stdout and stderr to the text widget
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")

        # Ensure necessary directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.resources_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def browse_data_directory(self):
        """Opens a dialog to select the FLAIR data directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir.set(directory)
            print(f"Selected data directory: {directory}")

    def start_training(self):
        """Starts the model training pipeline in a separate thread to keep GUI responsive."""
        data_path = self.data_dir.get()
        prepared_data_path = os.path.join(self.output_dir, "prepared_data.csv")
        
        # Check if we have either a data path or prepared_data.csv
        if not data_path and not os.path.exists(prepared_data_path):
            print("Por favor seleccione un directorio de datos FLAIR o asegúrese de que existe el archivo prepared_data.csv.")
            return
        
        print("Starting model training pipeline...")
        # Disable buttons during training
        self.train_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)

        # Run the training in a separate thread
        import threading
        training_thread = threading.Thread(target=self._run_training_pipeline, args=(data_path,))
        training_thread.start()

    def _run_training_pipeline(self, data_path):
        """Private method to run the entire training pipeline."""
        try:
            prepared_data_path = os.path.join(self.output_dir, "prepared_data.csv")
            
            # Step 1: Run FLAIR reformatting if prepared_data.csv doesn't exist
            if not os.path.exists(prepared_data_path):
                if not data_path:
                    print("Error: No se proporcionó un directorio de datos FLAIR y no existe prepared_data.csv.")
                    return
                    
                print("Prepared data not found. Running FLAIR reformatting...")
                self.reformatter.data_dir = data_path
                column_names = self.reformatter.process_data()
                if column_names is None:
                    print("FLAIR reformatting failed. Aborting training.")
                    return
            else:
                print("Using existing prepared_data.csv.")
                # Load column names if prepared_data.csv already exists
                df_prepared = pd.read_csv(prepared_data_path)
                column_names = df_prepared.drop(columns=["Id", "highGrade"], errors='ignore').columns.tolist()

            # Step 2: Load prepared data
            df = self.trainer.load_data(prepared_data_path)
            if df is None:
                print("Failed to load prepared data. Aborting training.")
                return

            # Step 3: Train and save models
            best_model_name, best_model, metrics, pdf_path = self.trainer.train_and_save_models(df, column_names)
            print("\nModel training pipeline completed successfully!")
            
            # After successful training, enable the view report button and store the path
            if pdf_path:
                self.last_report_path = pdf_path
                self.view_report_button.config(state=tk.NORMAL)
                print(f"El reporte más reciente está en: {self.last_report_path}")
            else:
                print("No se generó un reporte PDF o la ruta no es válida.")

        except Exception as e:
            print(f"An error occurred during training: {e}")
        finally:
            # Re-enable buttons after training (or failure)
            self.train_button.config(state=tk.NORMAL)
            self.browse_button.config(state=tk.NORMAL)

    def open_report(self):
        """Opens the last generated PDF report in the system's default viewer."""
        if not self.last_report_path:
            print("No se ha generado ningún reporte PDF aún.")
            return
            
        if not os.path.exists(self.last_report_path):
            print(f"Error: No se encuentra el archivo del reporte en: {self.last_report_path}")
            print("El reporte podría haber sido movido o eliminado.")
            return

        try:
            import subprocess
            import platform
            
            system = platform.system().lower()
            
            if system == "windows":
                # Windows
                try:
                    os.startfile(self.last_report_path)
                    print(f"Abriendo reporte en el visor predeterminado: {self.last_report_path}")
                except Exception:
                    # Fallback to explorer.exe
                    subprocess.Popen(["explorer.exe", self.last_report_path])
                    print(f"Abriendo reporte usando explorer.exe: {self.last_report_path}")
            
            elif system == "linux":
                # Linux
                try:
                    # Try xdg-open first (most common in Linux)
                    subprocess.Popen(["xdg-open", self.last_report_path])
                    print(f"Abriendo reporte usando xdg-open: {self.last_report_path}")
                except FileNotFoundError:
                    try:
                        # Try with evince (GNOME PDF viewer)
                        subprocess.Popen(["evince", self.last_report_path])
                        print(f"Abriendo reporte usando evince: {self.last_report_path}")
                    except FileNotFoundError:
                        try:
                            # Try with okular (KDE PDF viewer)
                            subprocess.Popen(["okular", self.last_report_path])
                            print(f"Abriendo reporte usando okular: {self.last_report_path}")
                        except FileNotFoundError:
                            print("No se pudo abrir el reporte automáticamente.")
                            print(f"Puedes abrir el reporte manualmente desde: {self.last_report_path}")
                            print("El reporte se encuentra en la carpeta 'report'")
            
            elif system == "darwin":
                # macOS
                subprocess.Popen(["open", self.last_report_path])
                print(f"Abriendo reporte en el visor predeterminado: {self.last_report_path}")
            
            else:
                print(f"Sistema operativo no soportado: {system}")
                print(f"Puedes abrir el reporte manualmente desde: {self.last_report_path}")
                print("El reporte se encuentra en la carpeta 'report'")
                
        except Exception as e:
            print(f"Error al abrir el reporte PDF: {e}")
            print(f"Puedes abrir el reporte manualmente desde: {self.last_report_path}")
            print("El reporte se encuentra en la carpeta 'report'")

if __name__ == "__main__":
    # Disable scikit-learn and joblib logging warnings
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.WARNING)

    root = tk.Tk()
    app = ModelTrainerApp(root)
    root.mainloop() 