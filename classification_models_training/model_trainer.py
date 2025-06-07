import os
import random
import numpy as np
import pandas as pd
import pickle
import zipfile
from io import BytesIO
import datetime
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Scikit-learn imports
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE

class ModelTrainer:
    def __init__(self, output_dir="data", resources_dir="resources", report_dir="report"):
        self.output_dir = output_dir
        self.resources_dir = resources_dir
        self.report_dir = report_dir
        
        # Ensure necessary directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.resources_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def load_data(self, base_path):
        """Loads data from a CSV file."""
        try:
            df = pd.read_csv(base_path)
            print(f"\n---------------\nDataset size: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: Dataset not found at path: {base_path}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
        return None

    def basic_random_forest_model(self, X_train, y_train, rd):
        """Trains a basic Random Forest classifier."""
        rf_model = RandomForestClassifier(random_state=rd)
        rf_model.fit(X_train, y_train)
        return rf_model

    def random_forest_best_hyperparameters(self, X_train, y_train, rd):
        """Trains a Random Forest classifier with optimized hyperparameters."""
        param_grid = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced", {True: 1, False: 2}, {True: 1, False: 3}],
        }
        scoring = "roc_auc"
        new_rf_model = RandomForestClassifier(random_state=rd)
        random_search = RandomizedSearchCV(
            estimator=new_rf_model,
            param_distributions=param_grid,
            n_iter=50,
            scoring=scoring,
            n_jobs=-1,
            cv=5,
            verbose=0,
            random_state=rd,
        )
        random_search.fit(X_train, y_train)
        best_rf_model = random_search.best_estimator_
        return best_rf_model

    def basic_xgboost_model(self, X_train, y_train, rd):
        """Trains a basic XGBoost classifier."""
        xgb_model = xgb.XGBClassifier(
            seed=rd, objective="binary:logistic", eval_metric="aucpr"
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model

    def xgboost_best_hyperparameters(self, X_train, y_train, rd):
        """Trains an XGBoost classifier with optimized hyperparameters."""
        param_grid = {
            "gamma": [0, 0.1, 0.25, 0.5, 1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "reg_lambda": [1, 5, 10],
            "scale_pos_weight": [np.sum(y_train == False) / np.sum(y_train == True)],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
        xgb_model = xgb.XGBClassifier(
            seed=rd, objective="binary:logistic", eval_metric="aucpr"
        )

        def custom_scorer(estimator, X, y):
            y_pred_proba = estimator.predict_proba(X)[:, 1]
            return average_precision_score(y, y_pred_proba)

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring=custom_scorer,
            cv=5,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        return best_xgb

    def basic_svm_model(self, X_train, y_train, rd):
        """Trains a basic Support Vector Machine (SVM) classifier."""
        svm_model = SVC(probability=True, random_state=rd)
        svm_model.fit(X_train, y_train)
        return svm_model

    def svm_best_hyperparameters(self, X_train, y_train, rd):
        """Trains a Support Vector Machine (SVM) classifier with optimized hyperparameters."""
        param_grid = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "degree": [2, 3],
            "coef0": [0, 1],
        }
        svm_model = SVC(probability=True, random_state=rd)
        grid_search = GridSearchCV(
            estimator=svm_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_svm = grid_search.best_estimator_
        return best_svm

    def create_confusion_matrix_plot(self, cm, model_name):
        """Creates a confusion matrix plot and returns it as a BytesIO object."""
        # Create figure with explicit figure number to avoid warnings
        plt.figure(num=1, figsize=(8, 6), clear=True)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicción False", "Predicción True"],
            yticklabels=["Verdaderos Negativos", "Verdaderos Positivos"],
        )
        plt.title(f"Matriz de confusión - {model_name}")
        plt.xlabel("Predicción")
        plt.ylabel("Valor real")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close('all')  # Close all figures to free memory
        buf.seek(0)
        return buf

    def select_best_model(self, models_dict, X_test, y_test, rd):
        """Evaluates all models, selects the best one, and generates a PDF report."""
        metrics = {}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define custom colors
        primary_color = colors.HexColor("#2F4F4F")
        secondary_color = colors.HexColor("#1E90FF")
        accent_color = colors.HexColor("#F0F8FF")
        text_color = colors.HexColor("#333333")

        # Create PDF document
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=32,
            spaceAfter=30,
            alignment=1,
            textColor=primary_color,
            leading=40,
        )

        subtitle_style = ParagraphStyle(
            "CustomSubtitle",
            parent=styles["Heading2"],
            fontSize=24,
            spaceAfter=20,
            alignment=1,
            textColor=secondary_color,
            leading=30,
        )

        section_style = ParagraphStyle(
            "SectionStyle",
            parent=styles["Heading2"],
            fontSize=18,
            spaceAfter=15,
            textColor=primary_color,
            leading=24,
        )

        date_style = ParagraphStyle(
            "DateStyle",
            parent=styles["Normal"],
            fontSize=12,
            textColor=text_color,
            alignment=1,
            spaceAfter=30,
        )

        # Load weights from JSON file
        weights_file = "metrics_weights.json"
        try:
            with open(weights_file, "r") as f:
                weights = json.load(f)
            print(f"Pesos de métricas cargados desde {weights_file}")
        except FileNotFoundError:
            print(f"Error: Archivo de pesos {weights_file} no encontrado.")
            return None, None, None, None
        except json.JSONDecodeError:
            print(f"Error: Archivo {weights_file} contiene JSON inválido.")
            return None, None, None, None

        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            y_pred_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred
            )

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            tn, fn, fp, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            positive_precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Store metrics
            metrics[model_name] = {
                "accuracy": accuracy,
                "auc_roc": auc_roc,
                "f1_score": f1,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "positive_precision": positive_precision,
                "confusion_matrix": cm,
            }

        weighted_scores = {}
        for model_name, model_metrics in metrics.items():
            score = sum(
                model_metrics[metric] * weight for metric, weight in weights.items()
            )
            weighted_scores[model_name] = score

        best_model_name = max(weighted_scores.items(), key=lambda x: x[1])[0]
        best_model = models_dict[best_model_name]

        # Create PDF report
        pdf_path = os.path.join(self.report_dir, f"model_evaluation_{best_model_name.replace(' ', '_')}_{rd}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        story = []

        # Title and subtitle
        story.append(Paragraph("Reporte de Evaluación de Modelos", title_style))
        story.append(Paragraph("Análisis Comparativo de Modelos de Machine Learning", subtitle_style))
        story.append(Spacer(1, 20))

        # Add timestamp
        story.append(
            Paragraph(
                f"Generado el: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                date_style,
            )
        )
        story.append(Spacer(1, 30))

        # Add separator line
        story.append(Paragraph("_" * 100, ParagraphStyle("Separator", textColor=primary_color)))
        story.append(Spacer(1, 30))

        for model_name, model in models_dict.items():
            # Add model section
            story.append(Paragraph(f"Modelo: {model_name}", section_style))
            story.append(Spacer(1, 12))

            # Create metrics table
            data = [
                ["Métrica", "Valor"],
                ["Exactitud (Accuracy)", f"{metrics[model_name]['accuracy']:.4f}"],
                ["AUC-ROC", f"{metrics[model_name]['auc_roc']:.4f}"],
                ["F1-Score", f"{metrics[model_name]['f1_score']:.4f}"],
                ["Sensibilidad", f"{metrics[model_name]['sensitivity']:.4f}"],
                ["Especificidad", f"{metrics[model_name]['specificity']:.4f}"],
                ["Precisión Positiva", f"{metrics[model_name]['positive_precision']:.4f}"],
            ]

            table = Table(data, colWidths=[3 * inch, 2 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), accent_color),
                        ("TEXTCOLOR", (0, 1), (-1, -1), text_color),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 1), (-1, -1), 10),
                        ("GRID", (0, 0), (-1, -1), 1, primary_color),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )

            story.append(table)
            story.append(Spacer(1, 20))

            # Add confusion matrix plot
            cm_plot = self.create_confusion_matrix_plot(metrics[model_name]['confusion_matrix'], model_name)
            img = Image(cm_plot, width=6 * inch, height=4.5 * inch)
            story.append(img)
            story.append(Spacer(1, 30))

            # Add separator line between models
            story.append(Paragraph("_" * 100, ParagraphStyle("Separator", textColor=primary_color)))
            story.append(Spacer(1, 30))

        # Add final results section
        story.append(Paragraph("Resultado Final", title_style))
        story.append(Spacer(1, 20))

        # Create best model table
        best_model_data = [
            ["Mejor Modelo", best_model_name],
            ["Puntuación Ponderada", f"{weighted_scores[best_model_name]:.4f}"],
        ]

        best_model_table = Table(best_model_data, colWidths=[4 * inch, 2 * inch])
        best_model_table.setStyle(
            TableStyle(
                [
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 14),
                    ("GRID", (0, 0), (-1, -1), 1, primary_color),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BACKGROUND", (0, 0), (0, -1), secondary_color),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (0, -1), 14),
                    ("BACKGROUND", (1, 0), (1, -1), accent_color),
                    ("TEXTCOLOR", (1, 0), (1, -1), text_color),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(best_model_table)

        # Build PDF
        doc.build(story)
        print(f"Reporte generado exitosamente: {pdf_path}")

        return best_model_name, best_model, metrics, pdf_path

    def train_and_save_models(self, df, feature_names):
        """Trains various models, selects the best, and saves artifacts."""
        X = df.drop(columns=["Id", "highGrade"], errors='ignore')
        y = df["highGrade"]

        # Use a fixed random seed for reproducibility
        rd = random.randint(1, 3141592654)
        print(f"Using random seed: {rd}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rd
        )

        # Scaling data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame to retain feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

        # Apply SMOTE only for SVM models
        smote = SMOTE(random_state=rd)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
        X_train_smote = pd.DataFrame(X_train_smote, columns=feature_names)

        print("\n" + "=" * 50)
        print("Initiating model training...")
        print("=" * 50 + "\n")

        # Train basic models
        print("Training basic Random Forest...")
        basic_rf = self.basic_random_forest_model(X_train_scaled, y_train, rd)

        print("Training basic XGBoost...")
        basic_xgb = self.basic_xgboost_model(X_train_scaled, y_train, rd)

        print("Training basic SVM...")
        basic_svm = self.basic_svm_model(X_train_smote, y_train_smote, rd)

        # Train optimized models
        print("\nTraining optimized models...")
        print("Optimized Random Forest...")
        best_rf = self.random_forest_best_hyperparameters(
            X_train_scaled, y_train, rd
        )

        print("Optimized XGBoost...")
        best_xgb = self.xgboost_best_hyperparameters(
            X_train_scaled, y_train, rd
        )

        print("Optimized SVM...")
        best_svm = self.svm_best_hyperparameters(X_train_smote, y_train_smote, rd)

        # Create dictionary of all models
        models = {
            "Random Forest Básico": basic_rf,
            "Random Forest Optimizado": best_rf,
            "XGBoost Básico": basic_xgb,
            "XGBoost Optimizado": best_xgb,
            "SVM Básico": basic_svm,
            "SVM Optimizado": best_svm,
        }

        print("\nEvaluating models and generating report...")
        best_model_name, best_model, metrics, pdf_path = self.select_best_model(
            models, X_test_scaled, y_test, rd
        )

        if best_model is None:
            print("Model evaluation failed. No model saved.")
            return None, None, None, None

        # Save the best model, scaler, and columns into a zip file
        zip_path = os.path.join(self.resources_dir, "resources.zip")

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Save the best model
                model_bytes = BytesIO()
                pickle.dump(best_model, model_bytes)
                model_bytes.seek(0)
                zf.writestr('model.pkl', model_bytes.read())

                # Save the scaler
                scaler_bytes = BytesIO()
                pickle.dump(scaler, scaler_bytes)
                scaler_bytes.seek(0)
                zf.writestr('scaler.pkl', scaler_bytes.read())

                # Save the column names
                columns_bytes = BytesIO()
                pickle.dump(feature_names, columns_bytes)
                columns_bytes.seek(0)
                zf.writestr('columns.pkl', columns_bytes.read())

            print(f"All resources (model, scaler, columns) saved to: {zip_path}")
            return best_model_name, best_model, metrics, pdf_path
        except Exception as e:
            print(f"Error saving resources to zip file: {e}")
            return None, None, None, None 