import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Third party libraries
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import datetime
import logging
import sys
import os
from contextlib import contextmanager
import json
import pickle
import zipfile

# Scikit-learn
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# Defining data path
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación y selección del mejor modelo de IA"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="prepared_data.csv",
        help="Ruta al archivo CSV de features (X). Por defecto: prepared_data.csv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para la generación de números aleatorios. Si no se especifica, se generará una aleatoria.",
    )

    return parser.parse_args()


# Data loading
def load_data(base_path):
    try:
        df = pd.read_csv(base_path)
        print("\n\n ---------------\n Tamaño del dataset:", df.shape)
        return df
    except FileNotFoundError:
        print(f"El dataset no se encuentra en la ruta: {base_path}")
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
    return None


# Basic Random Forest model
def basic_random_forest_model(X_train, y_train, rd):
    rf_model = RandomForestClassifier(random_state=rd)
    rf_model.fit(X_train, y_train)
    return rf_model


# Random Forest with Best Hyperparameters
def random_forest_best_hyperparameters(X_train, y_train, rd):
    """Trains a Random Forest classifier with optimized hyperparameters using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        rd (int): Random state for reproducibility.

    Returns:
        RandomForestClassifier: The trained Random Forest model with best hyperparameters.
    """
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


# Basic XGBoost model
def basic_xgboost_model(X_train, y_train, rd):
    """Trains a basic XGBoost classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        rd (int): Random state for reproducibility.

    Returns:
        xgb.XGBClassifier: The trained XGBoost model.
    """
    xgb_model = xgb.XGBClassifier(
        seed=rd, objective="binary:logistic", eval_metric="aucpr"
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


# XGBoost with Best Hyperparameters
def xgboost_best_hyperparameters(X_train, y_train, rd):
    """Trains an XGBoost classifier with optimized hyperparameters using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        rd (int): Random state for reproducibility.

    Returns:
        xgb.XGBClassifier: The trained XGBoost model with best hyperparameters.
    """
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

    # Custom scorer for average_precision_score
    def custom_scorer(estimator, X, y):
        y_pred_proba = estimator.predict_proba(X)[:, 1]
        return average_precision_score(y, y_pred_proba)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=custom_scorer,
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=0,  # Show progress
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    return best_xgb


# Basic SVM model
def basic_svm_model(X_train, y_train, rd):
    """Trains a basic Support Vector Machine (SVM) classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        rd (int): Random state for reproducibility.

    Returns:
        SVC: The trained SVM model.
    """
    svm_model = SVC(probability=True, random_state=rd)
    svm_model.fit(X_train, y_train)
    return svm_model


# SVM with Best Hyperparameters
def svm_best_hyperparameters(X_train, y_train, rd):
    """Trains a Support Vector Machine (SVM) classifier with optimized hyperparameters using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        rd (int): Random state for reproducibility.

    Returns:
        SVC: The trained SVM model with best hyperparameters.
    """
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


def create_confusion_matrix_plot(cm, model_name):
    """Creates a confusion matrix plot and returns it as a BytesIO object.

    Args:
        cm (np.ndarray): The confusion matrix.
        model_name (str): The name of the model for the plot title.

    Returns:
        io.BytesIO: A BytesIO object containing the PNG image of the confusion matrix plot.
    """
    plt.figure(figsize=(8, 6))
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

    # Save plot to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    buf.seek(0)
    return buf


def select_best_model(models_dict, X_test, y_test, rd):
    """Evaluates all models, selects the best one based on multiple metrics,
    and generates a professional PDF report.

    Args:
        models_dict (dict): Dictionary containing model names and their instances.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        rd (int): Random state used for the training process, included in PDF filename.

    Returns:
        tuple: A tuple containing:
            - str: The name of the best performing model.
            - object: The instance of the best performing model.
            - dict: A dictionary containing all calculated metrics for each model.
    """
    metrics = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define custom colors
    primary_color = colors.HexColor("#2F4F4F")  # Dark slate gray
    secondary_color = colors.HexColor("#1E90FF")  # Dodger blue
    accent_color = colors.HexColor("#F0F8FF")  # Alice blue
    text_color = colors.HexColor("#333333")  # Dark gray

    # Create PDF document
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=32,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=primary_color,
        leading=40,  # Line spacing
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
    weights_file = "../metrics_weights.json"
    try:
        with open(weights_file, "r") as f:
            weights = json.load(f)
        print(f"Pesos de métricas cargados desde {weights_file}")
    except FileNotFoundError:
        print(f"Error: Archivo de pesos {weights_file} no encontrado.")
        return None, None, None # Indicate failure
    except json.JSONDecodeError:
        print(f"Error: Archivo {weights_file} contiene JSON inválido.")
        return None, None, None # Indicate failure

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
        # Calculate positive precision
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

    # Create PDF with best model name
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)
    pdf_path = os.path.join(report_dir, f"model_evaluation_{best_model_name.replace(' ', '_')}_{rd}.pdf")
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
        # Add model section with custom style
        story.append(Paragraph(f"Modelo: {model_name}", section_style))
        story.append(Spacer(1, 12))

        # Create metrics table with enhanced styling
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
        cm_plot = create_confusion_matrix_plot(metrics[model_name]['confusion_matrix'], model_name)
        img = Image(cm_plot, width=6 * inch, height=4.5 * inch)
        story.append(img)
        story.append(Spacer(1, 30))

        # Add separator line between models
        story.append(Paragraph("_" * 100, ParagraphStyle("Separator", textColor=primary_color)))
        story.append(Spacer(1, 30))

    # Add final results section
    story.append(Paragraph("Resultado Final", title_style))
    story.append(Spacer(1, 20))

    # Create best model table with enhanced styling
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
                # Style for the first column (header)
                ("BACKGROUND", (0, 0), (0, -1), secondary_color),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (0, -1), 14),
                # Style for the second column (data)
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

    # Save the best model
    model_dir = "resources"
    os.makedirs(model_dir, exist_ok=True)
    zip_filename = f"resources.zip"
    zip_path = os.path.join(model_dir, zip_filename)

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

        print(f"Modelo y scaler guardados en: {zip_path}")
    except Exception as e:
        print(f"Error al guardar el modelo y el scaler: {e}")

    return best_model_name, best_model, metrics


@contextmanager
def suppress_stdout_stderr():
    """Context manager to temporarily suppress stdout and stderr.

    This is useful for silencing verbose output from libraries during specific operations.
    """
    new_stdout, new_stderr = os.pipe()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = os.fdopen(new_stdout, "w"), os.fdopen(new_stderr, "w")
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.close(new_stdout)
        os.close(new_stderr)


if __name__ == "__main__":
    # Desactivar logs de scikit-learn
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.WARNING)

    # Configurar logging básico solo para consola
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    df = load_data(args.data_path)

    if df is not None:
        # Dividir datos, quitando "Id" solo si existe
        drop_cols = ["highGrade"]
        if "Id" in df.columns:
            drop_cols.insert(0, "Id")
        X = df.drop(columns=drop_cols)
        y = df["highGrade"]

        # Guardar los nombres de las características
        feature_names = X.columns.tolist()

        print("Features seleccionadas:")
        print(X.columns.tolist())

        # Splitting data
        if args.seed is None:
            rd = random.randint(1, 3141592654)
            print(f"No se proporcionó semilla, se generó una aleatoria: {rd}")
        else:
            rd = args.seed
            print(f"Usando semilla proporcionada: {rd}")
            random.seed(rd) # Set the seed for the random module

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rd
        )

        # Scaling data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convertir de nuevo a DataFrame para mantener los nombres de las características
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=rd)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_scaled, y_train
        )

        # Convertir de nuevo a DataFrame después de SMOTE
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=feature_names)

        print("\n" + "=" * 50)
        print("Iniciando entrenamiento de modelos...")
        print("=" * 50 + "\n")

        # Train basic models
        print("Entrenando Random Forest básico...")
        basic_rf = basic_random_forest_model(X_train_resampled, y_train_resampled, rd)

        print("Entrenando XGBoost básico...")
        basic_xgb = basic_xgboost_model(X_train_resampled, y_train_resampled, rd)

        print("Entrenando SVM básico...")
        basic_svm = basic_svm_model(X_train_resampled, y_train_resampled, rd)

        # Train optimized models
        print("\nEntrenando modelos optimizados...")
        print("Random Forest optimizado...")
        best_rf = random_forest_best_hyperparameters(
            X_train_resampled, y_train_resampled, rd
        )

        print("XGBoost optimizado...")
        best_xgb = xgboost_best_hyperparameters(
            X_train_resampled, y_train_resampled, rd
        )

        print("SVM optimizado...")
        best_svm = svm_best_hyperparameters(X_train_resampled, y_train_resampled, rd)

        # Create dictionary of all models
        models = {
            "Random Forest Básico": basic_rf,
            "Random Forest Optimizado": best_rf,
            "XGBoost Básico": basic_xgb,
            "XGBoost Optimizado": best_xgb,
            "SVM Básico": basic_svm,
            "SVM Optimizado": best_svm,
        }

        print("\nEvaluando modelos y generando reporte...")
        best_model_name, best_model, metrics = select_best_model(
            models, X_test_scaled, y_test, rd
        )

        pdf_filename = (
            f"model_evaluation_{best_model_name.replace(' ', '_')}_{rd}.pdf"
        )

        print("\n" + "=" * 50)
        print(f"Proceso completado exitosamente")
        print(f"El mejor modelo es: {best_model_name}")
        print(f"Se ha generado un reporte detallado en formato PDF: report/{pdf_filename}")
        print(f"Semilla utilizada: {rd}")
        print("=" * 50 + "\n")
