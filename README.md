# Sistema de clasificación de lesiones cerebrales intra-axiales de tipo masa a partir de Imágenes de Resonancia Magnética (IRM)

Este proyecto desarrolla un modelo de inteligencia artificial para la clasificación de lesiones cerebrales intra-axiales de tipo masa según el grado de malignidad, utilizando datos radiómicos extraídos a partir de imágenes de resonancia magnética. El objetivo es optimizar el tiempo del diagnóstico, minimizar el abordaje quirúrgico y mejorar el pronóstico en pacientes.

El sistema implementa un pipeline completo que trabaja con datos radiómicos proporcionados por la clínica, incluyendo análisis exploratorio, entrenamiento de modelos predictivos, y aplicaciones web para el uso clínico de los modelos desarrollados. 

## Estructura del Proyecto

### `classification_models_training/`

Pipeline de entrenamiento de modelos predictivos para la clasificación del grado de malignidad de tumores cerebrales, que incluye una implementación automatizada del proceso de entrenamiento con validación cruzada, optimización de hiperparámetros, el uso de algoritmos de machine learning aplicados a datos radiómicos para predecir el grado del tumor, y la generación automática de reportes con métricas de evaluación como precisión, recall, F1-score y matrices de confusión.

**Guia de uso**
- Dos versiones del pipeline:
    * **`cli-version/`**: versión de línea de comandos que permite lanzar el entrenamiento directamente desde la terminal, configurando manualmente la ruta a los datos y una semilla para controlar la aleatoriedad en la partición entre datos de entrenamiento y prueba.

    - **`gui-version/`**: versión con interfaz gráfica de escritorio, que permite:
        - Seleccionar la carpeta con los archivos de datos radiómicos.
        - Configurar semilla del entrenamiento.
        - Ejecutar el pipeline.
        - Visualizar un reporte con las estadísticas más relevantes de cada modelo entrenado.
    - **`build_exe.py`**: generador de archivo ejecutable de la version de gui. 

**Salida del pipeline:**

- Ambas versiones generan un archivo `resources/resources.zip` que contiene el mejor modelo y otros recursos del entrenamiento en un archivo `.zip`.

**Uso básico:**

*cli-version*
```bash
cd classification_models_training/cli-version/
python best_model_selector.py --data_path /ruta/a/dataset.csv --seed 744193200
```

*gui-version*
```bash
cd classification_models_training/gui-version/
python model_trainer_gui.py
```
*build.exe*
```bash
cd classification_models_training/cli-version/
python build_exe.py
```

### `despliegue/`

Sistema de aplicaciones web para uso clínico de los modelos predictivos.

**Componentes:**

- **Frontend:** Aplicación en Flask para la carga de datos radiómicos, visualización de predicciones y carga de nuevos modelos.  
- **Backend:** API desarrollada con FastAPI para la predicción del grado de malignidad. Predice el grado del tumor y registra las predicciones localmente.  
- **Docker Compose:** Configuración para el despliegue en un entorno clínico.

**Instalación:**

Es necesario instalar Docker, ya que esta herramienta permite al usuario desplegar el sistema de forma automatizada, sin depender de un profesional.  
Recurso: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

**Lanzar el sistema**

Desde la carpeta raíz del proyecto, abre una terminal y ejecuta el siguiente comando:

```bash
cd despliegue/despliegue_docker
docker-compose up -d

```
Después de esto, las aplicaciones de frontend y backend se desplegarán automáticamente.


**Uso del producto**

Entrar en el navegador a http://127.0.0.1:5000 para conectarse a la web app.

En esta encontrará 3 opciones:

- clasificar un archivo CSV: importar un archivo de datos radiomicos de un tumor cerebral con el fin de clasificar su grado de malignidad. Al presionar el boton de clasificar archivo, se mostrara la prediccion del modelo en pantalla.
- Obtener ultimas predicciones: Al presionar el boton obtener, se mostrara en pantalla una lista con las ultimas 5 predicciones registradas.
- importar modelo: ventana para subir el archivo resources.zip resultante del pipeline de entrenamiento. 



### `data_processing/`

Módulo de procesamiento y formateo de datos radiómicos proporcionados por la clínica.

**Scripts:**

- `flairReformatting.py`  
  - Carga y validación de datasets radiómicos de la clínica  
  - Limpieza y normalización de características radiómicas  
  - Transformación y preparación de datos para análisis y entrenamiento

**Uso:**
```bash
cd data_processing/

python flairReformatting.py ruta/directorio/de/datos ruta/de/salida
# ayuda: python flairReformatting -h
```
Este script recibe archivos CSV de características radiómicas extraídas de imágenes FLAIR.  
Retorna un único archivo CSV con los datos formateados, procesados y listos para ser suministrados a un modelo de predicción.


- mixedReformatting.py:  
  - Esta versión permite, además de sus parámetros ya existentes, recibir la ruta a un archivo CSV que contiene los datos radiómicos extraídos del dataset BRATS.  
  - Facilita la combinación y el preprocesamiento conjunto de datos provenientes tanto de la clínica como del dataset público.

```bash
cd data_processing/

python mixedReformatting.py ruta/directorio/de/datos ruta/archivo/datos/brats ruta/de/salida
# ayuda: python flairReformatting -h
```
Este script recibe como entrada dos datasets: el de la clínica colaboradora y el de BRATS.  
Con ellos, se realiza un balanceo de la clase minoritaria utilizando datos radiómicos reales extraídos de las imágenes de resonancia magnética (RMI) del dataset BRATS.

### `jupyter_notebooks/`
Análisis exploratorio y experimentación con datos radiómicos.

**Contenido:**

**Análisis exploratorio:** 
- Visualizaciones de distribución de datos radiómicos
- Generación de insights para entendimiento clínico de patologías

**Experimento con BRATS2017:** 

Procesamiento completo de dataset público de resonancias magnéticas parcialmente compatible
  - Procesamiento de imágenes 3D de resonancia magnética.
  - Extracción de características radiómicas, manteniendo la misma configuración utilizada por la clínica.
  - Preparación y entrenamiento de modelos con excelentes resultados
  - Validación de metodología propuesta. Nuestro pipeline consigue resultados al nivel del estado del arte con este dataset.


**Experimento de segmentación volumétrica**

Se desarrolló todo el proceso, desde el procesamiento de las resonancias magnéticas hasta el entrenamiento de una U-Net utilizando el dataset BRATS. Se obtuvieron resultados prometedores con estos datos, aunque no aplicables directamente a los datos de la clínica. Estos resultados evidencian que, de contar con acceso a las máscaras volumétricas de los tumores y a las distintas secuencias de resonancia, principalmente la T1-C, sería posible extender el proyecto hacia un pipeline de segmentación automática.

**Experimento de segmentación volumétrica**

Cuadernos `ipynb` que hicieron parte de la experimentación y desarrollo de modelos predictivos para estimación de grado de malignidad. 
Representan una etapa previa a la implementación del pipeline de entrenamiento.


## Flujo de Trabajo Completo

1. **Procesamiento de datos:** Carga y preparación de datos radiómicos clínicos
2. **Análisis exploratorio:** Identificación de correlaciones radiómicas relevantes
3. **Desarrollo del modelo:** Entrenamiento y validación de algoritmos predictivos
4. **Validación experimental:** Experimentos con dataset BRATS2017 para validación de metodología
5. **Despliegue clínico:** Implementación de sistema web para uso en entorno hospitalario


## Documentación Técnica

Para información detallada sobre la metodología radiómica, algoritmos implementados, resultados de validación y análisis clínico del proyecto, consulte el escrito del proyecto de grado.
