# Consignia: 
Se requiere construir un modelo que sea capaz de predecir, dados los datos de síntomas de un paciente, si es posible o no que este sufra de alguna enfermedad. Esto se requiere tanto para enfermedades comunes (muchos datos) como para enfermedades huérfanas (pocos datos). 

Para el problema anterior, proponga un pipeline de ML en donde describa, de manera general, el proceso completo end-to-end para el desarrollo del modelo requerido.

Considerando el mismo problema, se necesita desarrollar un servicio que permita a un médico usar el modelo para obtener una respuesta

### ⚠️ Nota importante

Parte de este trabajo será reutilizado en mi proyecto de grado, enfocado en la clasificación de **tumores cerebrales**.  
- Para la **primera entrega**, se consideraron únicamente dos categorías: `HIGH GRADE` y `LOW GRADE`.
- En esta **segunda entrega**, se amplió a **cuatro categorías**:
  - `LOW GRADE de primer grado`
  - `LOW GRADE de segundo grado`.
  - `HIGH GRADE de tercer grado`
  - `HIGH GRADE de cuarto grado`

---

# Guia de uso:

## Instalacion 

1. Instalar Docker. 
Es necesario instalar docker ya que esta es la herramienta que facilitara al usuario desplegar el sistema de manera automatizada sin depender de un profesional.

recurso: https://docs.docker.com/engine/install/

2. Lanzar el sistema

Estando en la carpeta root del proyecto, abra una terminal y ejecute el comando: docker-compose up -d 

Posterior a esto, se desplegaran automaticamente las aplicaciones de front y back-end.


## Uso del producto

Entrar en el navegador a http://127.0.0.1:5000 para conectarse a la web app. 

En esta encontrara 2 opciones posibles: 

* clasificar un archivo CSV: importar un archivo de datos radiomicos de un tumor cerebral con el fin de clasificar su grado de malignidad. Al presionar el boton de clasificar archivo, se mostrara la prediccion del modelo en pantalla.
    * (El modelo no es mas que una funcion y para que esta funcione es necesario usar los archivos proporcionados en la carpeta data.)
* Obtener ultimas predicciones: Al presionar el bnoton obtener, se mostrara en pantalla una lista con las ultimas 5 predicciones registradas.

---

# Pipeline propuesto

### 1. Data Input

- Se reciben archivos generados a partir de la extracción de datos radiómicos de imágenes de resonancia magnética cerebral.
- Estos archivos son entregados por la clínica directamente al equipo técnico responsable.
- Posteriormente, se alojan en un bucket para que sean accesibles tanto para el pipeline como para el resto del equipo de desarrollo.  
  > Propuesta: **AWS S3**


### 2. Data Processing

- Los archivos se reformatean para poder tratarlos como registros en una tabla estructurada.
- Se realiza un análisis exploratorio de los datos, generando un reporte con estadísticas descriptivas relevantes del dataset.
- Si se detecta desbalanceo entre clases, se aplica **oversampling** a la clase minoritaria.
- Se preparan los datos mediante:
  - Normalización de características.
  - Detección y tratamiento de outliers.
  - Reduccion de dimensionalidad
  - feature extraction
  - Gestion de variables categoricas
  - Divisiones en conjuntos de entrenamiento y prueba.


### 3. Model Iterations

- Se entrenan tres modelos:
  - **XGBoost**
  - **Random Forest**
  - **SVM**
- Todos los modelos utilizan **validación cruzada (cross-validation)**.
- Se realiza una búsqueda de hiperparámetros para cada modelo con el fin de optimizar su desempeño.
- Se evalúan métricas clave como:
  - Precisión
  - Recall
  - F1-score
  - AUC



### 4. Model Selection

- Se selecciona el modelo con mejor desempeño en las métricas clave, priorizando **precisión** y **recall**, dado el contexto clínico.
- El modelo seleccionado es validado de acuerdo con los estándares de calidad establecidos por la clínica colaboradora.


### 5. Model Deployment

- Se desarrollan y despliegan dos componentes:
  - Un **frontend** en forma de aplicación web para interacción con médicos y personal de salud.
  - Un **backend** que expone una **API** encargada de cargar el modelo entrenado y realizar predicciones bajo demanda.
- El sistema completo se ejecuta en contenedores mediante `docker-compose` para facilitar su instalación.



### 6. Model Predictions

- El usuario accede a la aplicación web, carga un archivo y solicita una predicción.
- La predicción es procesada por el modelo alojado en el backend.
- Las predicciones son almacenadas automáticamente en una base de datos relacional para su consulta futura.  
  > 📌 Propuesta: **AWS RDS**
