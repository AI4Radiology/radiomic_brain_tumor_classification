# radiomic_tumor_classification

## SCRIPT: flairReformatting.py
este script construye un dataset usable con todos los registros radiomicos de las imagenes flair. En este dataset las columnas son las feature names y los valores son los valores correspondientes a cada feature. Las columnas de image type y feature class NO son tomadas en cuenta en el reformateo.  

### uso: 
python flairReformatting.py ruta/directorio/de/datos ruta/de/salida

### help: 
python flairReformatting.py -h

