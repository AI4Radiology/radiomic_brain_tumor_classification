from typing import Union
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
import re
from datetime import datetime, timezone
import json
from sklearn.preprocessing import StandardScaler
import DataProcessing
import zipfile
import os
import shutil


app = FastAPI()


@app.get("/")

def read_root():

    return {"Hello": "World"}

@app.post("/predict_one")
async def predict(file: UploadFile = File(...)):
    try:  
        print('CHECKPOINT 0: ', file.filename)

        df=DataProcessing.process(file)
        
        
        with open("resources/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    
        scaled_data = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data, columns=df.columns)

        with open('resources/model.pkl', 'rb') as f:
            model = pickle.load(f)

        pred=model.predict(df)
        print('PRED: ', pred)
        
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": file.filename,
            "tumor_grade": pred.tolist()[0]
        }
        with open("predictions_log.json", "a") as f:
            f.write(json.dumps(record) + "\n")

        return JSONResponse(content={"filename": file.filename, "prediction": pred.tolist()[0]})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/retreive_last_predictions")
async def retreive_predictions():
    file_path = "predictions_log.json"
    try:        
        df = pd.read_json(file_path, lines=True)
        
        if df.empty or "timestamp" not in df.columns:
            return JSONResponse(content={"message": "Prediction log is empty or invalid."}, status_code=400)
        print('CHECKPOINT 1: ', df.index)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=["timestamp"])  # remover filas con timestamps invalidos

        if df.empty:
            return JSONResponse(content={"message": "No valid timestamped predictions found."}, status_code=400)

        most_recent_5 = df.sort_values(by='timestamp', ascending=False).head(5)
        most_recent_5['timestamp']=most_recent_5['timestamp'].astype(str)
        
        predictions_json = most_recent_5.to_dict(orient='records')
        
        return JSONResponse(content=predictions_json)

    except Exception as e:
       return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):

    return {"item_id": item_id, "q": q}

@app.post("/update_model")
async def update_model(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.zip'):
            return JSONResponse(
                content={"error": "El archivo debe ser un ZIP"},
                status_code=400
            )
        
        # Crear directorio temporal para extraer los archivos
        temp_dir = "temp_extract"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Guardar el archivo ZIP temporalmente
        temp_zip_path = os.path.join(temp_dir, file.filename)
        with open(temp_zip_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extraer el ZIP
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Verificar que los archivos necesarios existen
        required_files = ['model.pkl', 'scaler.pkl', 'columns.pkl']
        for req_file in required_files:
            if not os.path.exists(os.path.join(temp_dir, req_file)):
                shutil.rmtree(temp_dir)
                return JSONResponse(
                    content={"error": f"Falta el archivo {req_file} en el ZIP"},
                    status_code=400
                )
        
        # Crear directorio resources si no existe
        os.makedirs("resources", exist_ok=True)
        
        # Mover los archivos a la carpeta resources
        for req_file in required_files:
            shutil.move(
                os.path.join(temp_dir, req_file),
                os.path.join("resources", req_file)
            )
        
        # Limpiar archivos temporales
        shutil.rmtree(temp_dir)
        
        return JSONResponse(content={"message": "Modelo actualizado exitosamente"})
        
    except Exception as e:
        # Limpiar en caso de error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

