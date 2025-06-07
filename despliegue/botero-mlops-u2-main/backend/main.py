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

        with open('resources/svm_model.pkl', 'rb') as f:
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

