import requests
from flask import Flask, render_template, request, redirect
from dotenv import load_dotenv
import os
import json
app = Flask(__name__)

load_dotenv()

FASTAPI_URL = os.getenv('API_URL')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_one', methods=['POST'])
def clasificar_uno():
    file = request.files['file']
    response = requests.post(
        f"{FASTAPI_URL}/predict_one",
        files={'file': (file.filename, file.stream, file.mimetype)}
    )
    
    try:
        # Parse the JSON response
        prediction = response.json()
        # Convert to list format that matches your template
        predictions = [prediction] if prediction else []
        return render_template("index.html", json_list=predictions)
    except json.JSONDecodeError:
        return render_template("index.html", 
                             error_message="Invalid JSON response from API")
    

@app.route('/retreive_last_predictions', methods=['GET'])
def retreive_last_predictions():
    # Hacer la solicitud a FastAPI
    response = requests.get(f"{FASTAPI_URL}/retreive_last_predictions")
    
    try:
        # Intentar parsear la respuesta como JSON
        predictions = response.json()
        
        # Verificar si es una lista de JSONs (ajusta según lo que devuelva tu API)
        if isinstance(predictions, list):
            return render_template('index.html', json_list=predictions)
        else:
            # Si es un solo JSON, se convierte a lista de un elemento
            return render_template('index.html', json_list=[predictions])
            
    except json.JSONDecodeError:
        # Si la respuesta no es JSON válido, mostrar el texto como error
        return render_template('index.html', 
                             error_message=f"Error al obtener datos: {response.text}")
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
