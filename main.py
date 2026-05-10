from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Clasificación ArXiv",
    description="API para clasificar artículos científicos de ArXiv según su abstract limpio.",
    version="1.0"
)

# Cargar modelo entrenado
modelo = joblib.load("modelo_arxiv_logistic_regression.pkl")

# Estructura de entrada
class Articulo(BaseModel):
    abstract_limpio: str

# Ruta principal
@app.get("/")
def inicio():
    return {
        "mensaje": "API de clasificación ArXiv activa",
        "documentacion": "/docs"
    }

# Ruta de predicción
@app.post("/predecir")
def predecir_categoria(articulo: Articulo):
    texto = [articulo.abstract_limpio]

    categoria = modelo.predict(texto)[0]

    probabilidades = modelo.predict_proba(texto)[0]
    clases = modelo.classes_

    prob_dict = {
        clase: round(float(prob), 4)
        for clase, prob in zip(clases, probabilidades)
    }

    return {
        "categoria_predicha": categoria,
        "probabilidades": prob_dict
    }