from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import json
import os

# ============================================================
# CONFIGURACIÓN DE LA API
# ============================================================

app = FastAPI(
    title="API de Clasificación ArXiv",
    description="API para clasificar artículos científicos de ArXiv según su abstract limpio.",
    version="1.0"
)

# Montar carpeta static para imágenes y archivos JSON
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar modelo entrenado
modelo = joblib.load("modelo_arxiv_logistic_regression.pkl")


# ============================================================
# MODELOS DE ENTRADA
# ============================================================

class Articulo(BaseModel):
    abstract_limpio: str = Field(
        ...,
        example="large language model transformer neural network text generation"
    )


class ArticuloAlternativo(BaseModel):
    abstract: str = Field(
        ...,
        example="large language model transformer neural network text generation"
    )


# ============================================================
# FUNCIÓN AUXILIAR PARA LEER JSON
# ============================================================

def cargar_json(ruta):
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as archivo:
            return json.load(archivo)

    return None


# ============================================================
# RUTA PRINCIPAL CON INTERFAZ WEB SIMPLE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def inicio():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>API ArXiv</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f9;
                padding: 40px;
                color: #222;
            }

            .container {
                max-width: 900px;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.12);
            }

            h1 {
                color: #1f3c88;
                text-align: center;
            }

            h2 {
                color: #1f3c88;
                margin-top: 30px;
            }

            code {
                background: #eef4ff;
                padding: 5px 8px;
                border-radius: 6px;
                font-weight: bold;
            }

            .endpoint {
                margin-bottom: 15px;
                padding: 12px;
                background: #f7f9fc;
                border-left: 5px solid #1f3c88;
                border-radius: 8px;
            }

            a {
                color: #1f3c88;
                font-weight: bold;
            }

            .footer {
                margin-top: 30px;
                text-align: center;
                color: #777;
                font-size: 13px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>API de Clasificación ArXiv</h1>

            <p>
                Esta API permite clasificar artículos científicos de ArXiv usando
                un modelo Logistic Regression con TF-IDF.
            </p>

            <h2>Rutas disponibles</h2>

            <div class="endpoint">
                <code>GET /</code> — Página principal de la API.
            </div>

            <div class="endpoint">
                <code>GET /docs</code> — Documentación interactiva de FastAPI.
            </div>

            <div class="endpoint">
                <code>GET /health</code> — Verifica si la API está activa.
            </div>

            <div class="endpoint">
                <code>GET /metricas</code> — Devuelve Accuracy, Precision, Recall y F1-Score.
            </div>

            <div class="endpoint">
                <code>GET /matriz</code> — Devuelve la matriz de confusión en formato JSON.
            </div>

            <div class="endpoint">
                <code>GET /categorias</code> — Devuelve las categorías que el modelo puede predecir.
            </div>

            <div class="endpoint">
                <code>GET /modelo</code> — Devuelve información general del modelo.
            </div>

            <div class="endpoint">
                <code>GET /ejemplo</code> — Devuelve un ejemplo de JSON para probar la API.
            </div>

            <div class="endpoint">
                <code>POST /predecir</code> — Recibe un abstract limpio y devuelve la categoría predicha.
            </div>

            <div class="endpoint">
                <code>POST /clasificar</code> — Ruta alternativa usando el campo abstract.
            </div>

            <p>
                Para probar la API visualmente, ingresa a:
                <a href="/docs">/docs</a>
            </p>

            <div class="footer">
                Proyecto Final ArXiv - FastAPI + GitHub + Render
            </div>
        </div>
    </body>
    </html>
    """


# ============================================================
# RUTA DE ESTADO DE LA API
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "mensaje": "La API está activa y funcionando correctamente."
    }


# ============================================================
# RUTA DE MÉTRICAS
# ============================================================

@app.get("/metricas")
def obtener_metricas():
    metricas = cargar_json("static/metricas_modelo.json")

    if metricas is None:
        return {
            "error": "No se encontró el archivo static/metricas_modelo.json"
        }

    return {
        "modelo": "Logistic Regression + TF-IDF",
        "metricas": metricas
    }


# ============================================================
# RUTA DE MATRIZ DE CONFUSIÓN
# ============================================================

@app.get("/matriz")
def obtener_matriz_confusion():
    matriz = cargar_json("static/matriz_confusion.json")

    if matriz is None:
        return {
            "error": "No se encontró el archivo static/matriz_confusion.json"
        }

    return matriz


# ============================================================
# RUTA DE CATEGORÍAS DISPONIBLES
# ============================================================

@app.get("/categorias")
def obtener_categorias():
    return {
        "categorias_disponibles": list(modelo.classes_),
        "total_categorias": len(modelo.classes_)
    }


# ============================================================
# RUTA DE INFORMACIÓN DEL MODELO
# ============================================================

@app.get("/modelo")
def informacion_modelo():
    return {
        "nombre_modelo": "modelo_arxiv_logistic_regression.pkl",
        "algoritmo": "Logistic Regression",
        "vectorizacion": "TF-IDF",
        "tipo_tarea": "Clasificación multiclase",
        "entrada": "abstract_limpio",
        "salida": "categoria_predicha",
        "plataforma_despliegue": "Render",
        "framework_api": "FastAPI"
    }


# ============================================================
# RUTA DE EJEMPLO PARA POSTMAN
# ============================================================

@app.get("/ejemplo")
def ejemplo_peticion():
    return {
        "metodo": "POST",
        "endpoint": "/predecir",
        "body_json": {
            "abstract_limpio": "large language model transformer neural network text generation"
        },
        "descripcion": "Enviar este JSON en Postman usando Body > raw > JSON."
    }


# ============================================================
# RUTA PRINCIPAL DE PREDICCIÓN
# ============================================================

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
        "abstract_recibido": articulo.abstract_limpio,
        "categoria_predicha": categoria,
        "probabilidades": prob_dict
    }


# ============================================================
# RUTA ALTERNATIVA DE CLASIFICACIÓN
# ============================================================

@app.post("/clasificar")
def clasificar_articulo(articulo: ArticuloAlternativo):
    texto = [articulo.abstract]

    categoria = modelo.predict(texto)[0]

    probabilidades = modelo.predict_proba(texto)[0]
    clases = modelo.classes_

    prob_dict = {
        clase: round(float(prob), 4)
        for clase, prob in zip(clases, probabilidades)
    }

    return {
        "abstract_recibido": articulo.abstract,
        "categoria_predicha": categoria,
        "probabilidades": prob_dict
    }