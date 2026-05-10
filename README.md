# API de Clasificación ArXiv

Esta API clasifica artículos científicos de ArXiv según el contenido procesado del abstract.

El modelo utilizado es Logistic Regression con TF-IDF, entrenado sobre abstracts limpios del dataset ArXiv Papers.

## Archivos principales

- `main.py`: contiene la API desarrollada con FastAPI.
- `requirements.txt`: contiene las librerías necesarias para ejecutar el proyecto.
- `modelo_arxiv_logistic_regression.pkl`: modelo entrenado y serializado con joblib.

## Endpoint principal

```http
GET /
```

Respuesta esperada:

```json
{
  "mensaje": "API de clasificación ArXiv activa",
  "documentacion": "/docs"
}
```

## Endpoint de predicción

```http
POST /predecir
```

Ejemplo de entrada:

```json
{
  "abstract_limpio": "large language model transformer neural network text generation"
}
```

Ejemplo de salida:

```json
{
  "categoria_predicha": "cs.CL",
  "probabilidades": {
    "cs.AI": 0.20,
    "cs.CL": 0.65,
    "cs.CV": 0.05,
    "cs.LG": 0.08,
    "cs.RO": 0.02
  }
}
```

## Ejecución local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar API:

```bash
uvicorn main:app --reload
```

Abrir documentación:

```text
http://127.0.0.1:8000/docs
```

## Despliegue en Render

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```