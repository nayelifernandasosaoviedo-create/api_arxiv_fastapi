from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import json
import os

app = FastAPI(
    title="API de Clasificación ArXiv",
    description="API para clasificar artículos científicos de ArXiv según su abstract limpio.",
    version="1.0"
)

# Montar carpeta static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cargar modelo entrenado
modelo = joblib.load("modelo_arxiv_logistic_regression.pkl")


class Articulo(BaseModel):
    abstract_limpio: str


def cargar_metricas():
    ruta_metricas = "static/metricas_modelo.json"

    if os.path.exists(ruta_metricas):
        with open(ruta_metricas, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "accuracy": "N/D",
        "precision": "N/D",
        "recall": "N/D",
        "f1_score": "N/D",
        "registros_entrenamiento_api": "N/D",
        "categorias": []
    }


@app.get("/", response_class=HTMLResponse)
def inicio():
    metricas = cargar_metricas()

    categorias_html = "".join([
        f"<span class='badge'>{cat}</span>"
        for cat in metricas.get("categorias", [])
    ])

    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clasificador ArXiv</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f6f9;
                margin: 0;
                padding: 0;
                color: #222;
            }}

            .container {{
                max-width: 1050px;
                margin: 40px auto;
                background: white;
                padding: 35px;
                border-radius: 16px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.12);
            }}

            h1 {{
                text-align: center;
                color: #1f3c88;
                margin-bottom: 8px;
            }}

            .subtitle {{
                text-align: center;
                color: #555;
                margin-bottom: 30px;
            }}

            .cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}

            .card {{
                background: #eef4ff;
                padding: 18px;
                border-radius: 12px;
                text-align: center;
                border-left: 5px solid #1f3c88;
            }}

            .card h3 {{
                margin: 0;
                font-size: 15px;
                color: #333;
            }}

            .card p {{
                margin: 8px 0 0 0;
                font-size: 24px;
                font-weight: bold;
                color: #1f3c88;
            }}

            label {{
                font-weight: bold;
                color: #333;
            }}

            textarea {{
                width: 100%;
                height: 160px;
                margin-top: 10px;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid #ccc;
                font-size: 15px;
                resize: vertical;
            }}

            button {{
                margin-top: 20px;
                width: 100%;
                background: #1f3c88;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                border-radius: 10px;
                cursor: pointer;
            }}

            button:hover {{
                background: #162d66;
            }}

            .result {{
                margin-top: 25px;
                padding: 20px;
                background: #eef4ff;
                border-left: 5px solid #1f3c88;
                border-radius: 10px;
                display: none;
            }}

            .error {{
                margin-top: 25px;
                padding: 20px;
                background: #ffecec;
                border-left: 5px solid #cc0000;
                border-radius: 10px;
                display: none;
                color: #990000;
            }}

            .category {{
                font-size: 26px;
                font-weight: bold;
                color: #1f3c88;
            }}

            .prob {{
                margin-top: 10px;
                font-size: 14px;
                color: #333;
            }}

            .example {{
                margin-top: 10px;
                font-size: 13px;
                color: #555;
                background: #f7f7f7;
                padding: 10px;
                border-radius: 8px;
            }}

            .section {{
                margin-top: 40px;
            }}

            .section h2 {{
                color: #1f3c88;
                border-bottom: 2px solid #1f3c88;
                padding-bottom: 8px;
            }}

            .chart {{
                max-width: 100%;
                border-radius: 12px;
                border: 1px solid #ddd;
                margin-top: 15px;
            }}

            .badge {{
                display: inline-block;
                background: #1f3c88;
                color: white;
                padding: 7px 12px;
                border-radius: 20px;
                margin: 4px;
                font-size: 13px;
            }}

            .footer {{
                text-align: center;
                margin-top: 35px;
                font-size: 13px;
                color: #777;
            }}

            a {{
                color: #1f3c88;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>

        <div class="container">
            <h1>Clasificador de Artículos ArXiv</h1>
            <p class="subtitle">
                API desplegada en Render para clasificar artículos científicos según su abstract limpio.
            </p>

            <div class="cards">
                <div class="card">
                    <h3>Accuracy</h3>
                    <p>{metricas["accuracy"]}</p>
                </div>
                <div class="card">
                    <h3>Precision</h3>
                    <p>{metricas["precision"]}</p>
                </div>
                <div class="card">
                    <h3>Recall</h3>
                    <p>{metricas["recall"]}</p>
                </div>
                <div class="card">
                    <h3>F1-Score</h3>
                    <p>{metricas["f1_score"]}</p>
                </div>
            </div>

            <p><strong>Registros usados para el modelo API:</strong> {metricas["registros_entrenamiento_api"]}</p>

            <p><strong>Categorías clasificadas:</strong></p>
            <div>
                {categorias_html}
            </div>

            <div class="section">
                <h2>Probar clasificación</h2>

                <label for="abstract">Ingrese un abstract limpio:</label>
                <textarea id="abstract" placeholder="Ejemplo: large language model transformer neural network text generation"></textarea>

                <div class="example">
                    Ejemplo: large language model transformer neural network text generation
                </div>

                <button onclick="predecir()">Predecir categoría</button>

                <div id="resultado" class="result">
                    <p>Categoría predicha:</p>
                    <div id="categoria" class="category"></div>
                    <div id="probabilidades" class="prob"></div>
                </div>

                <div id="error" class="error"></div>
            </div>

            <div class="section">
                <h2>Distribución de categorías</h2>
                <img src="/static/distribucion_categorias.png" class="chart" alt="Distribución de categorías">
            </div>

            <div class="section">
                <h2>Matriz de confusión</h2>
                <img src="/static/matriz_confusion.png" class="chart" alt="Matriz de confusión">
            </div>

            <div class="footer">
                Proyecto Final - ArXiv NLP | FastAPI + GitHub + Render |
                <a href="/docs">Ver documentación técnica</a>
            </div>
        </div>

        <script>
            async function predecir() {{
                const texto = document.getElementById("abstract").value;
                const resultado = document.getElementById("resultado");
                const error = document.getElementById("error");
                const categoria = document.getElementById("categoria");
                const probabilidades = document.getElementById("probabilidades");

                resultado.style.display = "none";
                error.style.display = "none";

                if (texto.trim() === "") {{
                    error.innerHTML = "Por favor, ingrese un abstract limpio.";
                    error.style.display = "block";
                    return;
                }}

                try {{
                    const response = await fetch("/predecir", {{
                        method: "POST",
                        headers: {{
                            "Content-Type": "application/json"
                        }},
                        body: JSON.stringify({{
                            abstract_limpio: texto
                        }})
                    }});

                    if (!response.ok) {{
                        throw new Error("Error en la predicción.");
                    }}

                    const data = await response.json();

                    categoria.innerHTML = data.categoria_predicha;

                    let htmlProb = "<strong>Probabilidades:</strong><br>";
                    for (const [clase, prob] of Object.entries(data.probabilidades)) {{
                        htmlProb += `${{clase}}: ${{(prob * 100).toFixed(2)}}%<br>`;
                    }}

                    probabilidades.innerHTML = htmlProb;
                    resultado.style.display = "block";

                }} catch (e) {{
                    error.innerHTML = "Ocurrió un error al consultar el modelo.";
                    error.style.display = "block";
                }}
            }}
        </script>

    </body>
    </html>
    """


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
