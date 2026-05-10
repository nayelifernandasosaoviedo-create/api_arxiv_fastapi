from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="API de Clasificación ArXiv",
    description="API para clasificar artículos científicos de ArXiv según su abstract limpio.",
    version="1.0"
)

# Cargar modelo entrenado
modelo = joblib.load("modelo_arxiv_logistic_regression.pkl")


class Articulo(BaseModel):
    abstract_limpio: str


@app.get("/", response_class=HTMLResponse)
def inicio():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clasificador ArXiv</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f9;
                margin: 0;
                padding: 0;
            }

            .container {
                max-width: 850px;
                margin: 50px auto;
                background: white;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.12);
            }

            h1 {
                text-align: center;
                color: #1f3c88;
                margin-bottom: 10px;
            }

            .subtitle {
                text-align: center;
                color: #555;
                margin-bottom: 30px;
            }

            label {
                font-weight: bold;
                color: #333;
            }

            textarea {
                width: 100%;
                height: 170px;
                margin-top: 10px;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid #ccc;
                font-size: 15px;
                resize: vertical;
            }

            button {
                margin-top: 20px;
                width: 100%;
                background: #1f3c88;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                border-radius: 10px;
                cursor: pointer;
            }

            button:hover {
                background: #162d66;
            }

            .result {
                margin-top: 25px;
                padding: 20px;
                background: #eef4ff;
                border-left: 5px solid #1f3c88;
                border-radius: 10px;
                display: none;
            }

            .error {
                margin-top: 25px;
                padding: 20px;
                background: #ffecec;
                border-left: 5px solid #cc0000;
                border-radius: 10px;
                display: none;
                color: #990000;
            }

            .category {
                font-size: 24px;
                font-weight: bold;
                color: #1f3c88;
            }

            .prob {
                margin-top: 10px;
                font-size: 14px;
                color: #333;
            }

            .footer {
                text-align: center;
                margin-top: 30px;
                font-size: 13px;
                color: #777;
            }

            .example {
                margin-top: 10px;
                font-size: 13px;
                color: #555;
                background: #f7f7f7;
                padding: 10px;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>

        <div class="container">
            <h1>Clasificador de Artículos ArXiv</h1>
            <p class="subtitle">
                Ingrese un abstract limpio y el modelo predecirá la categoría del artículo.
            </p>

            <label for="abstract">Abstract limpio:</label>
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

            <div class="footer">
                Proyecto Final - Clasificación de artículos científicos ArXiv con FastAPI y Render
            </div>
        </div>

        <script>
            async function predecir() {
                const texto = document.getElementById("abstract").value;
                const resultado = document.getElementById("resultado");
                const error = document.getElementById("error");
                const categoria = document.getElementById("categoria");
                const probabilidades = document.getElementById("probabilidades");

                resultado.style.display = "none";
                error.style.display = "none";

                if (texto.trim() === "") {
                    error.innerHTML = "Por favor, ingrese un abstract limpio.";
                    error.style.display = "block";
                    return;
                }

                try {
                    const response = await fetch("/predecir", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            abstract_limpio: texto
                        })
                    });

                    if (!response.ok) {
                        throw new Error("Error en la predicción.");
                    }

                    const data = await response.json();

                    categoria.innerHTML = data.categoria_predicha;

                    let htmlProb = "<strong>Probabilidades:</strong><br>";
                    for (const [clase, prob] of Object.entries(data.probabilidades)) {
                        htmlProb += `${clase}: ${(prob * 100).toFixed(2)}%<br>`;
                    }

                    probabilidades.innerHTML = htmlProb;
                    resultado.style.display = "block";

                } catch (e) {
                    error.innerHTML = "Ocurrió un error al consultar el modelo.";
                    error.style.display = "block";
                }
            }
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
