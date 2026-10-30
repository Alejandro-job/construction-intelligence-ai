# Construction Intelligence AI

**Prototipo independiente de portafolio** que integra modelos de machine learning, lectura documental y auditoría de cantidades BIM en un dashboard interactivo. Construido con Python, Streamlit, Plotly, PyTorch, scikit-learn, RapidOCR y IfcOpenShell.

> **Alcance:** demostración con datos públicos y cotizaciones ficticias. No es un sistema implementado en una constructora ni una integración activa con Revit/Dynamo o Rhino/Grasshopper. Los resultados no sustituyen validación comercial, fiscal ni técnica.

## Qué demuestra

- **Cotizaciones estructuradas:** lectura de PDF de demostración, extracción de folio, cliente, conceptos, cantidades y precios; cálculo determinista de importes; edición, aprobación humana y exportación JSON/CSV.
- **Document AI:** comparación de regresión logística y red MLP para clasificar *líneas ya transcritas* de recibos públicos CORD v1.
- **OCR neuronal:** RapidOCR frente a configuraciones Tesseract, con clasificación posterior y alertas.
- **Forecasting:** experimentos LSTM/GRU frente a persistencia sobre los índices PPI estadounidenses FRED WPU101 y WPU08; no representan precios en MXN.
- **BIM:** auditoría y extracción de cantidades de un modelo IFC público; reglas de coherencia y partidas bloqueadas en caso de alerta.

## Resultados experimentales (con alcance y límites)

| Tarea | Resultado observado | Interpretación |
|---|---|---|
| Forecasting acero | LSTM MAE 5.965 vs persistencia 7.825, mejora observada 23.77% | Prueba temporal de un paso; experimento exploratorio tras revisar varias veces ese periodo |
| Clasificación de líneas CORD v1 | MLP accuracy 91.77%, F1 macro 83.56% | Texto transcrito; **no** es exactitud end-to-end desde fotografía |
| RapidOCR | 40 de 51 líneas emparejadas (~78.4%) | Emparejamiento textual aproximado en solo 5 recibos, no CER/WER |
| IFC público | 123 elementos, 357 mediciones y 16 alertas | Evidencia de auditoría; no conexión directa a Revit o Rhino |

La regresión logística se eligió para el clasificador por **validación** (macro F1 81.44% vs 81.16% de MLP); el desempeño en prueba se muestra como referencia. La madera conserva el método de persistencia.

## Instalación en Windows

1. Tener Python 3.10 u 3.11 disponible.
2. Abrir una terminal PowerShell en esta carpeta.
3. Ejecutar:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abrir `http://localhost:8501`. La primera instalación puede descargar paquetes y modelos OCR.

## Demostración rápida: cotización con discrepancia

1. En el menú elegir **Cotizaciones estructuradas**.
2. Cargar `demos/cotizacion_demo_error.pdf`.
3. Revisar el ejemplo: 2 unidades × 3,000 = 6,000, pero el documento reporta 5,990.
4. Mostrar las alertas, corregir el dato con autorización del revisor y exportar el registro estructurado.
5. Comparar con `demos/cotizacion_demo_correcta.pdf`.

Es una plantilla de demostración con formato conocido: no se afirma que extraiga cualquier cotización de proveedor.

## Estructura

```text
app.py                 Dashboard Streamlit
quotation_pipeline.py  Extracción + validaciones deterministas
quotation_ui.py        Interfaz editable de cotizaciones
data/                   Métricas y resultados públicos procesados
models/                 Pesos y clasificador de demostración
assets/                 Evidencia gráfica de experimentos
demos/                  Cotizaciones sintéticas y JSON de ejemplo
schemas/                Contrato JSON de cotización
tests/                  Pruebas automatizadas de reglas de cotización
experimentos/           Scripts históricos de entrenamiento exploratorio
```

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Procedencia y ética

- FRED: índices WPU101 y WPU08 — [Federal Reserve Economic Data](https://fred.stlouisfed.org/).
- Recibos CORD v1 — [dataset público](https://huggingface.co/datasets/Heliosoph/CORD); las imágenes reproducidas son de sus ejemplos públicos; revisar sus términos/licencia para reutilización.
- Modelo BIM IFC de acceso público, con cantidades observadas en el experimento.
- Los PDFs en `demos/` son **ficticios**, sin datos de clientes reales.

Los módulos de OCR, clasificación, forecasting y BIM se evalúan con métricas distintas: no existe una “precisión global” de este prototipo. Para utilizarlo en producción se requieren documentos reales autorizados, pruebas independientes, protección de datos, monitoreo y validación humana de cifras.

## Colab

Los scripts de `experimentos/` documentan versiones **anteriores** del entrenamiento de series temporales. Los resultados finales del dashboard corresponden a experimentos posteriores con retornos logarítmicos y auditoría de métricas: **no** se presenta esos scripts históricos como reproducción exacta de los pesos empaquetados.
