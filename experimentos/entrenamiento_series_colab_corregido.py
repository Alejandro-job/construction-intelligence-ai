# -*- coding: utf-8 -*-
"""Construction Intelligence AI: entrenamiento de redes GRU y LSTM en CPU Colab.
Proyecto personal de portafolio entrenado con datos públicos.
Fuentes de ejemplo: FRED WPU101 (acero) y WPU08 (madera).
Salidas: /content/construction_intelligence_ai y ZIP de métricas/modelos.
"""

# ===== ETAPA 1 =====
# 1) Entorno y datos públicos (no instala ni modifica pandas)
from pathlib import Path
from io import StringIO
import json, copy, random, time, zipfile
import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from IPython.display import display

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(2)
DEVICE = 'cpu'  # CPU gratis
OUT = Path('/content/construction_intelligence_ai')
OUT.mkdir(parents=True, exist_ok=True)
SERIES = {'Acero': 'WPU101', 'Madera': 'WPU08'}


# Descarga tolerante a fallos, con cache verificable.
# Fuentes: FRED directamente y espejo DBnomics (agrega series FRED).
# Si ambas conexiones fallan, acepta un CSV real exportado por FRED.
from io import BytesIO

CACHE = OUT / "cache_fuentes"
CACHE.mkdir(parents=True, exist_ok=True)


def _validar_serie(s, series_id):
    s = pd.Series(pd.to_numeric(s.values, errors="coerce"),
                  index=pd.to_datetime(s.index, errors="coerce"), name=series_id)
    s = s[s.index.notna()].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s.loc["2000-01-01":]) < 120 or (s.dropna() <= 0).any():
        raise ValueError(f"Serie {series_id}: menos de 120 meses validos desde 2000 o valores invalidos")
    return s


def _parsear_fred_csv(data, series_id):
    df = pd.read_csv(BytesIO(data), encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    fecha = next((c for c in ("DATE", "observation_date", "date", "fecha")
                  if c in df.columns), None)
    valor = next((c for c in (series_id, "value", "VALUE")
                  if c in df.columns), None)
    if not fecha or not valor:
        raise ValueError(f"CSV de {series_id}: se esperan DATE y {series_id}; columnas: {list(df.columns)}")
    return _validar_serie(pd.Series(df[valor].values, index=df[fecha].values), series_id)


def _parsear_dbnomics(data, series_id):
    js = json.loads(data.decode("utf-8-sig"))
    docs = js.get("series", {}).get("docs", [])
    if not docs:
        raise ValueError(f"DBnomics no devolvio observaciones para {series_id}")
    serie = docs[0]
    periodos = serie.get("period", [])
    valores = serie.get("value", [])
    if len(periodos) != len(valores) or not periodos:
        raise ValueError("Periodos y valores incompletos en DBnomics")
    return _validar_serie(pd.Series(valores, index=periodos), series_id)


def cargar_fred(series_id):
    cache = CACHE / f"{series_id}.csv"
    if cache.exists():
        try:
            s = _parsear_fred_csv(cache.read_bytes(), series_id)
            print(f"✅ {series_id}: {len(s)} observaciones desde caché")
            return s
        except Exception as exc:
            print(f"Caché invalida ({series_id}): {exc}")

    fuentes = [
        ("DBnomics", f"https://api.db.nomics.world/v22/series/FRED/{series_id}?observations=1", _parsear_dbnomics),
        ("FRED", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", _parsear_fred_csv),
    ]
    errores = []
    for origen, url, parsear in fuentes:
        try:
            print(f"Descargando {series_id} desde {origen}...", flush=True)
            r = requests.get(url, timeout=(8, 15), headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            s = parsear(r.content, series_id)
            pd.DataFrame({"DATE": s.index, series_id: s.values}).to_csv(cache, index=False)
            print(f"✅ {series_id}: {len(s)} meses desde {origen}; guardado en caché", flush=True)
            return s
        except Exception as exc:
            errores.append(f"{origen}: {type(exc).__name__}: {str(exc)[:130]}")
            print(f"⚠️ {origen} no disponible: {type(exc).__name__}", flush=True)

    print("No se pudieron descargar datos de las fuentes automáticas:")
    for e in errores:
        print(" -", e)
    print(f"Descarga {series_id} en https://fred.stlouisfed.org/series/{series_id}")
    print(f"Selecciona Download > CSV y sube el archivo {series_id}.csv")
    try:
        from google.colab import files
        subidos = files.upload()
        candidatos = [data for nombre, data in subidos.items()
                      if nombre.lower().endswith('.csv')]
        for data in candidatos:
            try:
                s = _parsear_fred_csv(data, series_id)
                pd.DataFrame({"DATE": s.index, series_id: s.values}).to_csv(cache, index=False)
                print(f"✅ {series_id}: CSV real importado correctamente")
                return s
            except Exception as exc:
                print("CSV incompatible:", exc)
    except ImportError:
        pass
    raise RuntimeError(f"No se puede entrenar sin datos auténticos de {series_id}. Reintenta o sube su CSV de FRED.")

historicos = pd.concat({nombre: cargar_fred(sid) for nombre, sid in SERIES.items()}, axis=1)
historicos = historicos.loc['2000-01-01':].resample('MS').last()
# Relleno causal: solo propaga la observación anterior, nunca la futura.
historicos = historicos.ffill().dropna()
assert len(historicos) >= 120, 'Pocas observaciones para el experimento.'
assert (historicos > 0).all().all(), 'Los índices deben ser positivos.'
print('Meses disponibles:', len(historicos), '|', historicos.index.min().date(), 'a', historicos.index.max().date())
display(historicos.tail())
historicos.to_csv(OUT/'indices_historicos_fred.csv', index_label='fecha')
(OUT/'FUENTES.txt').write_text('Series: WPU101 y WPU08. Fuentes: FRED; espejo de datos DBnomics. Observaciones recuperadas de la fuente pública o CSV del usuario. Índices PPI estadounidenses, NO precios mexicanos.', encoding='utf-8')


# ===== ETAPA 2 =====
# 2) Separación temporal correcta, ventanas de 12 meses, scaler SOLO en entrenamiento
LOOKBACK = 12
N = len(historicos)
train_end = int(N * 0.70)
val_end = int(N * 0.85)
assert train_end > LOOKBACK + 30 and val_end < N-12

valores_reales = historicos.to_numpy(dtype=np.float64)
valores_log = np.log(valores_reales)
scaler = StandardScaler().fit(valores_log[:train_end])
z = scaler.transform(valores_log).astype(np.float32)

X, Y, T = [], [], []
for t in range(LOOKBACK, N):
    X.append(z[t-LOOKBACK:t])
    Y.append(z[t])
    T.append(t)
X = np.stack(X)
Y = np.stack(Y)
T = np.array(T)
mask_train = T < train_end
mask_val = (T >= train_end) & (T < val_end)
mask_test = T >= val_end

print('Train:', int(mask_train.sum()), 'ventanas | Val:', int(mask_val.sum()), '| Test:', int(mask_test.sum()))
print('Evaluación sin mezclar cronologías; test empieza:', historicos.index[val_end].date())

def volver_a_indices(y_norm):
    return np.exp(scaler.inverse_transform(np.asarray(y_norm)))

# Baseline de un paso: último valor observado hasta ese mes (sin mirar el futuro)
y_real = valores_reales[T[mask_test]]
y_naive = valores_reales[T[mask_test] - 1]

class RedRecurrente(nn.Module):
    def __init__(self, tipo='GRU', n_features=2, hidden=32):
        super().__init__()
        RNN = nn.GRU if tipo == 'GRU' else nn.LSTM
        self.rnn = RNN(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)
    def forward(self, x):
        salida, _ = self.rnn(x)
        return self.head(salida[:, -1, :])


def entrenar(tipo, max_epochs=65, paciencia=10):
    red = RedRecurrente(tipo).to(DEVICE)
    optim = torch.optim.Adam(red.parameters(), lr=0.002, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    x_train = torch.tensor(X[mask_train], dtype=torch.float32)
    y_train = torch.tensor(Y[mask_train], dtype=torch.float32)
    x_val = torch.tensor(X[mask_val], dtype=torch.float32)
    y_val = torch.tensor(Y[mask_val], dtype=torch.float32)
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    mejor, estado, sin_mejora = float('inf'), None, 0
    historia=[]
    for epoca in range(max_epochs):
        red.train()
        for xb, yb in tr:
            optim.zero_grad()
            loss = loss_fn(red(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            optim.step()
        red.eval()
        with torch.no_grad():
            v = float(loss_fn(red(x_val), y_val).item())
        historia.append(v)
        if v < mejor - 1e-5:
            mejor, estado, sin_mejora = v, copy.deepcopy(red.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(estado)
    red.eval()
    return red, historia

redes={}
historias={}
for tipo in ['GRU', 'LSTM']:
    tic = time.time()
    redes[tipo], historias[tipo] = entrenar(tipo)
    print(f'{tipo}: {len(historias[tipo])} épocas, {time.time()-tic:.1f} s, mejor MSE validación={min(historias[tipo]):.5f}')


# ===== ETAPA 3 =====
# 3) Prueba de predicción a un mes en periodo NO utilizado para entrenamiento

def inferir_test(red):
    with torch.no_grad():
        y = red(torch.tensor(X[mask_test], dtype=torch.float32)).numpy()
    return volver_a_indices(y)

pred_test = {'Persistencia': y_naive}
for tipo, red in redes.items():
    pred_test[tipo] = inferir_test(red)

filas=[]
for nombre, pred in pred_test.items():
    for j, material in enumerate(historicos.columns):
        filas.append({
            'modelo':nombre, 'material':material,
            'mae_indice':round(float(mean_absolute_error(y_real[:,j], pred[:,j])), 4),
            'rmse_indice':round(float(np.sqrt(mean_squared_error(y_real[:,j], pred[:,j]))), 4),
            'meses_test':len(y_real)})
metricas = pd.DataFrame(filas)
display(metricas)
metricas.to_csv(OUT/'metricas_test.csv', index=False)

pred_df = pd.DataFrame({'fecha':historicos.index[T[mask_test]]})
for j, material in enumerate(historicos.columns):
    pred_df[f'{material}_real'] = y_real[:, j]
    for nombre, valores in pred_test.items():
        pred_df[f'{material}_{nombre}'] = valores[:,j]
pred_df.to_csv(OUT/'predicciones_test.csv', index=False)

fig, axs = plt.subplots(2,1, figsize=(12,7), sharex=True)
for j, mat in enumerate(historicos.columns):
    axs[j].plot(pred_df['fecha'], pred_df[f'{mat}_real'], label='Real', linewidth=2)
    for mod in pred_test:
        axs[j].plot(pred_df['fecha'], pred_df[f'{mat}_{mod}'], label=mod, alpha=.8)
    axs[j].set_title(f'{mat}: periodo de prueba (índice, NO precio en MXN)')
    axs[j].legend()
    axs[j].grid(alpha=.2)
plt.tight_layout()
plt.savefig(OUT/'comparacion_test.png', dpi=150)
plt.show()


# ===== ETAPA 4 =====
# 4) Escenario de forecasting a 6 meses; los valores futuros no son datos observados
# Selección del mejor modelo neural usando SOLAMENTE validación.
# El baseline de persistencia también se reporta para evitar inflar resultados.

# Evaluar calidad en validación, no elegir usando test
metricas_validacion={}
for nombre, red in redes.items():
    with torch.no_grad():
        val_norm = red(torch.tensor(X[mask_val], dtype=torch.float32)).numpy()
    val_pred = volver_a_indices(val_norm)
    val_real = valores_reales[T[mask_val]]
    metricas_validacion[nombre] = float(np.mean(np.abs(val_real - val_pred)))

red_elegida = min(metricas_validacion, key=metricas_validacion.get)
red = redes[red_elegida]
print('Red elegida por validación:', red_elegida)

# Pronóstico recursivo: predicción mes 2 usa predicción mes 1. Error puede acumularse.
ventana = z[-LOOKBACK:].copy()
futuros=[]
for h in range(6):
    with torch.no_grad():
        siguiente = red(torch.tensor(ventana[None], dtype=torch.float32)).numpy()[0]
    futuros.append(siguiente)
    ventana = np.concatenate([ventana[1:], siguiente[None]], axis=0)

pred_seis = volver_a_indices(np.array(futuros))
fechas = pd.date_range(historicos.index[-1]+pd.offsets.MonthBegin(1), periods=6, freq='MS')
pronostico = pd.DataFrame(pred_seis, columns=[f'{x}_indice_predicho' for x in historicos.columns])
pronostico.insert(0,'fecha',fechas)
pronostico['modelo_red'] = red_elegida
for x in historicos.columns:
    pronostico[f'{x}_persistencia'] = float(historicos[x].iloc[-1])

print('Pronóstico experimental; no equivale a precio real de compra ni costo de obra.')
display(pronostico)
pronostico.to_csv(OUT/'pronostico_seis_meses.csv', index=False)

# Guardar modelos PyTorch con sus transformaciones y metadatos
for nombre, red in redes.items():
    torch.save({
        'state_dict':red.state_dict(),
        'tipo':nombre, 'lookback':LOOKBACK,
        'features':list(historicos.columns),
        'scaler_mean':scaler.mean_.tolist(),
        'scaler_scale':scaler.scale_.tolist(),
        'train_fin':str(historicos.index[train_end-1].date()),
        'fecha_ultima_observacion':str(historicos.index[-1].date()),
        'dataset_FRED':SERIES,
    }, OUT/f'modelo_{nombre.lower()}.pt')

reporte = {
    'titulo':'Construction Intelligence AI - Neural Forecasting',
    'fuentes':SERIES,
    'observaciones':N,
    'ventanas':LOOKBACK,
    'train_fin':str(historicos.index[train_end-1].date()),
    'validacion_fin':str(historicos.index[val_end-1].date()),
    'test_inicio':str(historicos.index[val_end].date()),
    'ultimo_dato':str(historicos.index[-1].date()),
    'seleccion_validacion':red_elegida,
    'mae_val_promedio':metricas_validacion,
    'aviso':'Índices PPI estadounidenses; no predicen precios unitarios, consumo de obra ni costos mexicanos. Revisar revisiones de series y fechas de disponibilidad.'
}
(OUT/'reporte_metodologia.json').write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding='utf-8')

paquete = Path('/content/Construction_Intelligence_AI_Redes_Neuronales.zip')
with zipfile.ZipFile(paquete, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.iterdir():
        if p.is_file():
            zf.write(p, p.name)
print('Archivos listos:', paquete)

try:
    from google.colab import files
    files.download(str(paquete))
except ImportError:
    print('Fuera de Colab: archivo guardado localmente.')

