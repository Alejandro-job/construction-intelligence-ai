"""Construction Intelligence AI | local portfolio dashboard.
Results derive only from bundled publicly sourced experiments. No company data.
Run: python -m streamlit run app.py
"""

from __future__ import annotations

import io
import json
import html
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
F = ROOT / "data" / "forecasting"
D = ROOT / "data" / "document_ai"
O = ROOT / "data" / "ocr"
B = ROOT / "data" / "bim"
M = ROOT / "models"
A = ROOT / "assets"
COLORS = {"real":"#0f766e", "neural":"#6857e8", "base":"#f59e0b", "pale":"#94a3b8"}

st.set_page_config(page_title="Construction Intelligence AI", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root {color-scheme: light;}
.block-container {max-width:1440px; padding-top:1.4rem; padding-bottom:4rem;}
[data-testid="stSidebar"] {background:#f5f7fb; border-right:1px solid #e4e9f1;}
[data-testid="stSidebar"] h2 {letter-spacing:-0.02em;}
h1,h2,h3 {letter-spacing:-0.035em; color:#0f172a;}
h1 {font-size:2.25rem !important;}
small,.small {color:#64748b;}
.hero {padding:1.55rem 1.75rem;background:linear-gradient(115deg,#0b1732 0%,#18295d 65%,#0f756f 155%);
  color:white;border-radius:20px;margin-bottom:1.4rem;}
.hero h1 {color:white !important; margin:.2rem 0 .35rem 0;font-size:2.0rem!important;}
.hero p {color:#cbd5e1;margin:0;max-width:850px;line-height:1.55;}
.eyebrow {text-transform:uppercase;font-size:.70rem;letter-spacing:.14em;font-weight:800;color:#8fdcd1;}
.metric-card {background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px 18px 14px;
box-shadow:0 4px 18px rgba(15,23,42,.035);margin:0 0 10px;min-height:124px;}
.metric-card .label {color:#64748b;font-size:.76rem;font-weight:700;}
.metric-card .value {color:#0f172a;font-size:2.0rem;font-weight:780;letter-spacing:-.035em;margin:.13rem 0;}
.metric-card .sub {color:#64748b;font-size:.71rem;line-height:1.25;}
.section-note {background:#f3f6fa;border-left:3px solid #64748b;padding:12px 16px;color:#475569;
border-radius:0 9px 9px 0;font-size:.88rem;margin:12px 0 20px;}
[data-testid="stFileUploader"] {background:#fbfdff;border:1px dashed #b6c3d3;border-radius:12px;padding:8px;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:8px;}
[data-testid="stTabs"] button[role="tab"] {background:#f1f5f9;border-radius:8px 8px 0 0;padding:8px 15px;}
.architecture {display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; margin:10px 0 22px;}
.architecture > div:not(.arch-arrow){flex:1 1 170px; background:linear-gradient(140deg,#f8fafc,#eff6ff);border:1px solid #dce6f3;border-radius:14px;padding:15px 12px;display:flex;flex-direction:column;gap:6px;min-width:130px;}
.architecture strong{color:#0f172a;font-size:0.92rem;}.architecture small{color:#64748b;font-size:.75rem;}
.arch-icon{color:#4f46e5;font-size:1.35rem;}.arch-arrow{display:flex;align-items:center;color:#94a3b8;font-size:1.4rem;font-weight:700;}
.evidence {border:1px solid #c7d2fe;background:#eef2ff;border-radius:14px;padding:16px 18px;}
.evidence h4{margin:0 0 8px;color:#1e1b4b;}.evidence p{margin:0;color:#312e81;font-size:.87rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def csv(path: str, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    p = ROOT / path
    df = pd.read_csv(p)
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def js(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric(label: str, value: str, sub: str) -> None:
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


def fig_layout(fig, height=400):
    fig.update_layout(
        height=height, margin=dict(l=5,r=12,t=35,b=10),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#334155", size=12),
        hovermode="x unified",legend=dict(orientation="h",y=1.13,x=0),
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#e8edf3")
    )
    return fig


def as_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


forecast_sum = csv("data/forecasting/resumen_modelos.csv")
forecast_pred = csv("data/forecasting/predicciones_auditadas.csv", ("fecha",))
forecast_future = csv("data/forecasting/forecast_experimental.csv", ("fecha",))
forecast_year = csv("data/forecasting/metricas_por_anio.csv")
doc_scores = csv("data/document_ai/metricas_document_ai.csv")
ocr_comp = csv("data/ocr/comparacion_ocr_neuronal.csv")
bim_budget = csv("data/bim/prepresupuesto_bim.csv")
bim_alerts = csv("data/bim/alertas_bim.csv")
bim_info = js("data/bim/resumen.json")

with st.sidebar:
    st.markdown("## 🏗️ Construction Intelligence")
    st.caption("AI PORTFOLIO · 2026 · EDICIÓN PRO")
    st.divider()
    page = st.radio("Explorar módulos", [
        "Resumen ejecutivo", "Centro de métricas IA", "Forecasting neuronal", "Document AI",
        "OCR neuronal", "Automatización BIM", "Cotizaciones estructuradas", "Laboratorio interactivo"
    ], label_visibility="collapsed")
    st.divider()
    st.caption("FUENTES ABIERTAS")
    st.markdown("FRED WPU101 / WPU08  \\\nCORD v1 · IFC público")
    st.caption("Proyecto independiente de demostración. Sin conexión activa a Revit, Dynamo, Rhino o Grasshopper.")

st.markdown('<div class="hero"><div class="eyebrow">Open-data AI portfolio</div><h1>Construction Intelligence AI</h1><p>Forecasting de materiales · Automatización documental · OCR neuronal · Auditoría de modelos BIM. Resultados reproducibles con datos públicos y controles de calidad.</p></div>', unsafe_allow_html=True)

if page == "Resumen ejecutivo":
    st.subheader("Indicadores principales")
    cols = st.columns(4)
    acero = forecast_sum[forecast_sum.material=="Acero"].iloc[0]
    doc_mlp = doc_scores[(doc_scores.modelo=="MLP") & (doc_scores.periodo=="prueba")].iloc[0]
    rapid = ocr_comp[ocr_comp.modo=="RapidOCR_neuronal"].iloc[0]
    with cols[0]: metric("Mejora observada · acero", f"{acero.mejora_porcentaje:.1f}%", "LSTM vs persistencia · 48 meses de prueba")
    with cols[1]: metric("Document AI · MLP", f"{doc_mlp.accuracy*100:.1f}%", "Accuracy en transcripciones CORD · 1,336 líneas")
    with cols[2]: metric("RapidOCR", f"{rapid.cobertura_textual_aproximada*100:.1f}%", "Cobertura textual aproximada · cinco recibos")
    with cols[3]: metric("Alertas BIM", str(len(bim_alerts)), "Revisión de cantidades · IFC público")
    note("<b>Interpretación:</b> no son métricas de un mismo sistema de extremo a extremo. Cada módulo tiene un dataset, una evaluación y limitaciones propias. Los índices PPI estadounidenses no representan precios en México.")
    c1, c2 = st.columns([1.2,1])
    with c1:
        st.markdown("#### Predicción mensual de acero")
        tmp = forecast_pred[forecast_pred.material=="Acero"].copy()
        fig = go.Figure()
        for col, label, color in [("real","Real",COLORS["real"]),("persistencia","Persistencia",COLORS["base"]),("prediccion","LSTM",COLORS["neural"])]:
            fig.add_scatter(x=tmp.fecha,y=tmp[col],mode="lines",name=label,line=dict(color=color,width=2.4 if col=="real" else 2))
        fig.update_yaxes(title="Índice PPI")
        st.plotly_chart(fig_layout(fig),use_container_width=True)
    with c2:
        st.markdown("#### Cobertura de OCR")
        tmp = ocr_comp.copy()
        tmp["cobertura_%"] = tmp["cobertura_textual_aproximada"] * 100
        fig = px.bar(tmp, x="cobertura_%", y="modo", orientation="h", text="cobertura_%", color="modo", color_discrete_sequence=["#94a3b8","#cbd5e1","#64748b",COLORS["real"]])
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_xaxes(title="% de líneas emparejadas",range=[0,100]); fig.update_yaxes(title=None)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig_layout(fig),use_container_width=True)
    st.markdown("#### Arquitectura de la solución")
    st.markdown("""
    <div class="architecture">
      <div><span class="arch-icon">▤</span><strong>Datos públicos</strong><small>FRED · CORD · IFC</small></div>
      <div class="arch-arrow">→</div>
      <div><span class="arch-icon">✦</span><strong>Modelos de IA</strong><small>LSTM · MLP · OCR</small></div>
      <div class="arch-arrow">→</div>
      <div><span class="arch-icon">◈</span><strong>Validación</strong><small>MAE · F1 · Alertas</small></div>
      <div class="arch-arrow">→</div>
      <div><span class="arch-icon">▧</span><strong>Decisiones auditables</strong><small>Revisión humana</small></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("#### Qué es demostración y qué está validado")
    st.dataframe(pd.DataFrame([
        {"Componente":"Forecasting (acero)","Evidencia":"MAE 5.965 vs 7.825 · LSTM","Alcance":"Mejora exploratoria; prueba reutilizada"},
        {"Componente":"Document AI","Evidencia":"F1 macro 0.825 · logística en prueba","Alcance":"Líneas con transcripción CORD; no facturas mexicanas"},
        {"Componente":"OCR neuronal","Evidencia":"40/51 líneas emparejadas","Alcance":"5 recibos; coincidencia aproximada, no CER/WER"},
        {"Componente":"BIM / IFC","Evidencia":"123 elementos · 16 alertas","Alcance":"Archivo público, cantidades sin precios aprobados"},
    ]), hide_index=True, use_container_width=True)

elif page == "Centro de métricas IA":
    st.subheader("Centro de métricas y evaluación de modelos")
    st.caption("Cuatro módulos, cuatro metodologías distintas. Cada indicador conserva su fuente, denominador y limitación.")
    acero = forecast_sum[forecast_sum.material=="Acero"].iloc[0]
    madera = forecast_sum[forecast_sum.material=="Madera"].iloc[0]
    doc_log = doc_scores[(doc_scores.modelo=="LOGISTICA") & (doc_scores.periodo=="prueba")].iloc[0]
    doc_mlp = doc_scores[(doc_scores.modelo=="MLP") & (doc_scores.periodo=="prueba")].iloc[0]
    rapid = ocr_comp[ocr_comp.modo=="RapidOCR_neuronal"].iloc[0]
    num_elements = int(bim_info.get("elementos", 123))
    n_no_qto = int(bim_info.get("elementos_sin_qto", 28))

    st.markdown("#### Resultados destacados de tus experimentos")
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Acero · mejora LSTM", f"{acero.mejora_porcentaje:.2f}%", "MAE 5.965 frente a 7.825 · prueba histórica")
    with c2: metric("Document AI · MLP", f"{doc_mlp.accuracy*100:.2f}%", "Accuracy, líneas ya transcritas")
    with c3: metric("RapidOCR · cobertura", f"{rapid.cobertura_textual_aproximada*100:.2f}%", "40/51 coincidencias textuales aproximadas")
    with c4: metric("IFC · registros de cantidades", str(bim_info.get("registros_cantidades",357)), "123 elementos · 16 alertas de coherencia")

    st.markdown("""<div class="evidence"><h4>🔬 Evidencia antes que promesas</h4>
    <p>Los modelos se comparan contra referencias sencillas y muestran también los errores. El forecasting se evaluó con un periodo histórico examinado durante el desarrollo; OCR se probó en cinco recibos ya vistos. Ningún indicador permite afirmar exactitud comercial ni emitir presupuestos sin revisión.</p></div>""",unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["🧠 Forecasting", "📄 Document AI", "👁️ OCR neuronal", "🏗️ BIM / IFC"])

    with t1:
        st.markdown("#### Comparación justa con persistencia")
        c1,c2,c3,c4=st.columns(4)
        with c1: metric("Acero · LSTM MAE", f"{acero.MAE_modelo:.3f}", "Puntos de índice PPI")
        with c2: metric("Acero · línea base MAE", f"{acero.MAE_persistencia:.3f}", "Persistencia · menor es mejor")
        with c3: metric("Acierto de dirección", f"{acero.acierto_direccion_porcentaje:.2f}%", "35 de 48 meses · acero")
        with c4: metric("Madera · línea base MAE", f"{madera.MAE_persistencia:.3f}", "Persistencia seleccionada")
        left,right=st.columns([1,1.15])
        with left:
            mae = pd.DataFrame({"Material":["Acero","Acero","Madera","Madera"],"Modelo":["Persistencia","LSTM","Persistencia","Seleccionado"],"MAE":[float(acero.MAE_persistencia),float(acero.MAE_modelo),float(madera.MAE_persistencia),float(madera.MAE_modelo)]})
            fig=px.bar(mae,x="Material",y="MAE",color="Modelo",barmode="group",text="MAE",color_discrete_map={"Persistencia":"#f59e0b","LSTM":"#6857e8","Seleccionado":"#0f766e"})
            fig.update_traces(texttemplate="%{text:.2f}",textposition="outside")
            st.plotly_chart(fig_layout(fig,350),use_container_width=True)
            st.caption("MAE: error absoluto medio en puntos de índice, no MXN. La madera no mejoró a la referencia.")
        with right:
            material = st.selectbox("Analizar errores mensuales", ["Acero", "Madera"], key="metro_material")
            datos = forecast_pred[forecast_pred.material==material].copy()
            datos["Error L1 · seleccionado"] = (datos.real-datos.prediccion).abs()
            datos["Error L1 · referencia"] = (datos.real-datos.persistencia).abs()
            fig=go.Figure()
            fig.add_scatter(x=datos.fecha,y=datos["Error L1 · seleccionado"],name="Seleccionado",mode="lines+markers",line_color=COLORS["neural"])
            fig.add_scatter(x=datos.fecha,y=datos["Error L1 · referencia"],name="Persistencia",mode="lines",line_color=COLORS["base"])
            fig.update_yaxes(title="Error absoluto |real - predicción|")
            st.plotly_chart(fig_layout(fig,350),use_container_width=True)
        st.info(f"Intervalo bootstrap exploratorio de mejora en acero: {acero.IC95_aprox_inferior:.2f}% a {acero.IC95_aprox_superior:.2f}%. Incluye 0: todavía hay incertidumbre de generalización.")

    with t2:
        st.markdown("#### Calidad de clasificación por categoría")
        import sklearn.metrics as skm
        real = csv("data/document_ai/predicciones_documentales.csv")
        clases = ["ARTICULO","PRECIO","CANTIDAD","SUBTOTAL","TOTAL","OTRO"]
        informe = skm.classification_report(real.categoria,real.prediccion,labels=clases,output_dict=True,zero_division=0)
        byclass = pd.DataFrame([{"Categoria":n,"Precision":informe[n]["precision"],"Recall":informe[n]["recall"],"F1":informe[n]["f1-score"],"Lineas":int(informe[n]["support"])} for n in clases])
        c1,c2,c3=st.columns(3)
        with c1: metric("Selección por validación","Logística", "F1 macro validación 81.44%")
        with c2: metric("F1 prueba · logística",f"{doc_log.macro_F1*100:.2f}%","Modelo elegido sin mirar la prueba")
        with c3: metric("F1 prueba · MLP",f"{doc_mlp.macro_F1*100:.2f}%","Red conservada como comparativa")
        left,right=st.columns([1.1,1])
        with left:
            byclass["F1 (%)"] = byclass.F1*100
            fig=px.bar(byclass,x="F1 (%)",y="Categoria",orientation="h",text="F1 (%)",color="F1 (%)",color_continuous_scale=[[0,"#db2777"],[0.5,"#7c3aed"],[1,"#0f766e"]])
            fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
            fig.update_xaxes(range=[0,110]);fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_layout(fig,380),use_container_width=True)
        with right:
            cm=skm.confusion_matrix(real.categoria,real.prediccion,labels=clases)
            fig=px.imshow(cm,x=clases,y=clases,text_auto=True,color_continuous_scale="Blues",aspect="auto")
            fig.update_xaxes(title="Predicho");fig.update_yaxes(title="Verdadero")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_layout(fig,380),use_container_width=True)
        st.dataframe(byclass[["Categoria","Precision","Recall","F1","Lineas"]].round(3),hide_index=True,use_container_width=True)
        st.warning("La categoría OTRO tiene el F1 más bajo. El clasificador trabaja con transcripciones del dataset CORD v1, no con facturas CFDI ni precios de obras en México.")

    with t3:
        st.markdown("#### Resultados comparables de reconocimiento textual")
        d=ocr_comp.copy(); d["Cobertura (%)"]=100*d.cobertura_textual_aproximada
        d["Clasificación correcta / 51 (%)"]=100*d.clases_correctas/d.lineas_referencia
        left,right=st.columns(2)
        with left:
            fig=go.Figure()
            fig.add_bar(x=d.modo,y=d["Cobertura (%)"],name="Cobertura aproximada",marker_color="#0f766e",text=[f"{v:.1f}%" for v in d["Cobertura (%)"]],textposition="outside")
            fig.add_scatter(x=d.modo,y=d["Clasificación correcta / 51 (%)"],name="Correctamente clasificadas / 51",mode="lines+markers",line=dict(color="#6857e8",width=3))
            fig.update_yaxes(title="Porcentaje de 51 líneas referencia",range=[0,100])
            st.plotly_chart(fig_layout(fig,380),use_container_width=True)
        with right:
            nr=d[d.modo=="RapidOCR_neuronal"].iloc[0]
            metric("RapidOCR · líneas reconocidas",f"{int(nr.lineas_ocr)}", "Segmentos OCR: pueden dividir una sola línea")
            metric("RapidOCR · emparejamientos",f"{int(nr.coincidencias)}/51", "Coincidencia aproximada con la transcripción")
            metric("RapidOCR · categorías correctas",f"{int(nr.clases_correctas)}/51", "Entre emparejamientos: 32/40 correctos")
        st.dataframe(d[["modo","lineas_ocr","coincidencias","clases_correctas","Cobertura (%)"]].round(2),hide_index=True,use_container_width=True)
        st.caption("Comparación exploratoria en cinco imágenes. Las puntuaciones son aproximaciones basadas en emparejamientos, no CER, WER ni precisión financiera.")

    with t4:
        st.markdown("#### Trazabilidad y control de calidad del IFC")
        c1,c2,c3,c4=st.columns(4)
        with c1: metric("Elementos IFC",f"{num_elements}","Modelo público")
        with c2: metric("Con QTO",f"{num_elements-n_no_qto}",f"{(num_elements-n_no_qto)/num_elements*100:.1f}% de elementos")
        with c3: metric("Sin QTO",f"{n_no_qto}","No inventar cantidades")
        with c4: metric("Alertas",str(len(bim_alerts)),"Revisión obligatoria")
        left,right=st.columns([1,1.05])
        with left:
            counts=bim_alerts.regla.value_counts().rename_axis("Regla").reset_index(name="Alertas")
            fig=px.bar(counts,x="Alertas",y="Regla",orientation="h",text="Alertas",color_discrete_sequence=["#e11d48"])
            st.plotly_chart(fig_layout(fig,350),use_container_width=True)
        with right:
            st.markdown("**Partidas y restricciones de uso**")
            st.dataframe(bim_budget[["codigo","concepto","cantidad","unidad","estado_calidad"]],hide_index=True,use_container_width=True)
        st.warning("Sin precios unitarios aprobados; no hay presupuesto monetario. Las partidas con alertas permanecen bloqueadas, aun cuando se simulen cantidades.")

    st.divider()
    st.markdown("#### Reporte ejecutivo descargable")
    st.caption("Un resumen offline de métricas y límites metodológicos, útil como anexo técnico de portafolio o entrevista.")
    reporte_html = f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Construction Intelligence AI · Informe</title>
    <style>body{{font:15px Arial,sans-serif;background:#f4f7fb;color:#192b44;padding:40px;max-width:1000px;margin:auto}}
    h1{{color:#172f6b}}h2{{margin-top:35px}}table{{border-collapse:collapse;width:100%;background:white}}
    th,td{{padding:12px;border-bottom:1px solid #e0e6ef;text-align:left}}th{{background:#e8eefb}}
    .warning{{background:#fff7e6;padding:15px;border-left:4px solid #f59e0b}}
    .hero{{background:#172f6b;color:white;padding:27px;border-radius:12px}}</style></head><body>
    <div class='hero'><h1 style='color:white'>Construction Intelligence AI</h1>AI portfolio · informe de resultados experimentales</div>
    <h2>1. Forecasting de índices industriales</h2>
    {forecast_sum[['material','modelo_elegido','MAE_persistencia','MAE_modelo','mejora_porcentaje','meses_prueba']].round(3).to_html(index=False,escape=True)}
    <h2>2. Clasificación de recibos</h2>{doc_scores.round(4).to_html(index=False,escape=True)}
    <h2>3. Comparación de motores OCR</h2>{ocr_comp[['modo','lineas_referencia','coincidencias','clases_correctas','cobertura_textual_aproximada']].round(3).to_html(index=False,escape=True)}
    <h2>4. Calidad BIM</h2><p>Elementos: {num_elements} · sin QTO: {n_no_qto} · alertas: {len(bim_alerts)} · cantidades QTO: {bim_info.get('registros_cantidades',357)}</p>
    <div class='warning'><strong>Limitaciones:</strong> Índices de EE. UU. no equivalen a costos en MXN; evaluación exploratoria reutilizada; OCR comparado en cinco recibos; clasificación sobre textos transcritos; IFC público sin precios aprobados. Ninguna cifra es precisión global del sistema.</div>
    <p style='color:#64748b'>Proyecto personal de demostración. No está afiliado a ninguna empresa.</p></body></html>"""
    st.download_button("📥 Descargar informe ejecutivo (HTML)", reporte_html.encode("utf-8"), file_name="construction_ai_reporte_ejecutivo.html", mime="text/html", type="primary")

elif page == "Forecasting neuronal":
    st.subheader("Forecasting neuronal de materiales")
    note("Series FRED WPU101 (hierro/acero) y WPU08 (madera), índices de precios industriales en EE. UU. Evaluación a un mes; el pronóstico a seis meses es exploratorio. La prueba histórica ya fue examinada durante el desarrollo.")
    material = st.radio("Material", options=["Acero","Madera"],index=0,horizontal=True)
    material = material or "Acero"
    s = forecast_sum[forecast_sum.material==material].iloc[0]
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Modelo seleccionado",str(s.modelo_elegido),"Elegido por validación temporal")
    with c2: metric("MAE seleccionado",f"{s.MAE_modelo:.3f}","Puntos de índice PPI")
    with c3: metric("MAE persistencia",f"{s.MAE_persistencia:.3f}","Referencia sin IA")
    with c4: metric("Cambio observado",f"{s.mejora_porcentaje:+.1f}%","Menor MAE es mejor")
    tmp=forecast_pred[forecast_pred.material==material]
    fig=go.Figure()
    for col,name,color,dash in [("real","Real",COLORS["real"],None),("persistencia","Persistencia",COLORS["base"],"dash"),("prediccion",str(s.modelo_elegido),COLORS["neural"],None)]:
        fig.add_scatter(x=tmp.fecha,y=tmp[col],name=name,mode="lines",line=dict(color=color,width=2.5 if col=="real" else 2,dash=dash))
    fig.update_yaxes(title="Índice PPI")
    st.plotly_chart(fig_layout(fig,460),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### MAE por año de prueba")
        y=forecast_year[forecast_year.material==material]
        fig=go.Figure()
        for col,name,color in [("MAE_modelo",str(s.modelo_elegido),COLORS["neural"]),("MAE_baseline","Persistencia",COLORS["base"])]:
            fig.add_bar(x=y.anio,y=y[col],name=name,marker_color=color)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig_layout(fig,300),use_container_width=True)
    with c2:
        st.markdown("#### Próximos seis meses (experimental)")
        col=f"{material}_indice"
        fy=forecast_future[["fecha",col]].copy(); fy[col]=fy[col].round(3)
        st.dataframe(fy.rename(columns={"fecha":"Mes",col:"Índice previsto"}),hide_index=True,use_container_width=True)
        if material=="Acero":
            st.caption(f"Intervalo bootstrap exploratorio de mejora: {s.IC95_aprox_inferior:.1f}% a {s.IC95_aprox_superior:.1f}%. Incluye cero; no demuestra mejora universal.")
        else:
            st.caption("Persistencia fue seleccionada: los pronósticos se mantienen constantes, no hay evidencia de mejora de la red en madera.")
    st.download_button("Descargar predicciones de prueba (CSV)",as_csv(tmp),file_name=f"pronosticos_{material.lower()}.csv",mime="text/csv")

elif page == "Document AI":
    st.subheader("Clasificación inteligente de documentos")
    note("Entrenamiento con CORD v1: 800 recibos para entrenamiento, 100 para validación y 100 para prueba. Entrada: <b>texto ya reconocido</b>. F1 y accuracy no miden lectura de fotos ni de facturas mexicanas.")
    c1,c2,c3,c4=st.columns(4)
    for i,(modelo,periodo,col,ttl) in enumerate([("LOGISTICA","prueba","macro_F1","F1 · modelo seleccionado"),("MLP","prueba","macro_F1","F1 · MLP"),("LOGISTICA","prueba","accuracy","Accuracy · logística"),("MLP","prueba","accuracy","Accuracy · MLP")]):
        value=doc_scores[(doc_scores.modelo==modelo)&(doc_scores.periodo==periodo)].iloc[0][col]
        with [c1,c2,c3,c4][i]: metric(ttl, f"{value*100:.1f}%","Métricas sobre 1,336 líneas")
    c1,c2=st.columns([1,1])
    with c1:
        plot = doc_scores.copy();plot["F1 (%)"] = plot.macro_F1*100
        fig=px.bar(plot,x="modelo",y="F1 (%)",color="periodo",barmode="group",text="F1 (%)",color_discrete_map={"validacion":"#94a3b8","prueba":COLORS["neural"]})
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_yaxes(range=[0,100])
        st.plotly_chart(fig_layout(fig,350),use_container_width=True)
    with c2:
        st.image(str(A/"matriz_confusion.png"),caption="Matriz de confusión · regresión logística (prueba)",use_container_width=True)
    st.markdown("#### Errores del clasificador seleccionado")
    pred=csv("data/document_ai/predicciones_documentales.csv")
    filt=st.checkbox("Mostrar solo clasificaciones incorrectas",value=True)
    if filt: pred=pred[pred.categoria != pred.prediccion]
    st.dataframe(pred[["documento","texto","categoria","prediccion"]].head(250),hide_index=True,use_container_width=True)
    st.caption("MLP logró un F1 ligeramente superior en prueba, pero la regresión logística quedó seleccionada usando validación, sin usar prueba para elegir el modelo.")

elif page == "OCR neuronal":
    st.subheader("Comparativa OCR: Tesseract vs RapidOCR")
    note("Experimento diagnóstico de solo cinco recibos CORD ya examinados. Se cuentan líneas emparejadas mediante similitud textual ≥ 65%. Esto <b>no</b> es la tasa oficial de acierto de OCR ni una evaluación independiente.")
    tmp=ocr_comp.copy()
    tmp["Cobertura %"]=tmp.cobertura_textual_aproximada*100
    tmp["Clasificación condicional %"]=tmp.clasificacion_condicional_aproximada*100
    rapid=tmp[tmp.modo=="RapidOCR_neuronal"].iloc[0]
    c1,c2,c3=st.columns(3)
    with c1:metric("RapidOCR · cobertura",f"{rapid['Cobertura %']:.1f}%","40 de 51 líneas emparejadas")
    with c2:metric("Líneas detectadas",str(int(rapid.lineas_ocr)),"RapidOCR detectó 79 fragmentos")
    with c3:metric("Clasificaciones emparejadas",f"{rapid.clases_correctas:.0f}/{rapid.coincidencias:.0f}","No representa exactitud end-to-end")
    fig=go.Figure()
    fig.add_bar(x=tmp.modo,y=tmp["Cobertura %"],marker_color=["#a0aec0","#a0aec0","#a0aec0",COLORS["real"]],text=[f"{x:.1f}%" for x in tmp["Cobertura %"]],textposition="outside")
    fig.update_yaxes(title="Cobertura aproximada (%)",range=[0,100])
    st.plotly_chart(fig_layout(fig,355),use_container_width=True)
    st.dataframe(tmp[["modo","lineas_referencia","lineas_ocr","coincidencias","clases_correctas","Cobertura %","Clasificación condicional %"]].round(2),hide_index=True,use_container_width=True)
    st.markdown("#### Inspeccionar una fotografía pública y la lectura neuronal")
    num=st.selectbox("Recibo de prueba",options=list(range(5)),format_func=lambda n:f"CORD test {n:04d}")
    a,b=st.columns([0.65,1.35])
    with a:st.image(str(A/f"cord-test-{num:04d}.png"),caption="Vista pública CORD (redimensionada)",use_container_width=True)
    with b:
        lines=csv("data/ocr/lineas_ocr_neuronal.csv")
        l=lines[lines.documento==f"cord-test-{num:04d}"].copy()
        st.dataframe(l[["texto_ocr","categoria_predicha","score_ocr_no_calibrado","requiere_revision"]],hide_index=True,use_container_width=True)
        st.caption("Las puntuaciones del OCR y del clasificador no equivalen a probabilidades calibradas de que un importe sea correcto.")

elif page == "Automatización BIM":
    st.subheader("Auditoría de modelos IFC / BIM")
    note("Resultados extraídos de un modelo IFC público, independiente de los recibos CORD. No existen precios aprobados y no hay integración en vivo con Dynamo o Grasshopper.")
    c1,c2,c3,c4=st.columns(4)
    for col,lab,v,sub in [(c1,"Elementos BIM",bim_info.get("elementos",123),"IfcElement + IfcSpace"),(c2,"Sin cantidades",bim_info.get("elementos_sin_qto",28),"QTO no disponible"),(c3,"Alertas detectadas",len(bim_alerts),"Revisión de coherencia"),(c4,"Partidas preliminares",len(bim_budget),"Ningún precio autorizado")]:
        with col: metric(lab,str(v),sub)
    c1,c2=st.columns([1,1])
    with c1:
        st.markdown("#### Catálogo preliminar")
        st.dataframe(bim_budget[["codigo","concepto","cantidad","unidad","estado_calidad"]],hide_index=True,use_container_width=True)
        st.caption("No se suma m², m³ y piezas entre sí. Las cantidades son únicamente del archivo IFC de ejemplo.")
    with c2:
        st.markdown("#### Alertas por regla")
        counts=bim_alerts.regla.value_counts().rename_axis("regla").reset_index(name="alertas")
        fig=px.bar(counts,x="alertas",y="regla",orientation="h",color_discrete_sequence=[COLORS["base"]],text="alertas")
        st.plotly_chart(fig_layout(fig,320),use_container_width=True)
    st.markdown("#### Simulación de sensibilidad (no es diseño geométrico)")
    pct=st.slider("Variación hipotética de cantidades",min_value=-20,max_value=20,value=0,step=5,format="%d%%")
    scenario=bim_budget[["codigo","concepto","unidad","cantidad","estado_calidad"]].copy()
    scenario["Cantidad escenario"] = np.where(scenario.estado_calidad.ne("PRELIMINAR"),np.nan,scenario.cantidad*(1+pct/100))
    st.dataframe(scenario.round(3),hide_index=True,use_container_width=True)
    st.info("Las partidas bloqueadas no se utilizan para cálculos. Los importes quedan vacíos hasta contar con precios verificados y proyecto/alcance coincidentes.")
    with st.expander("Ver alertas completas"):
        st.dataframe(bim_alerts,hide_index=True,use_container_width=True)
    st.download_button("Descargar escenario (CSV)",as_csv(scenario),file_name="escenario_bim.csv")

elif page == "Cotizaciones estructuradas":
    from quotation_ui import render_quotation_ui
    render_quotation_ui()

elif page == "Laboratorio interactivo":
    st.subheader("Laboratorio: probar tus propios archivos")
    note("Estas herramientas ejecutan inferencia y auditoría localmente, sin reentrenar modelos. No sustituyen la revisión humana de importes, facturas o cantidades BIM. Las pruebas no se mezclan con las métricas oficiales de los datasets públicos.")
    tabs=st.tabs(["Clasificar texto", "Leer imagen o PDF", "Inspeccionar IFC"])
    with tabs[0]:
        st.markdown("#### Clasificador documental entrenado")
        user_text=st.text_area("Pega una línea de recibo o cotización por renglón",value="CONCRETO PREMEZCLADO\n2.00\nPRECIO 3000\nSUBTOTAL 6000\nTOTAL 6000",height=145)
        if st.button("Clasificar líneas",type="primary",key="clasificar_texto"):
            try:
                import joblib
                @st.cache_resource(show_spinner=False)
                def get_classifier():return joblib.load(M/"document_ai_logistica.joblib")
                pack=get_classifier();vect=pack["vectorizador"];model=pack["modelo"];cats=pack["categorias"]
                texts=[x.strip() for x in user_text.splitlines() if x.strip()]
                if not texts:st.warning("Escribe al menos una línea.")
                else:
                    X=vect.transform(texts);proba=model.predict_proba(X);idx=proba.argmax(axis=1)
                    res=pd.DataFrame({"texto":texts,"categoria":[cats[int(model.classes_[i])] for i in idx],"puntuación_no_calibrada":proba.max(axis=1)})
                    res["revisar"]=(res["puntuación_no_calibrada"]<.65)|(res["categoria"]=="OTRO")
                    st.dataframe(res.round(3),hide_index=True,use_container_width=True)
                    st.download_button("Descargar resultado CSV",as_csv(res),file_name="clasificacion_documental.csv")
                    if hasattr(model, "coef_"):
                        with st.expander("🔍 IA interpretable: caracteres que influyeron en las predicciones"):
                            st.caption("Explicación estadística del clasificador lineal mediante aportes TF-IDF × coeficiente. No representa una justificación causal ni valida el contenido del documento.")
                            nombres=vect.get_feature_names_out()
                            for ri, clasepos in enumerate(idx[:3]):
                                cidx=int(model.classes_[int(clasepos)])
                                st.markdown(f"**Línea {ri+1}:** `{texts[ri][:80]}` → **{cats[cidx]}**")
                                vector=X[ri].toarray().ravel()
                                coef=model.coef_[int(clasepos)]
                                contrib=vector*coef
                                mejores=np.argsort(contrib)[-8:][::-1]
                                datos_aportes=pd.DataFrame([{"Fragmento":str(nombres[k]),"Aporte":float(contrib[k])} for k in mejores if contrib[k]>0])
                                if not datos_aportes.empty:
                                    fig=px.bar(datos_aportes,x="Aporte",y="Fragmento",orientation="h",color_discrete_sequence=[COLORS["neural"]])
                                    st.plotly_chart(fig_layout(fig,245),use_container_width=True)
                                else:st.caption("No hay fragmentos positivos significativos para esta predicción.")
                    st.caption("Entrenado en recibos CORD en inglés/indonesio; los conceptos de construcción en español están fuera de su dominio de entrenamiento y requieren revisión.")
            except Exception as e:st.error(f"No se pudo cargar el modelo local: {e}")
    with tabs[1]:
        st.markdown("#### OCR neuronal + clasificación de líneas")
        st.caption("RapidOCR puede descargar pesos ONNX en el primer uso. Ningún archivo se guarda persistentemente en este dashboard.")
        up=st.file_uploader("Sube una imagen JPG/PNG o PDF (máximo 20 MB)",type=["jpg","jpeg","png","pdf"],key="ocr_upload")
        if up is not None and st.button("Procesar con RapidOCR",type="primary"):
            try:
                raw=up.getvalue()
                if len(raw)>20*1024*1024:st.error("El archivo supera 20 MB.")
                else:
                    from PIL import Image
                    from rapidocr import RapidOCR
                    import joblib
                    imlist=[]
                    if up.name.lower().endswith(".pdf"):
                        import fitz
                        pdf=fitz.open(stream=raw,filetype="pdf")
                        for k in range(min(len(pdf),3)):
                            pm=pdf[k].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
                            imlist.append((k+1,Image.open(io.BytesIO(pm.tobytes("png"))).convert("RGB")))
                    else:
                        imlist=[(1,Image.open(io.BytesIO(raw)).convert("RGB"))]
                    @st.cache_resource(show_spinner=False)
                    def get_ocr():return RapidOCR()
                    @st.cache_resource(show_spinner=False)
                    def get_clf():return joblib.load(M/"document_ai_logistica.joblib")
                    engine=get_ocr();pkg=get_clf();result=[]
                    with st.spinner("Reconociendo imágenes y clasificando líneas..."):
                        for p,img in imlist:
                            if img.width*img.height>16_000_000:
                                st.warning(f"Página {p} demasiado grande; omitida.")
                                continue
                            out=engine(np.asarray(img));texts=list(out.txts) if out.txts is not None else []
                            scores=list(out.scores) if out.scores is not None else [np.nan]*len(texts)
                            if not texts:continue
                            probs=pkg["modelo"].predict_proba(pkg["vectorizador"].transform(texts))
                            ids=probs.argmax(axis=1)
                            for k,(t,sc) in enumerate(zip(texts,scores)):
                                cat=pkg["categorias"][int(pkg["modelo"].classes_[ids[k]])]
                                clf=float(probs[k,ids[k]])
                                result.append({"pagina":p,"texto":t,"clase":cat,"score_ocr":float(sc),"score_clasificador":clf,"revisar":cat=="OTRO" or clf<.65 or float(sc)<.65})
                    if result:
                        rdf=pd.DataFrame(result);st.success(f"{len(rdf)} líneas reconocidas. Pendientes de revisión: {rdf.revisar.sum()}")
                        st.dataframe(rdf.round(3),hide_index=True,use_container_width=True)
                        st.download_button("Descargar OCR + clasificación",as_csv(rdf),file_name="ocr_documento.csv")
                    else:st.warning("No se detectaron líneas. Intenta otra foto con mayor resolución.")
            except ImportError as e:
                st.error(f"Falta un módulo opcional: {e}. Ejecuta pip install -r requirements.txt")
            except Exception as e: st.error(f"No se pudo procesar el documento: {e}")
    with tabs[2]:
        st.markdown("#### Extracción básica de cantidades IFC")
        st.caption("Permite cargar un IFC sin Revit ni Rhino. No estima cantidades geométricas que no estén declaradas como QTO.")
        upload=st.file_uploader("Selecciona un IFC (máximo 30 MB)",type=["ifc"],key="bim_upload")
        if upload and st.button("Analizar IFC",type="primary"):
            try:
                import ifcopenshell
                import ifcopenshell.util.element
                if len(upload.getvalue())>30*1024*1024:st.error("El archivo excede 30 MB.")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".ifc",delete=False) as tmpfile:
                        tmpfile.write(upload.getvalue());tmpname=tmpfile.name
                    try:mod=ifcopenshell.open(tmpname)
                    finally:Path(tmpname).unlink(missing_ok=True)
                    elems=mod.by_type("IfcElement")
                    rows=[]
                    for e in elems[:15000]:
                        try: qs=ifcopenshell.util.element.get_psets(e,qtos_only=True)
                        except Exception:qs={}
                        for qname,qvals in qs.items():
                            for name,val in qvals.items():
                                if name=="id" or isinstance(val,bool) or not isinstance(val,(int,float)):continue
                                rows.append({"Guid":e.GlobalId,"Clase":e.is_a(),"QTO":qname,"Medida":name,"Valor_original":val})
                    st.success(f"{len(elems):,} elementos; {len(rows):,} cantidades QTO documentadas")
                    if rows:
                        q=pd.DataFrame(rows);st.dataframe(q.head(1000),hide_index=True,use_container_width=True)
                        st.download_button("Descargar QTO IFC",as_csv(q),file_name="cantidades_ifc.csv")
                    else:st.warning("No hay QTO asociados. No se calcularán áreas o volúmenes por suposición.")
            except ImportError:st.error("IfcOpenShell no instalado. Ejecuta pip install ifcopenshell")
            except Exception as e:st.error(f"Error leyendo IFC: {e}")

st.divider()
st.caption("Construction Intelligence AI · Demo independiente sobre datos públicos · Evaluaciones exploratorias no aptas para presupuestos ni compras reales sin validación adicional · GitHub/FRED/CORD/buildingSMART")
