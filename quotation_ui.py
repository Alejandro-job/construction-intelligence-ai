"""Streamlit tab: PDF/image quotation -> editable structured JSON/CSV with validation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from quotation_pipeline import extract_pages, extract_quote, validate_quote, build_payload

HERE = Path(__file__).resolve().parent
DEMO = HERE / 'demos'
FIELD_NAMES = [
    ('folio', 'Folio'), ('fecha', 'Fecha del documento'), ('cliente', 'Cliente / razón social'),
    ('rfc', 'RFC del cliente'), ('correo', 'Correo del cliente'), ('proyecto', 'Proyecto / obra'),
    ('moneda', 'Moneda'), ('subtotal', 'Subtotal declarado'), ('iva', 'IVA declarado'), ('total', 'Total declarado'),
]
EDITABLE = ['codigo', 'descripcion', 'cantidad', 'unidad', 'precio_unitario', 'importe_documento']


def render_quotation_ui():
    st.subheader('Cotizaciones estructuradas')
    st.caption('Demostración de ingestión y control: PDF o fotografía → texto/OCR → campos → validación → revisión → JSON/CSV.')
    st.info(
        'La extracción de PDF con texto es determinista; las imágenes y PDFs escaneados pasan por RapidOCR neuronal. '
        'Los conceptos se interpretan con un formato de tabla explícito; otros diseños de proveedor pueden requerir '
        'corregir datos o completar partidas. Nunca se inventan precios faltantes.'
    )

    st.markdown('#### 1. Prueba con dos cotizaciones de ejemplo')
    a, b = st.columns(2)
    with a:
        st.download_button('⬇️ Cotización correcta (PDF)',
            data=(DEMO / 'cotizacion_demo_correcta.pdf').read_bytes(),
            file_name='cotizacion_demo_correcta.pdf', mime='application/pdf', key='q_demo_ok')
        st.caption('3 partidas · Subtotal 9,800 MXN · IVA 1,568 · Total 11,368. Valores ficticios.')
    with b:
        st.download_button('⬇️ Cotización con error (PDF)',
            data=(DEMO / 'cotizacion_demo_error.pdf').read_bytes(),
            file_name='cotizacion_demo_error.pdf', mime='application/pdf', key='q_demo_bad')
        st.caption('La primera partida declara 5,990 en lugar de 6,000. Debe generar alertas.')

    uploaded = st.file_uploader('Sube una cotización PDF, JPG o PNG (máximo 20 MB)',
        type=['pdf', 'jpg', 'jpeg', 'png'], key='q_upload')
    if uploaded is None:
        st.markdown('**Para demostrarlo:** descarga el PDF con error, súbelo aquí y pulsa **Extraer datos**. '
                    'Después corrige las partidas y enseña cómo se desbloquea la exportación validada.')
        return
    data = uploaded.getvalue()
    identity = hashlib.sha256(uploaded.name.encode() + data).hexdigest()
    if len(data) > 20 * 1024 * 1024:
        st.error('El archivo supera el máximo de 20 MB.')
        return

    if st.button('🔎 Extraer datos y detectar conceptos', type='primary', key='q_parse'):
        with st.spinner('Leyendo PDF, identificando campos y comprobando importes...'):
            try:
                extracted = extract_pages(data, uploaded.name)
                parsed = extract_quote(extracted['texto'])
                initial_check = validate_quote(parsed['campos'], parsed['conceptos'])
                st.session_state['q_identity'] = identity
                st.session_state['q_extracted'] = extracted
                st.session_state['q_original'] = parsed
                st.session_state['q_initial_alerts'] = initial_check['alertas']
                st.session_state['q_approval_sig'] = None
                st.session_state['q_ack_source'] = False
                st.session_state['q_reviewer_note'] = ''
                st.session_state['q_confirm_content'] = False
                for key, _ in FIELD_NAMES:
                    st.session_state[f'q_field_{key}'] = parsed['campos'].get(key, '')
                st.session_state['q_editor_items'] = pd.DataFrame(parsed['conceptos'], columns=EDITABLE)
                st.success(f"Extracción terminada · {extracted['paginas']} página(s) · "
                           f"{len(parsed['conceptos'])} partida(s) detectada(s).")
            except Exception as exc:
                st.error(f'No se pudo extraer el documento: {exc}')
                return

    if st.session_state.get('q_identity') != identity:
        st.warning('Pulsa **Extraer datos** para cargar este nuevo archivo.')
        return

    parsed = st.session_state['q_original']
    extracted = st.session_state['q_extracted']
    raw_alerts = st.session_state.get('q_initial_alerts', [])
    motors = ', '.join(dict.fromkeys(extracted['motores']))
    st.caption(f'Archivo: **{uploaded.name}** · Extracción: **{motors}** · {extracted["paginas"]} página(s)')

    st.markdown('#### 2. Verificar la información del cliente y los importes')
    st.caption('Puedes corregir cualquier campo. Los valores se revisarán de nuevo después de cada edición.')
    cols = st.columns(3)
    for index, (key, label) in enumerate(FIELD_NAMES):
        with cols[index % 3]:
            st.text_input(label, key=f'q_field_{key}')
    fields = {key: st.session_state.get(f'q_field_{key}', '') for key, _ in FIELD_NAMES}

    st.markdown('#### 3. Conceptos extraídos — tabla editable')
    st.caption('Añade filas manualmente si el formato del proveedor no se puede identificar. '
               'Cantidad × precio se recalcula independientemente del importe del documento.')
    original_frame = st.session_state.get('q_editor_items', pd.DataFrame(columns=EDITABLE))
    edit = st.data_editor(
        original_frame,
        hide_index=True,
        num_rows='dynamic',
        use_container_width=True,
        key=f'q_data_editor_{identity[:12]}',
        column_config={
            'codigo': st.column_config.TextColumn('Código'),
            'descripcion': st.column_config.TextColumn('Descripción', width='large'),
            'cantidad': st.column_config.NumberColumn('Cantidad', format='%.2f'),
            'unidad': st.column_config.TextColumn('Unidad'),
            'precio_unitario': st.column_config.NumberColumn('Precio unitario', format='%.2f'),
            'importe_documento': st.column_config.NumberColumn('Importe declarado', format='%.2f'),
        },
    )
    rows = edit[EDITABLE].to_dict(orient='records')
    checked = validate_quote(fields, rows)
    counts = st.columns(3)
    with counts[0]:
        st.metric('Partidas registradas', len(rows))
    with counts[1]:
        sub = checked['totales']['subtotal_recalculado']
        st.metric('Subtotal recalculado', f'{sub:,.2f} {fields.get("moneda", "")}' if sub is not None else 'N/D')
    with counts[2]:
        st.metric('Alertas actuales', len(checked['alertas']))

    if checked['alertas']:
        st.error('**Documento pendiente de revisión:**')
        for a in checked['alertas']:
            st.markdown('- ' + a)
    else:
        st.success('Las verificaciones **aritméticas y de campos obligatorios** pasaron. '
                   'Esto no garantiza autenticidad del cliente, validez fiscal ni precios de mercado.')

    if raw_alerts:
        with st.expander(f'Alertas del documento original ({len(raw_alerts)}) — historial conservado', expanded=True):
            for a in raw_alerts:
                st.write('• ' + a)
            st.warning('Si se corrigió el dato, se guardará como modificación del revisor. '
                       'No cambia el PDF original: una discrepancia debe confirmarse con su emisor.')

    with st.expander('Auditoría técnica: texto extraído del PDF/OCR'):
        st.code(extracted['texto'][:18000], language=None)
        if parsed['lineas_no_estructuradas']:
            st.caption(f"{len(parsed['lineas_no_estructuradas'])} línea(s) fuera de la tabla estructurada. "
                       'Revisa que no falten conceptos antes de aprobar.')

    st.markdown('#### 4. Revisión y autorización para exportar')
    note = st.text_area('Nota de revisión (qué comprobaste y con quién)',
        key='q_reviewer_note', placeholder='Ejemplo: Importes contrastados con el proveedor...')
    reviewed = st.checkbox('Confirmo que cotejé el cliente, todos los conceptos y cantidades con la cotización.',
        key='q_confirm_content')
    acknowledged = True
    if raw_alerts:
        acknowledged = st.checkbox(
            'Confirmo que cualquier discrepancia del PDF original fue aclarada con el emisor; '
            'el registro conservará el cambio y las alertas originales.',
            key='q_ack_source'
        )
    if raw_alerts and not note.strip():
        st.caption('Con discrepancias de origen, agrega una nota de revisión antes de aprobar.')

    # Any change to values invalidates a previous manual approval.
    fingerprint_source = json.dumps({'fields': fields, 'items': rows, 'ack': acknowledged,
                                      'reviewed': reviewed, 'note': note},
                                     ensure_ascii=False, sort_keys=True, default=str)
    approval_fingerprint = hashlib.sha256(fingerprint_source.encode('utf-8')).hexdigest()
    eligible = bool(checked['apto_aprobacion'] and reviewed and acknowledged and (not raw_alerts or note.strip()))
    if st.button('✅ Aprobar estructura revisada', disabled=not eligible, key='q_approve'):
        st.session_state['q_approval_sig'] = approval_fingerprint
    approved = eligible and st.session_state.get('q_approval_sig') == approval_fingerprint
    if approved:
        st.success('Estructura **aprobada por el revisor**. Lista para exportación; no se envía automáticamente a ningún ERP.')
    else:
        st.info('Estado actual: **REVISIÓN PENDIENTE**. Puedes exportar el borrador, pero no enviarlo como aprobado.')

    # Audit of changed fields and item values, without tampering with original PDF.
    changes = []
    for key, _ in FIELD_NAMES:
        old = str(parsed['campos'].get(key, '') or '')
        new = str(fields.get(key, '') or '')
        if old != new:
            changes.append({'campo': key, 'original': old, 'revisado': new})
    original_items = parsed.get('conceptos', [])
    for index, row in enumerate(rows):
        oldrow = original_items[index] if index < len(original_items) else {}
        for key in EDITABLE:
            if str(oldrow.get(key, '')) != str(row.get(key, '')):
                changes.append({'campo': f'concepto_{index+1}.{key}',
                                'original': oldrow.get(key), 'revisado': row.get(key)})

    payload = build_payload(fields, checked, uploaded.name, extracted['motores'], approved)
    payload['control']['alertas_de_origen'] = raw_alerts
    payload['control']['correcciones_registradas'] = changes
    payload['control']['nota_revisor'] = note.strip()
    payload['control']['formato_reconocido'] = 'TABLA_DELIMITADA' if parsed['conceptos'] else 'REVISION_MANUAL'
    payload['control']['advertencia'] = (
        'Prototipo independiente. Estructura generada tras extracción heurística y revisión, '
        'no implica envío, integración ni validación fiscal.'
    )

    with st.expander('Ver JSON estructurado antes de descargar', expanded=True):
        st.json(payload, expanded=False)
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    export_df = pd.DataFrame(checked['conceptos'])
    export_df.insert(0, 'folio', payload['folio'])
    export_df.insert(1, 'cliente', payload['cliente']['nombre'])
    export_df.insert(2, 'proyecto', payload['proyecto'])
    export_df.insert(3, 'moneda', payload['moneda'])
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            '⬇️ Exportar JSON ' + ('APROBADO' if approved else 'BORRADOR'),
            data=json_bytes, mime='application/json',
            file_name=f"cotizacion_{'aprobada' if approved else 'borrador'}.json", key='q_json')
    with c2:
        st.download_button(
            '⬇️ Exportar conceptos CSV ' + ('APROBADO' if approved else 'BORRADOR'),
            data=export_df.to_csv(index=False).encode('utf-8-sig'), mime='text/csv',
            file_name=f"conceptos_{'aprobados' if approved else 'borrador'}.csv", key='q_csv')
    st.caption('La salida queda disponible para una integración futura mediante API o ETL. '
               'Aquí **no se transmite** ningún dato a un CRM, ERP, base de datos o correo.')
