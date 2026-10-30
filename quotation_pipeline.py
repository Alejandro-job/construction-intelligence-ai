"""Local quotation extraction and deterministic validations for a portfolio demo.

This is a prototype for structured import, not a certified invoice parser.
Text-based PDFs use embedded text, while scans use optional RapidOCR.
No values are inferred when absent; a reviewer must approve to export as approved.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from PIL import Image

PENNY = Decimal('0.01')
MAX_BYTES = 20 * 1024 * 1024


def number(value: Any) -> Decimal | None:
    """Parse common MX and US numeric notation without fabricating missing values."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            v = Decimal(str(value))
            return v if v.is_finite() else None
        except InvalidOperation:
            return None
    raw = str(value).strip()
    if not raw or raw.lower() in {'nan', 'none', 'null', 'n/a', '-', '—'}:
        return None
    raw = re.sub(r'(?i)\b(?:MXN|USD|PESOS|MX\$)\b', '', raw)
    raw = re.sub(r'[$\s]', '', raw)
    if not re.fullmatch(r'-?[\d.,]+', raw):
        return None
    if ',' in raw and '.' in raw:
        decimal_sep = '.' if raw.rfind('.') > raw.rfind(',') else ','
        thousands_sep = ',' if decimal_sep == '.' else '.'
        raw = raw.replace(thousands_sep, '').replace(decimal_sep, '.')
    elif ',' in raw:
        chunks = raw.lstrip('-').split(',')
        if len(chunks) > 1 and all(len(k) == 3 for k in chunks[1:]):
            raw = raw.replace(',', '')
        else:
            raw = raw.replace(',', '.')
    elif '.' in raw:
        chunks = raw.lstrip('-').split('.')
        if len(chunks) > 1 and all(len(k) == 3 for k in chunks[1:]):
            raw = raw.replace('.', '')
    try:
        val = Decimal(raw)
        return val if val.is_finite() else None
    except InvalidOperation:
        return None


def money(value: Any) -> Decimal | None:
    val = number(value)
    return val.quantize(PENNY, rounding=ROUND_HALF_UP) if val is not None else None


def as_float(v: Decimal | None) -> float | None:
    return float(v) if v is not None else None


def extract_pages(raw: bytes, filename: str) -> dict:
    """Extract PDF embedded text or run OCR only where embedded text is absent."""
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('Archivo vacío o superior a 20 MB.')
    name = filename.lower().strip()
    pages = []
    modes = []
    if name.endswith('.pdf'):
        import fitz
        pdf = fitz.open(stream=raw, filetype='pdf')
        if pdf.is_encrypted:
            raise ValueError('PDF protegido. Debes desbloquearlo antes de importarlo.')
        if len(pdf) > 5:
            raise ValueError('Por seguridad, esta demostración admite hasta 5 páginas.')
        for page in pdf:
            text = page.get_text('text', sort=True).strip()
            if len(text) >= 45:
                pages.append(text)
                modes.append('TEXTO_PDF')
            else:
                from rapidocr import RapidOCR
                import numpy as np
                p = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                im = Image.open(io.BytesIO(p.tobytes('png'))).convert('RGB')
                if im.width * im.height > 16_000_000:
                    raise ValueError('Página demasiado grande para OCR.')
                output = RapidOCR()(np.asarray(im))
                texts = list(output.txts) if output.txts is not None else []
                pages.append('\n'.join(map(str, texts)))
                modes.append('RAPIDOCR')
        pdf.close()
    elif name.endswith(('.jpg', '.jpeg', '.png')):
        import numpy as np
        from rapidocr import RapidOCR
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        if img.width * img.height > 16_000_000:
            raise ValueError('La fotografía tiene demasiados píxeles para procesarla.')
        output = RapidOCR()(np.asarray(img))
        pages = ['\n'.join(map(str, output.txts)) if output.txts is not None else '']
        modes = ['RAPIDOCR']
    else:
        raise ValueError('Formato no soportado; utiliza PDF, JPG o PNG.')
    return {'texto': '\n'.join(pages), 'motores': modes, 'paginas': len(pages)}


# Designed around an explicit, inspectable quotation schema. Other supplier
# formats may require mapping/review; do not silently claim general extraction.
FIELD_PATTERNS = {
    'folio': r'^(?:FOLIO|NO\.?\s*DE\s*COTIZACI[OÓ]N|COTIZACI[OÓ]N\s*(?:NO\.?|#))\s*[:#\-]\s*(.+)$',
    'fecha': r'^FECHA\s*[:\-]\s*(.+)$',
    'cliente': r'^CLIENTE\s*[:\-]\s*(.+)$',
    'rfc': r'^(?:RFC|RFC\s+CLIENTE)\s*[:\-]\s*(.+)$',
    'correo': r'^(?:CORREO|EMAIL|E-MAIL)\s*[:\-]\s*(.+)$',
    'proyecto': r'^(?:PROYECTO|OBRA)\s*[:\-]\s*(.+)$',
    'moneda': r'^MONEDA\s*[:\-]\s*(MXN|USD|EUR)\b',
    'subtotal': r'^SUBTOTAL\s*[:\-]\s*(.+)$',
    'iva': r'^IVA(?:\s*\(?\s*\d+(?:[.,]\d+)?\s*%\s*\)?)?\s*[:\-]\s*(.+)$',
    'total': r'^TOTAL(?:\s+A\s+PAGAR)?\s*[:\-]\s*(.+)$',
}


def extract_quote(text: str) -> dict:
    fields = {key: '' for key in FIELD_PATTERNS}
    items = []
    other_lines = []
    for original in text.splitlines():
        line = original.strip()
        if not line:
            continue
        found = False
        for key, pattern in FIELD_PATTERNS.items():
            m = re.match(pattern, line, flags=re.I)
            if m:
                fields[key] = m.group(1).strip()
                found = True
                break
        if found:
            continue
        # Explicit delimited item table: CODIGO | DESCRIPCION | CANTIDAD |
        # UNIDAD | PRECIO UNITARIO | IMPORTE. The user can correct/add rows.
        if '|' in line:
            cols = [k.strip() for k in line.split('|')]
            if len(cols) == 6 and number(cols[2]) is not None and number(cols[4]) is not None:
                items.append({
                    'codigo': cols[0], 'descripcion': cols[1],
                    'cantidad': as_float(number(cols[2])), 'unidad': cols[3],
                    'precio_unitario': as_float(money(cols[4])),
                    'importe_documento': as_float(money(cols[5])),
                })
                continue
        other_lines.append(line)
    return {'campos': fields, 'conceptos': items, 'lineas_no_estructuradas': other_lines}


def _valid_date(text: str, fmt: str) -> bool:
    try:
        datetime.strptime(text, fmt)
        return True
    except ValueError:
        return False


def validate_quote(fields: dict, concepts: list[dict]) -> dict:
    """Validate edited values, preserving document amounts (never silently overwrite)."""
    alerts = []
    missing = []
    for key, title in [('folio', 'folio'), ('fecha', 'fecha'), ('cliente', 'cliente'), ('proyecto', 'proyecto'), ('moneda', 'moneda')]:
        if not str(fields.get(key, '') or '').strip():
            missing.append(title)
    if missing:
        alerts.append('Faltan campos obligatorios: ' + ', '.join(missing))
    date = str(fields.get('fecha', '') or '').strip()
    if date and not any(_valid_date(date, fmt) for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y')):
        alerts.append('Fecha no reconocida; utiliza AAAA-MM-DD o DD/MM/AAAA.')
    email = str(fields.get('correo', '') or '').strip()
    if email and not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        alerts.append('Formato de correo no reconocido; requiere revisión.')
    rfc = str(fields.get('rfc', '') or '').upper().strip()
    if rfc and not re.fullmatch(r'[A-ZÑ&]{3,4}[0-9]{6}[A-Z0-9]{3}', rfc):
        alerts.append('Formato de RFC no reconocido (solo revisión de patrón, no validez fiscal).')
    currency = str(fields.get('moneda', '') or '').strip().upper()
    if currency and currency not in {'MXN', 'USD', 'EUR'}:
        alerts.append('La moneda no está entre las opciones admitidas (MXN, USD, EUR).')
    if not concepts:
        alerts.append('No hay conceptos; agrega las partidas manualmente.')

    checked = []
    sum_calculated = Decimal('0.00')
    sum_document = Decimal('0.00')
    document_sum_complete = True
    for i, row in enumerate(concepts, start=1):
        desc = str(row.get('descripcion', '') or '').strip()
        unit = str(row.get('unidad', '') or '').strip()
        qty = number(row.get('cantidad'))
        unit_price = money(row.get('precio_unitario'))
        listed = money(row.get('importe_documento'))
        if not desc or not unit or qty is None or unit_price is None or qty <= 0 or unit_price < 0:
            alerts.append(f'Concepto {i}: descripción, unidad, cantidad positiva y precio válido requeridos.')
        calc = money(qty * unit_price) if qty is not None and unit_price is not None else None
        if calc is not None:
            sum_calculated += calc
        else:
            document_sum_complete = False
        if listed is None:
            alerts.append(f'Concepto {i}: falta el importe declarado; debe revisarse.')
            document_sum_complete = False
        else:
            sum_document += listed
            if calc is not None and abs(calc - listed) > PENNY:
                alerts.append(
                    f'Concepto {i} ({desc or "sin descripción"}): cantidad × precio = {calc}, '
                    f'pero el PDF declara {listed}.'
                )
        checked.append({
            'codigo': str(row.get('codigo', '') or ''),
            'descripcion': desc, 'cantidad': as_float(qty), 'unidad': unit,
            'precio_unitario': as_float(unit_price),
            'importe_documento': as_float(listed),
            'importe_calculado': as_float(calc),
            'coincide_importe': calc is not None and listed is not None and abs(calc - listed) <= PENNY,
        })

    subtotal = money(fields.get('subtotal'))
    iva = money(fields.get('iva'))
    total = money(fields.get('total'))
    if subtotal is None:
        alerts.append('Falta subtotal declarado en la cotización.')
    if total is None:
        alerts.append('Falta total declarado en la cotización.')
    if iva is None:
        alerts.append('IVA no identificado; indicar explícitamente 0 si no aplica.')
    if not concepts:
        sum_calculated_out = None
    else:
        sum_calculated_out = as_float(sum_calculated)
    if subtotal is not None and concepts and abs(sum_calculated - subtotal) > PENNY:
        alerts.append(f'Subtotal: suma recalculada {sum_calculated} ≠ subtotal del documento {subtotal}.')
    if subtotal is not None and concepts and document_sum_complete and abs(sum_document - subtotal) > PENNY:
        alerts.append(f'Subtotal: suma de importes transcritos {sum_document} ≠ subtotal del documento {subtotal}.')
    if subtotal is not None and iva is not None and total is not None:
        expected_total = subtotal + iva
        if abs(expected_total - total) > PENNY:
            alerts.append(f'Total: subtotal + IVA = {expected_total} ≠ total del documento {total}.')
    if subtotal is not None and iva is not None and iva < 0:
        alerts.append('IVA negativo; revisar.')
    if total is not None and total < 0:
        alerts.append('Total negativo; revisar.')
    return {
        'alertas': alerts,
        'apto_aprobacion': not alerts,
        'conceptos': checked,
        'totales': {
            'subtotal_documento': as_float(subtotal),
            'subtotal_recalculado': sum_calculated_out,
            'iva_documento': as_float(iva),
            'total_documento': as_float(total),
            'total_esperado_segun_documento': as_float(subtotal + iva) if subtotal is not None and iva is not None else None,
        },
    }


def build_payload(fields: dict, checked: dict, filename: str, modes: list[str], confirmed: bool) -> dict:
    state = 'APROBADO_POR_REVISOR' if confirmed and checked['apto_aprobacion'] else 'REVISION_PENDIENTE'
    return {
        'schema_version': '1.0',
        'tipo': 'cotizacion',
        'origen': {'archivo': filename, 'metodos_extraccion': modes},
        'folio': str(fields.get('folio', '') or '').strip(),
        'fecha': str(fields.get('fecha', '') or '').strip(),
        'cliente': {
            'nombre': str(fields.get('cliente', '') or '').strip(),
            'rfc': str(fields.get('rfc', '') or '').strip(),
            'correo': str(fields.get('correo', '') or '').strip(),
        },
        'proyecto': str(fields.get('proyecto', '') or '').strip(),
        'moneda': str(fields.get('moneda', '') or '').strip().upper(),
        'conceptos': checked['conceptos'],
        'totales': checked['totales'],
        'control': {
            'estado': state,
            'validaciones_aprobadas': checked['apto_aprobacion'],
            'revision_manual_confirmada': bool(confirmed and checked['apto_aprobacion']),
            'alertas': checked['alertas'],
            'nota': 'Demo. No se envía a un ERP; JSON y CSV quedan listos para integración previa autorización.',
        },
    }
