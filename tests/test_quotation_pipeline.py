import json
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quotation_pipeline import extract_pages, extract_quote, validate_quote, build_payload, number

class DemoQuotationTests(unittest.TestCase):
    def load(self, filename):
        path = ROOT / 'demos' / filename
        doc = extract_pages(path.read_bytes(), filename)
        self.assertEqual(doc['motores'], ['TEXTO_PDF'])
        parsed = extract_quote(doc['texto'])
        self.assertEqual(len(parsed['conceptos']), 3)
        self.assertEqual(parsed['campos']['cliente'], 'Constructora Ejemplo SA de CV')
        return parsed, doc

    def test_pdf_correcto(self):
        parsed, doc = self.load('cotizacion_demo_correcta.pdf')
        report = validate_quote(parsed['campos'], parsed['conceptos'])
        self.assertTrue(report['apto_aprobacion'])
        self.assertEqual(report['totales']['subtotal_recalculado'], 9800.00)
        self.assertEqual(report['totales']['total_documento'], 11368.00)
        out = build_payload(parsed['campos'], report, 'cotizacion_demo_correcta.pdf', doc['motores'], True)
        self.assertEqual(out['control']['estado'], 'APROBADO_POR_REVISOR')
        self.assertEqual(out['conceptos'][0]['importe_calculado'], 6000)
        self.assertEqual(len(json.dumps(out, ensure_ascii=False)), len(json.dumps(out, ensure_ascii=False)))

    def test_pdf_error_bloqueado(self):
        parsed, doc = self.load('cotizacion_demo_error.pdf')
        report = validate_quote(parsed['campos'], parsed['conceptos'])
        self.assertFalse(report['apto_aprobacion'])
        self.assertTrue(any('5990.00' in x for x in report['alertas']))
        out = build_payload(parsed['campos'], report, 'cotizacion_demo_error.pdf', doc['motores'], True)
        self.assertEqual(out['control']['estado'], 'REVISION_PENDIENTE')
        parsed['conceptos'][0]['importe_documento'] = 6000.00
        corrected = validate_quote(parsed['campos'], parsed['conceptos'])
        self.assertTrue(corrected['apto_aprobacion'])

    def test_missing_and_ambiguous(self):
        report = validate_quote({'cliente':'', 'folio':'', 'proyecto':'', 'moneda':'', 'subtotal':'', 'iva':'', 'total':''}, [])
        self.assertFalse(report['apto_aprobacion'])
        self.assertGreater(len(report['alertas']), 1)
        self.assertEqual(number('3,000.00'), 3000)
        self.assertEqual(number('3.000,00'), 3000)
        self.assertIsNone(number('???'))

if __name__ == '__main__':
    unittest.main()
