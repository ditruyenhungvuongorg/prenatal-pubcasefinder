"""Unit tests for Stage 2 Clinical Assertion Engine."""
import unittest
from assertion_engine import ClinicalAssertionEngine
from pipeline import PrenatalPhenotypePipeline


class ClinicalAssertionTests(unittest.TestCase):
    def setUp(self):
        self.engine = ClinicalAssertionEngine()

    def test_inner_modifier_one_part(self):
        text = "Thai: bất sản 1 phần thuỳ nhộng"
        phrase = "bất sản 1 phần thuỳ nhộng"
        start = text.index(phrase)
        end = start + len(phrase)
        status, reason = self.engine.predict(text, start, end, phrase)
        self.assertEqual(status, "NGHI NGỜ")
        self.assertIn("1 phần", reason)

    def test_inner_modifier_mot_phan(self):
        text = "Thai: thiểu sản một phần thể chai"
        phrase = "thiểu sản một phần thể chai"
        start = text.index(phrase)
        end = start + len(phrase)
        status, reason = self.engine.predict(text, start, end, phrase)
        self.assertEqual(status, "NGHI NGỜ")
        self.assertIn("một phần", reason)

    def test_inner_phrase_with_theo_doi(self):
        text = "Ghi nhận van ba lá đóng thấp theo dõi Ebstein"
        phrase = "van ba lá đóng thấp theo dõi Ebstein"
        start = text.index(phrase)
        end = start + len(phrase)
        status, reason = self.engine.predict(text, start, end, phrase)
        self.assertEqual(status, "NGHI NGỜ")
        self.assertIn("theo dõi", reason)

    def test_left_cue_td(self):
        text = "Thai 22 tuần, TD kênh nhĩ thất toàn phần"
        phrase = "kênh nhĩ thất toàn phần"
        start = text.index(phrase)
        end = start + len(phrase)
        status, reason = self.engine.predict(text, start, end, phrase)
        self.assertEqual(status, "NGHI NGỜ")
        self.assertIn("TD", reason)

    def test_left_cue_theo_doi(self):
        text = "Thai 32 tuần, theo dõi u cơ tim"
        phrase = "u cơ tim"
        start = text.index(phrase)
        end = start + len(phrase)
        status, reason = self.engine.predict(text, start, end, phrase)
        self.assertEqual(status, "NGHI NGỜ")
        self.assertIn("theo dõi", reason)

    def test_delimiter_dash_stops_td_scope(self):
        text = "Thai: TD đục thuỷ tinh thể - Hở van 3 lá"
        p1 = "đục thuỷ tinh thể"
        p2 = "Hở van 3 lá"
        s1, e1 = text.index(p1), text.index(p1) + len(p1)
        s2, e2 = text.index(p2), text.index(p2) + len(p2)

        st1, _ = self.engine.predict(text, s1, e1, p1)
        st2, _ = self.engine.predict(text, s2, e2, p2)
        self.assertEqual(st1, "NGHI NGỜ")
        self.assertEqual(st2, "CÓ")

    def test_delimiter_comma_stops_td_scope(self):
        text = "Thai: theo dõi u cơ tim, ít dịch màng ngoài tim, ối ít"
        p1 = "u cơ tim"
        p2 = "ít dịch màng ngoài tim"
        p3 = "ối ít"
        s1, e1 = text.index(p1), text.index(p1) + len(p1)
        s2, e2 = text.index(p2), text.index(p2) + len(p2)
        s3, e3 = text.index(p3), text.index(p3) + len(p3)

        st1, _ = self.engine.predict(text, s1, e1, p1)
        st2, _ = self.engine.predict(text, s2, e2, p2)
        st3, _ = self.engine.predict(text, s3, e3, p3)

        self.assertEqual(st1, "NGHI NGỜ")
        self.assertEqual(st2, "CÓ")
        self.assertEqual(st3, "CÓ")

    def test_delimiter_period_stops_td_scope(self):
        text = "Thai theo dõi hẹp thực quản. Đa ối, con to"
        p1 = "hẹp thực quản"
        p2 = "Đa ối"
        s1, e1 = text.index(p1), text.index(p1) + len(p1)
        s2, e2 = text.index(p2), text.index(p2) + len(p2)

        st1, _ = self.engine.predict(text, s1, e1, p1)
        st2, _ = self.engine.predict(text, s2, e2, p2)
        self.assertEqual(st1, "NGHI NGỜ")
        self.assertEqual(st2, "CÓ")

    def test_definitive_present_findings(self):
        text = "Thai 24 tuần: Tứ chứng Fallot, Cung ĐMC bên P, thoát vị rốn"
        for phrase in ["Tứ chứng Fallot", "Cung ĐMC bên P", "thoát vị rốn"]:
            s, e = text.index(phrase), text.index(phrase) + len(phrase)
            status, _ = self.engine.predict(text, s, e, phrase)
            self.assertEqual(status, "CÓ")

    def test_cue_nghi_ngo(self):
        text = "Thai 31 tuần, bất cân xứng tim phải trái, nghi ngờ hẹp eo ĐMC"
        phrase = "hẹp eo ĐMC"
        s, e = text.index(phrase), text.index(phrase) + len(phrase)
        status, reason = self.engine.predict(text, s, e, phrase)
        self.assertEqual(status, "NGHI NGỜ")
        self.assertIn("nghi ngờ", reason)

    def test_pipeline_integration(self):
        pipeline = PrenatalPhenotypePipeline()
        text = "Thai: TD kênh nhĩ thất toàn phần, nang thận bên P"
        spans = [
            {'mention_text': 'kênh nhĩ thất toàn phần', 'span_start': text.index('kênh nhĩ thất toàn phần'), 'span_end': text.index('kênh nhĩ thất toàn phần') + 23},
            {'mention_text': 'nang thận bên P', 'span_start': text.index('nang thận bên P'), 'span_end': text.index('nang thận bên P') + 15}
        ]
        res = pipeline.process_spans(text, spans)
        self.assertEqual(res['total_phenotypes'], 2)
        self.assertEqual(res['phenotypes'][0]['status'], "NGHI NGỜ")
        self.assertEqual(res['phenotypes'][1]['status'], "CÓ")


if __name__ == '__main__':
    unittest.main()
