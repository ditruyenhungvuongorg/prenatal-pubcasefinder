import unittest
from unittest.mock import Mock
from web_service import WebSystem
from model1_v38_package.core import align

class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = WebSystem()
        cls.system.initialize()

    def test_search_id_alias_and_vietnamese(self):
        s = self.system
        self.assertEqual(s.search_hpo('HP:0009729')[0]['id'], 'HP:0009729')
        self.assertEqual(s.search_hpo('HP:0001366')[0]['id'], 'HP:0000252')
        self.assertTrue(s.search_hpo('dau nho'))
        self.assertEqual(len({t['id'] for t in s.search_hpo('tim')}), len(s.search_hpo('tim')))

    def test_input_validation(self):
        for payload in ({'hpos': []}, {'hpos': [{'id': 'HP:9999999', 'status': 'CÓ'}]},
                        {'hpos': [{'id': 'HP:0000252', 'status': 'KHÔNG'}]},
                        {'hpos': [{'id': 'HP:0000252', 'status': 'MAYBE'}]}):
            with self.assertRaises(ValueError):
                self.system.match(payload)

    def test_real_ranking_and_inheritance(self):
        result = self.system.match({'hpos': [{'id': 'HP:0009729', 'status': 'CÓ'}]})
        self.assertEqual(len(result['candidates']), 20)
        self.assertTrue(any(c['inheritance_modes'] for c in result['candidates']))
        for c in result['candidates']:
            self.assertTrue(0 <= c['match_percentage'] <= 100)
            self.assertTrue(all(p['relation'] not in ('no_match', 'ignored') for p in c['matched_phenotypes']))
            self.assertNotIn('HP:0009729', {p['id'] for p in c['clinical_features_to_check']})

    def test_extraction_requires_review_and_preserves_spans(self):
        s = self.system
        original = s.runner
        text = 'Không ghi nhận đầu nhỏ.'
        start = text.index('đầu nhỏ')
        s.runner = Mock()
        s.runner.extract_spans.return_value = [{'mention_text': 'đầu nhỏ', 'span_start': start, 'span_end': start + 7}]
        try:
            output = s.extract(text)
            self.assertTrue(output['review_required'])
            self.assertTrue(output['mentions'][0]['context_review'])
            self.assertFalse(output['mentions'][0]['approved'])
            self.assertTrue(output['mentions'][0]['candidates'])
        finally:
            s.runner = original

    def test_strict_alignment(self):
        with self.assertRaises(ValueError):
            align('đầu nhỏ, đầu nhỏ', '{"phrases":["đầu nhỏ"]}')
        with self.assertRaises(ValueError):
            align('tim bình thường', '{"phrases":["đầu nhỏ"]}')

if __name__ == '__main__':
    unittest.main(verbosity=2)
