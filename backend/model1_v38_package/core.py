"""Extraction-only contract; no GPU dependencies and no HPO mapping."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'unsloth/Qwen3.5-4B'
REVISION = '3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636'
SEED = 3407
SYSTEM = '''Trích các cụm biểu hiện được nhắc đến trong văn bản bác sĩ. Chỉ trả JSON {"phrases":["cụm nguyên văn",...]}, không giải thích.
Sao chép đúng chữ hoa, dấu và khoảng trắng của từng cụm. Giữ cả phần bổ nghĩa giải phẫu, bên, mức độ trong cùng cụm; hai biểu hiện độc lập là hai cụm.
Không lấy chủ thể, từ phủ định/nghi ngờ, lời dẫn hay dấu câu kết thúc vào cụm. Vẫn trích biểu hiện khi được phủ định, nghi ngờ, nhắc trong tiền sử hoặc chưa đánh giá; đây chỉ là trích văn bản, không xác nhận bệnh.
Liệt kê theo thứ tự xuất hiện; nếu cụm được nhắc hai lần thì liệt kê hai lần. Không suy diễn, không đổi tên, không sinh mã HPO hoặc vị trí ký tự. Không có biểu hiện thì trả {"phrases":[]}.'''


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_key(text):
    text = ''.join(c for c in unicodedata.normalize('NFD', text.casefold()) if not unicodedata.combining(c)).replace('đ', 'd')
    return re.sub(r'^tat ', '', ' '.join(re.findall(r'[a-z0-9]+', text)))


def split_for(key):
    n = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return 'train' if n < 80 else 'validation' if n < 90 else 'test'


def occurrences(text, phrase):
    def word(c):
        return c.isalnum() or c == '_' or unicodedata.category(c).startswith('M')
    # Lookahead includes overlapping occurrences; ambiguity must not be hidden by re.finditer.
    return [(m.start(1), m.end(1)) for m in re.finditer('(?=('+re.escape(phrase)+'))', text)
            if (not word(phrase[0]) or m.start(1) == 0 or not word(text[m.start(1)-1]))
            and (not word(phrase[-1]) or m.end(1) == len(text) or not word(text[m.end(1)]))]


def strict_json(raw):
    def pairs(items):
        obj = {}
        for k,v in items:
            if k in obj:
                raise ValueError('Duplicate JSON key: '+k)
            obj[k] = v
        return obj
    return json.loads(raw, object_pairs_hook=pairs)


def align(text, raw):
    """Accept one strict JSON. Unique monotonic exact alignment only, never fix wording."""
    obj = strict_json(raw)
    if not isinstance(obj, dict) or set(obj) != {'phrases'}:
        raise ValueError('Expected exactly one phrases field')
    phrases = obj['phrases']
    if not isinstance(phrases, list) or len(phrases) > 40:
        raise ValueError('phrases must be a list with at most 40 entries')
    if any(not isinstance(p, str) or not p or p != p.strip() for p in phrases):
        raise ValueError('Every phrase must be a nonempty exact string')
    choices = [occurrences(text, p) for p in phrases]
    if any(not c for c in choices):
        raise ValueError('Phrase absent from source or partial word')
    # Cap search at two solutions: ambiguous repeated occurrences must not be guessed.
    from functools import lru_cache
    @lru_cache(None)
    def walk(i, previous):
        if i == len(phrases):
            return ((),)
        solutions = []
        for start, end in choices[i]:
            if start < previous:
                continue
            for tail in walk(i + 1, end):
                solutions.append(((start, end),) + tail)
                if len(solutions) == 2:
                    return tuple(solutions)
        return tuple(solutions)
    solutions = walk(0, 0)
    if len(solutions) != 1:
        raise ValueError('Ambiguous repeated occurrences' if solutions else 'Overlapping or out-of-order phrases')
    return [{'mention_text': p, 'span_start': a, 'span_end': b}
            for p, (a, b) in zip(phrases, solutions[0])]


def answer(row):
    return json.dumps({'phrases': [m['mention_text'] for m in row['mentions']]}, ensure_ascii=False, separators=(',', ':'))


def validate_row(row):
    text = row['text']
    if not text or text != text.strip():
        raise ValueError('Empty text or external whitespace')
    expected = [{k: m[k] for k in ('mention_text', 'span_start', 'span_end')} for m in row['mentions']]
    if align(text, answer(row)) != expected:
        raise ValueError('Gold offsets or phrase order disagree: ' + row['sample_id'])


def score(records, expected_count=None):
    tp = fp = fn = exact = invalid = fragmented = 0
    for r in records:
        gold = {(m['span_start'], m['span_end']) for m in r['gold']}
        if r.get('error'):
            invalid += 1
            fn += len(gold)
            continue
        pred = {(m['span_start'], m['span_end']) for m in r['predicted']}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += gold == pred
        fragmented += sum(1 for a, b in gold if (a, b) not in pred and sum(a <= x < y <= b for x, y in pred) >= 2)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    n = len(records)
    return dict(samples=n, expected_samples=expected_count if expected_count is not None else n,
                complete=expected_count is None or n == expected_count,
                exact_span_rate=exact/n if n else 0, invalid_outputs=invalid,
                span_tp=tp, span_fp=fp, span_fn=fn, span_precision=p, span_recall=r,
                span_f1=2*p*r/(p+r) if p+r else 0, fragmented_gold_spans=fragmented,
                synthetic_evaluation=True, clinical_validation=False)
