"""Clinical Assertion Engine (Giai đoạn 2: Phân loại Trạng thái Lâm sàng).

Quy tắc chuẩn hóa nghiêm ngặt theo Bác sĩ:
Chỉ sử dụng 3 từ khóa/khái niệm cốt lõi để xác định trạng thái:
1. "1 phần" (và biến thể chữ "một phần")
2. "theo dõi" (và biến thể viết tắt "t/d", "td")
3. "nghi ngờ" (và biến thể "nghi")

- Nếu phát hiện 1 trong 3 từ này (bên trong cụm từ hoặc trong ngữ cảnh vế câu trước cụm từ) -> "NGHI NGỜ"
- Nếu không có -> "CÓ"
"""
import re
import unicodedata

# 3 từ khóa cốt lõi theo quy chuẩn bác sĩ
SUSPECTED_PATTERN = re.compile(
    r'\b(?:1\s*phần|một\s*phần|1\s*phan|mot\s*phan|'
    r'theo\s*dõi|theo\s*doi|t/d|td|'
    r'nghi\s*ngờ|nghi\s*ngo|nghi)\b',
    re.IGNORECASE
)

# Dấu phân cách tách biệt giữa các biểu hiện lâm sàng độc lập
DELIMITERS = ['-', '–', '—', ';', '\n', '/', '.', ',', '+']


class ClinicalAssertionEngine:
    def __init__(self):
        pass

    def predict(self, text, span_start, span_end, mention_text=None):
        """Phân loại trạng thái của một cụm từ dị tật trong văn bản.

        Parameters
        ----------
        text : str
            Toàn bộ câu/văn bản mô tả siêu âm lâm sàng.
        span_start : int
            Vị trí ký tự bắt đầu của cụm từ trong text.
        span_end : int
            Vị trí ký tự kết thúc của cụm từ trong text.
        mention_text : str, optional
            Nội dung cụm từ (nếu không truyền sẽ cắt từ text[span_start:span_end]).

        Returns
        -------
        tuple[str, str]
            (status, reason):
            - status: "CÓ" hoặc "NGHI NGỜ"
            - reason: giải thích căn cứ trên 3 từ khóa của bác sĩ.
        """
        if mention_text is None:
            mention_text = text[span_start:span_end]

        mention_normalized = unicodedata.normalize('NFC', mention_text.strip())

        # 1. Kiểm tra nếu có 1 trong 3 từ khóa nằm ngay bên trong tên cụm từ
        inner_match = SUSPECTED_PATTERN.search(mention_normalized)
        if inner_match:
            cue = inner_match.group(0)
            return "NGHI NGỜ", f"chứa từ khóa '{cue}' bên trong cụm từ"

        # 2. Phân tích ngữ cảnh dẫn trước trong phạm vi vế câu (sau dấu phân cách gần nhất)
        left_sub = text[:span_start]
        last_delim = max([left_sub.rfind(d) for d in DELIMITERS] + [-1])
        scope_text = left_sub[last_delim + 1:].strip()

        left_match = SUSPECTED_PATTERN.search(scope_text)
        if left_match:
            cue = left_match.group(0)
            return "NGHI NGỜ", f"chứa từ khóa '{cue}' trong phạm vi vế câu"

        # 3. Không có từ nào trong 3 từ -> Khẳng định hiện diện
        return "CÓ", "khẳng định hiện diện (không có 1 phần, theo dõi, nghi ngờ)"
