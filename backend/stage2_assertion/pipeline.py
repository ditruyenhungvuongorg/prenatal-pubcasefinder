"""Pipeline Tích hợp Toàn diện (End-to-End): Giai đoạn 1 (Model 1 Span Extraction) + Giai đoạn 2 (Clinical Assertion).

Pipeline nhận văn bản siêu âm lâm sàng thô, gọi Model 1 để trích xuất danh sách
cụm từ nguyên văn kèm vị trí ký tự [span_start, span_end), sau đó chuyển qua
Clinical Assertion Engine để gán trạng thái lâm sàng ("CÓ" vs "NGHI NGỜ").
"""
import json
import unicodedata
from assertion_engine import ClinicalAssertionEngine


class PrenatalPhenotypePipeline:
    def __init__(self, model1_runner=None, include_soft_markers=True):
        """Khởi tạo pipeline 2 giai đoạn.

        Parameters
        ----------
        model1_runner : callable, optional
            Hàm nhận vào text và trả về list dict:
            [{'mention_text': ..., 'span_start': ..., 'span_end': ...}, ...]
        """
        self.model1_runner = model1_runner
        self.assertion_engine = ClinicalAssertionEngine()

    def process_spans(self, text, mentions):
        """Giai đoạn 2: Tiếp nhận danh sách spans đã trích xuất từ Model 1 và gán trạng thái lâm sàng.

        Parameters
        ----------
        text : str
            Văn bản siêu âm gốc.
        mentions : list of dict
            Danh sách thực thể do Model 1 trích xuất, mỗi phần tử gồm:
            {'mention_text': str, 'span_start': int, 'span_end': int}

        Returns
        -------
        dict
            Cấu trúc JSON đầy đủ gồm văn bản gốc, số thực thể, và danh sách
            thực thể kèm vị trí, trạng thái lâm sàng ("CÓ" / "NGHI NGỜ") và giải thích.
        """
        phenotypes = []
        for m in mentions:
            start = m['span_start']
            end = m['span_end']
            phrase = m.get('mention_text', text[start:end])

            status, reason = self.assertion_engine.predict(
                text=text,
                span_start=start,
                span_end=end,
                mention_text=phrase
            )

            phenotypes.append({
                'phrase': phrase,
                'span': [start, end],
                'status': status,
                'explanation': reason
            })

        return {
            'text': text,
            'total_phenotypes': len(phenotypes),
            'phenotypes': phenotypes
        }

    def predict(self, text):
        """Chạy toàn diện 2 giai đoạn từ văn bản thô (yêu cầu model1_runner đã được cấu hình)."""
        if self.model1_runner is None:
            raise RuntimeError(
                "Chưa cấu hình model1_runner! "
                "Vui lòng truyền model1_runner hoặc sử dụng process_spans(text, mentions)."
            )
        mentions = self.model1_runner(text)
        return self.process_spans(text, mentions)


def demo_sample():
    """Hàm chạy thử nghiệm pipeline trên một câu lâm sàng thực tế."""
    text = (
        "Thai 19 tuần 2 ngày, Thai: bất thường tim (kênh nhĩ thất toàn phần) + "
        "bất thường hệ TK (bất sản thể chai hoàn toàn + bất sản 1 phần thuỳ nhộng, "
        "lểu tiểu não bị đẩy lên cao) + thoát vị rốn/ TD hẹp eo ĐMC"
    )
    # Giả lập kết quả trích xuất từ Model 1 (v3.7)
    mock_model1_spans = [
        {'mention_text': 'kênh nhĩ thất toàn phần', 'span_start': 43, 'span_end': 66},
        {'mention_text': 'bất sản thể chai hoàn toàn', 'span_start': 88, 'span_end': 114},
        {'mention_text': 'bất sản 1 phần thuỳ nhộng', 'span_start': 117, 'span_end': 142},
        {'mention_text': 'thoát vị rốn', 'span_start': 175, 'span_end': 187},
        {'mention_text': 'hẹp eo ĐMC', 'span_start': 191, 'span_end': 201},
    ]

    pipeline = PrenatalPhenotypePipeline()
    result = pipeline.process_spans(text, mock_model1_spans)

    print("KẾT QUẢ XỬ LÝ PIPELINE 2 GIAI ĐOẠN:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    demo_sample()
