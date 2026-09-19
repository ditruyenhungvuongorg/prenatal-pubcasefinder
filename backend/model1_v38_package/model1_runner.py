"""Serve the same pinned worker used by the v3.8 evaluation."""
import importlib.util
import sys
from pathlib import Path
from threading import Lock
from .core import align

class Model1V38Runner:
    def __init__(self, adapter_path=None, worker_dir=None):
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.worker_dir = Path(worker_dir) if worker_dir else None
        self.is_loaded = False
        self.lock = Lock()

    def load_model(self):
        if self.is_loaded:
            return
        if not self.adapter_path or not all((self.adapter_path / n).is_file() for n in ('adapter_config.json', 'adapter_model.safetensors')):
            raise RuntimeError('Adapter v3.8 chưa được cấu hình.')
        if not self.worker_dir or not (self.worker_dir / 'worker.py').is_file():
            raise RuntimeError('Chưa cấu hình worker inference.')
        sys.path.insert(0, str(self.worker_dir))
        spec = importlib.util.spec_from_file_location('prenatal_v38_worker', self.worker_dir / 'worker.py')
        self.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.worker)
        self.model, self.tokenizer, self.eos = self.worker.load(self.adapter_path, training=False)
        self.worker.generate_one(self.model, self.tokenizer, self.eos, 'Phiếu đã được lưu.', seconds=60)
        self.is_loaded = True

    def extract_spans(self, text):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Model đang bận.')
        try:
            self.load_model()
            result = self.worker.generate_one(self.model, self.tokenizer, self.eos, text)
            return align(text, result['raw'])
        finally:
            self.lock.release()

