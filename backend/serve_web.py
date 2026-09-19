"""HTTP web entrypoint; exposes static UI and reviewed APIs only."""
import hmac
import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import urlparse, parse_qs
from web_service import WebSystem

ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / 'static' if (ROOT / 'static').is_dir() else ROOT.parent / 'docs'
SYSTEM = WebSystem()
WORK = BoundedSemaphore(1)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(180)

    def log_message(self, *args):
        pass

    def allowed_origin(self):
        origin = self.headers.get('Origin')
        host = self.headers.get('Host', '')
        return not origin or origin in os.getenv('WEB_ALLOWED_ORIGINS', '').split(',') or origin in ('http://' + host, 'https://' + host)

    def end_headers(self):
        origin = self.headers.get('Origin')
        if origin and self.allowed_origin():
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        super().end_headers()

    def send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if not self.allowed_origin():
            self.send_json({'error': 'Nguồn truy cập chưa được cấu hình.'}, 403)
            return False
        token = os.getenv('WEB_ACCESS_TOKEN', '')
        if token and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
            self.send_json({'error': 'Nhập mã truy cập do quản trị viên cấp.'}, 401)
            return False
        return True

    def do_OPTIONS(self):
        if not self.allowed_origin():
            return self.send_json({'error': 'Origin denied'}, 403)
        self.send_response(204)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path)
        if path.path == '/api/status':
            return self.send_json({'status': 'online', 'profiles_count': len(SYSTEM.matcher.profiles),
                'hpo_terms_count': len(SYSTEM.canonical), 'model_ready': SYSTEM.runner.is_loaded,
                'model_configured': bool(SYSTEM.runner.adapter_path), 'auth_required': bool(os.getenv('WEB_ACCESS_TOKEN'))})
        if path.path.startswith('/api/'):
            if not self.authorized():
                return
            if path.path == '/api/hpo_search':
                return self.send_json(SYSTEM.search_hpo(parse_qs(path.query).get('q', [''])[0][:200]))
            return self.send_json({'error': 'Không có endpoint này.'}, 404)
        if path.path not in ('/', '/index.html', '/styles.css', '/app.js', '/config.js'):
            return self.send_error(404)
        super().do_GET()

    def do_POST(self):
        if not self.authorized():
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 100000:
                return self.send_json({'error': 'Yêu cầu quá lớn hoặc rỗng.'}, 413)
            payload = json.loads(self.rfile.read(size).decode())
            if not isinstance(payload, dict):
                raise ValueError()
        except (ValueError, UnicodeError):
            return self.send_json({'error': 'JSON không hợp lệ.'}, 400)
        path = urlparse(self.path).path
        if path not in ('/api/extract_hpo', '/api/match_diseases'):
            return self.send_json({'error': 'Không có endpoint này.'}, 404)
        if not WORK.acquire(blocking=False):
            return self.send_json({'error': 'Hệ thống đang xử lý ca khác. Vui lòng thử lại.'}, 503)
        try:
            data = SYSTEM.extract(payload.get('text')) if path.endswith('extract_hpo') else SYSTEM.match(payload)
            self.send_json(data)
        except ValueError as exc:
            self.send_json({'error': str(exc)}, 422)
        except RuntimeError:
            self.send_json({'error': 'Model chưa sẵn sàng hoặc đang bận. Liên hệ quản trị viên.'}, 503)
        except Exception as exc:
            print('Request failed:', type(exc).__name__, file=sys.stderr, flush=True)
            self.send_json({'error': 'Xử lý thất bại. Chưa tạo kết quả; vui lòng thử lại.'}, 500)
        finally:
            WORK.release()

if __name__ == '__main__':
    SYSTEM.initialize()
    if os.getenv('MODEL1_PRELOAD') == '1':
        SYSTEM.runner.load_model()
    port = int(os.getenv('PORT', '8000'))
    print(f'Web ready on 127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()

