import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baby Cry Detector - 1-Click File Downloads</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }
        .card { background: #1e293b; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4); max-width: 580px; width: 100%; padding: 32px; border: 1px solid #334155; }
        h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 8px; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
        p.subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px; }
        .file-list { display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px; }
        .file-item { display: flex; align-items: center; justify-content: space-between; background: #0f172a; padding: 14px 18px; border-radius: 10px; border: 1px solid #334155; }
        .file-info { display: flex; flex-direction: column; }
        .file-name { font-weight: 600; color: #e2e8f0; font-size: 1rem; }
        .file-desc { font-size: 0.8rem; color: #64748b; }
        .btn { background: #0284c7; color: white; border: none; padding: 8px 18px; border-radius: 8px; font-weight: 600; text-decoration: none; cursor: pointer; transition: background 0.2s; font-size: 0.9rem; }
        .btn:hover { background: #0369a1; }
        .btn-zip { background: #10b981; }
        .btn-zip:hover { background: #059669; }
        .btn-all { display: block; text-align: center; background: #38bdf8; color: #0f172a; padding: 12px; border-radius: 10px; font-weight: 700; text-decoration: none; font-size: 1.05rem; }
        .btn-all:hover { background: #7dd3fc; }
        .badge { background: #334155; color: #94a3b8; font-size: 0.75rem; padding: 3px 8px; border-radius: 6px; margin-top: 4px; display: inline-block; }
    </style>
</head>
<body>
    <div class="card">
        <h1>👶 Baby Cry Detector</h1>
        <p class="subtitle">Click any button below to immediately download the file to your device:</p>
        
        <div class="file-list">
            <div class="file-item">
                <div class="file-info">
                    <span class="file-name">preprocess.py</span>
                    <span class="file-desc">Audio preprocessing & feature extraction</span>
                </div>
                <a class="btn" href="/download/preprocess.py" download="preprocess.py">⬇ Download</a>
            </div>

            <div class="file-item">
                <div class="file-info">
                    <span class="file-name">inference.py</span>
                    <span class="file-desc">Live microphone listener & dashboard</span>
                </div>
                <a class="btn" href="/download/inference.py" download="inference.py">⬇ Download</a>
            </div>

            <div class="file-item">
                <div class="file-info">
                    <span class="file-name">baby_cry_model.tflite</span>
                    <span class="file-desc">Trained neural network model</span>
                </div>
                <a class="btn" href="/download/baby_cry_model.tflite" download="baby_cry_model.tflite">⬇ Download</a>
            </div>

            <div class="file-item">
                <div class="file-info">
                    <span class="file-name">sounddevice.py</span>
                    <span class="file-desc">Audio device interface (Zero-install standalone)</span>
                </div>
                <a class="btn" href="/download/sounddevice.py" download="sounddevice.py">⬇ Download</a>
            </div>

            <div class="file-item">
                <div class="file-info">
                    <span class="file-name">scaler_stats.npz</span>
                    <span class="file-desc">Feature normalization statistics</span>
                </div>
                <a class="btn" href="/download/scaler_stats.npz" download="scaler_stats.npz">⬇ Download</a>
            </div>
        </div>

        <a class="btn-all" href="/download/baby_cry_detector_pi.zip" download="baby_cry_detector_pi.zip">
            📦 Download All 4 in One Zip (.zip)
        </a>
    </div>
</body>
</html>
"""

class DownloadHandler(SimpleHTTPRequestHandler):
    def handle_download(self, is_head=False):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML_PAGE.encode("utf-8"))))
            self.end_headers()
            if not is_head:
                self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        filename = os.path.basename(self.path.split("?")[0])
        filepath = os.path.join(DIRECTORY, filename)

        if os.path.isfile(filepath):
            try:
                filesize = os.path.getsize(filepath)
                self.send_response(200)
                # Force instant browser download via Content-Disposition attachment header
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(filesize))
                self.end_headers()
                if not is_head:
                    with open(filepath, "rb") as f:
                        while chunk := f.read(65536):
                            self.wfile.write(chunk)
                return
            except Exception as e:
                self.send_error(500, f"Error reading file: {e}")
                return

        self.send_error(404, "File not found")

    def do_GET(self):
        self.handle_download(is_head=False)

    def do_HEAD(self):
        self.handle_download(is_head=True)

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), DownloadHandler)
    print(f"Serving download portal at http://0.0.0.0:{PORT}")
    server.serve_forever()
