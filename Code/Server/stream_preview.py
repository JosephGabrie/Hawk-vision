from http import server
import os
from socketserver import ThreadingMixIn
import sys
import cv2
import numpy as np
from camera import Camera

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

camera = Camera(stream_size=(640, 480))
camera.start_stream()

# Load YOLO pedestrian action models if available
cls_model = None
det_model = None
tracker = None
try:
    from detect_pedestrian_action import PedestrianVideoTracker, load_pedestrian_models

    cls_model, det_model, cls_path, det_path = load_pedestrian_models()
    tracker = PedestrianVideoTracker(history_len=10)
    print(f"Loaded YOLO standing/walking models for live stream preview ({cls_path}, {det_path}).")
except Exception as e:
    print(f"YOLO overlay disabled in stream_preview ({e})")

PAGE = """\
<html>
<head><title>Raspberry Pi Camera Preview (YOLO Pedestrian Action)</title></head>
<body style="display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#111;">
  <img src="stream.mjpg" style="border: 2px solid #555; max-width: 90vw;" />
</body>
</html>
"""


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE.encode("utf-8"))
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            try:
                while True:
                    frame = camera.get_frame()
                    if tracker is not None and det_model is not None and cls_model is not None:
                        bgr = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if bgr is not None:
                            annotated = tracker.update_and_classify(
                                bgr, det_model, cls_model, conf=0.4, is_video_stream=True
                            )
                            ok, enc = cv2.imencode(".jpg", annotated)
                            if ok:
                                frame = enc.tobytes()
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass
        else:
            self.send_error(404)


class StreamingServer(ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


try:
    address = ("", 8000)
    httpd = StreamingServer(address, StreamingHandler)
    print("Serving preview at http://<pi-ip-address>:8000")
    httpd.serve_forever()
finally:
    camera.stop_stream()
    camera.close()
