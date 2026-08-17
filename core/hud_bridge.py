"""
VORTEX -> HUD Bridge Client v2
Single background thread + queue. Zero blocking on caller.
If HUD is down, events are silently dropped after 100 queued.
"""
import threading
import queue
import time

try:
    import requests
    _session = requests.Session()
except ImportError:
    _session = None

HUD_URL = "http://127.0.0.1:8765/event"
_q = queue.Queue(maxsize=100)
_started = False
_lock = threading.Lock()

def _worker():
    while True:
        try:
            data = _q.get(timeout=1.0)
        except queue.Empty:
            continue
        if data is None:
            break
        if _session is None:
            continue
        try:
            _session.post(HUD_URL, json=data, timeout=0.3)
        except Exception:
            pass
        finally:
            _q.task_done()

def _ensure_started():
    global _started
    if not _started:
        with _lock:
            if not _started:
                t = threading.Thread(target=_worker, daemon=True, name="HudBridge")
                t.start()
                _started = True

def _send(data):
    _ensure_started()
    try:
        _q.put_nowait(data)
    except queue.Full:
        pass  # drop event silently

def send_state(state):
    _send({"type": "state", "state": state})

def send_transcript(text):
    _send({"type": "transcript", "text": text})

def send_response(text):
    _send({"type": "response", "text": text})

def send_audio_level(value):
    _send({"type": "audio_level", "value": float(value)})

def send_graph_update(nodes, edges):
    _send({"type": "graph_update", "nodes": nodes, "edges": edges})

def is_available():
    if _session is None:
        return False
    try:
        r = _session.get("http://127.0.0.1:8765/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False
