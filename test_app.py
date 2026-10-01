"""End-to-end: run frontend.py headless (Streamlit AppTest) against synthetic data + a mock Ollama."""
import json, os, re, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from make_synthetic_data import make

LOG = []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen3-vl:4b-instruct"}, {"name": "llama3:8b"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LOG.append(req)
        user = req["messages"][-1]
        self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
        if req.get("format") == "json":
            ids = sorted({int(i) for i in re.findall(r"^#(\d+):", user["content"], re.M)})
            payload = {"regions": [{"id": i, "real_change": i % 2 == 1, "category": "new building / structure",
                                    "reason": "mock verdict"} for i in ids]}
            self.wfile.write((json.dumps({"message": {"content": json.dumps(payload)}, "done": True}) + "\n").encode())
        else:
            for piece in ("Region #1 looks like ", "a new building. ", "Region #2 is uncertain."):
                self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())

srv = HTTPServer(("127.0.0.1", 11555), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
data = Path(tempfile.mkdtemp()); make(data)
os.environ["RS_DATA_DIR"] = str(data)

from streamlit.testing.v1 import AppTest
at = AppTest.from_file(str(ROOT / "frontend.py"), default_timeout=180)
at.run()
assert not at.exception, at.exception
at.sidebar.text_input[1].set_value("http://127.0.0.1:11555")   # Ollama URL
at.run(); assert not at.exception, at.exception
print("models in selectbox:", at.sidebar.selectbox[2].options)

btn = [b for b in at.button if "Detect changes" in b.label][0]; btn.click(); at.run()
assert not at.exception, at.exception
print("errors:", [e.value for e in at.error])
print("metrics:", [(m.label, m.value) for m in at.metric])

[b for b in at.button if "Audit regions" in b.label][0].click(); at.run()
assert not at.exception, at.exception
print("audit errors:", [e.value for e in at.error])
aud = LOG[-1]
print("audit payload: format=", aud.get("format"), "| system msg:", aud["messages"][0]["role"], "| images:", len(aud["messages"][-1]["images"]))

at.chat_input[0].set_value("Which changes look like new buildings?").run()
assert not at.exception, at.exception
print("chat errors:", [e.value for e in at.error])
print("assistant:", [m.markdown[0].value for m in at.chat_message if m.name == "assistant"])
q = LOG[-1]["messages"][-1]["content"]
print("question payload contains table + question:", "EVIDENCE TABLE" in q and "QUESTION: Which changes" in q, "| images:", len(LOG[-1]["messages"][-1]["images"]))
print("rows in region table after VLM audit:", len(at.dataframe[-1].value) if at.dataframe else None)
print("OK")
