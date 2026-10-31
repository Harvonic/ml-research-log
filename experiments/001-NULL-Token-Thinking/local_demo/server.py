"""Local, read-only comparison of the original baseline and NULL checkpoints."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import webbrowser

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Import the existing architecture/loader without writing caches into src/.
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
with contextlib.redirect_stdout(io.StringIO()):
    from ExperimentUtils import DEVICE, load_model


def parse_problem(value):
    if not isinstance(value, str) or len(value) > 100:
        raise ValueError("Enter an addition problem, such as 12345 + 67890.")
    match = re.fullmatch(r"\s*([0-9]{1,12})\s*\+\s*([0-9]{1,12})\s*=?\s*", value)
    if not match:
        raise ValueError("Use two nonnegative integers with up to 12 digits each, separated by +. Leave out the answer.")
    a, b = map(int, match.groups())
    return f"{a} + {b}", str(a + b), max(len(str(a)), len(str(b)))


class ModelPair:
    def __init__(self):
        torch.set_num_threads(4)
        self.lock = threading.Lock()
        self.models = {}
        for name, count in (("baseline", 0), ("null", 4)):
            path = ROOT / "results" / f"{name}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            model, vocab = load_model(path)
            self.models[name] = (model, vocab, count)
        # Warm both models before serving requests; no weights are changed.
        with torch.inference_mode():
            for model, vocab, count in self.models.values():
                ids = [vocab[c] for c in "1 + 1 = "]
                ids += [vocab["<NULL>"]] * count if count else []
                model(torch.tensor([ids], device=DEVICE))
        self.synchronize()

    @staticmethod
    def synchronize():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    def status(self):
        return {"device": DEVICE,
                "device_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
                "models": {name: {"parameters": sum(p.numel() for p in model.parameters()),
                                  "null_count": count}
                           for name, (model, _, count) in self.models.items()}}

    @torch.inference_mode()
    def predict(self, model, vocab, count, problem, expected):
        ids = [vocab[c] for c in problem + " = "]
        if count:
            ids += [vocab["<NULL>"]] * count
        reverse = {i: token for token, i in vocab.items()}
        forbidden = [vocab[t] for t in ("<PAD>", "<UNK>", "<NULL>") if t in vocab]
        output, terminated = [], False
        self.synchronize()
        started = time.perf_counter()
        for _ in range(24):
            logits = model(torch.tensor([ids], device=DEVICE))[0, -1].clone()
            logits[forbidden] = -float("inf")
            value = int(logits.argmax())
            if value == vocab["<EOS>"]:
                terminated = True
                break
            output.append(reverse[value])
            ids.append(value)
        self.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        answer = "".join(output)
        return {"answer": answer, "correct": terminated and answer == expected,
                "terminated": terminated, "answer_tokens": len(output),
                "milliseconds": round(elapsed, 2), "null_count": count}

    def compare(self, value):
        problem, expected, digits = parse_problem(value)
        with self.lock:
            results = {name: self.predict(model, vocab, count, problem, expected)
                       for name, (model, vocab, count) in self.models.items()}
        return {"problem": problem, "expected": expected, "max_operand_digits": digits,
                "results": results}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, pair):
        self.pair = pair
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            return self.respond(200, self.server.pair.status())
        assets = {"/": ("index.html", "text/html"),
                  "/style.css": ("style.css", "text/css"),
                  "/app.js": ("app.js", "text/javascript")}
        if path not in assets:
            return self.respond(404, {"error": "Not found."})
        name, mime = assets[path]
        self.respond(200, (HERE / name).read_bytes(), mime + "; charset=utf-8")

    def do_POST(self):
        if urlsplit(self.path).path != "/api/compare":
            return self.respond(404, {"error": "Not found."})
        origin = self.headers.get("Origin")
        allowed = {f"http://127.0.0.1:{self.server.server_port}",
                   f"http://localhost:{self.server.server_port}"}
        if origin and origin not in allowed:
            return self.respond(403, {"error": "Open the demo from its localhost URL."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Request must contain one short addition problem.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must contain an addition problem.")
            result = self.server.pair.compare(data.get("problem"))
        except (ValueError, UnicodeDecodeError) as exc:
            return self.respond(400, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            return self.respond(500, {"error": "Model inference failed. Check the server terminal for details."})
        self.respond(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the demo in your default browser")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    print("Loading original baseline and NULL checkpoints...", flush=True)
    pair = ModelPair()
    try:
        server = DemoServer(("127.0.0.1", args.port), pair)
    except OSError as exc:
        raise SystemExit(f"Cannot start on port {args.port}: {exc}. Try --port 8766.") from exc
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"\nReady on {pair.status()['device_name']}\nOpen {url}\nPress Ctrl+C to stop.\n", flush=True)
    if args.open:
        threading.Timer(.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
