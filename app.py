#!/usr/bin/env python3
"""Local GUI server. Run: python app.py, then open http://127.0.0.1:8765."""
from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from models import CACHE, model_catalog

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATIC = ROOT / "static"
LOCK = threading.RLock()
JOBS = {}
ENGINE = None
ACTIVE = None
HARDWARE_PREVIEW = None
HARDWARE_LOCK = threading.Lock()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def records(folder):
    return [read_json(p) for p in sorted((DATA / folder).glob("*.json"), reverse=True)]


def run_job(job_id, request):
    global ENGINE, ACTIVE
    def progress(message, fraction):
        with LOCK:
            JOBS[job_id].update(message=message, progress=fraction, hardware=ENGINE.hardware.describe())
    try:
        from engine import Engine
        if ENGINE is None:
            ENGINE = Engine()
        result = ENGINE.run(request, progress, lambda: JOBS[job_id].get("cancelled", False))
        result["id"] = job_id
        with LOCK:
            if JOBS[job_id].get("cancelled"):
                raise InterruptedError("Stopped. This experiment was not saved.")
            write_json(DATA / "runs" / f"{job_id}.json", result)
            JOBS[job_id].update(status="done", progress=1, message="Complete. Results were saved automatically to History.", result=result)
    except InterruptedError as exc:
        with LOCK:
            JOBS[job_id].update(status="cancelled", message=str(exc))
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            JOBS[job_id].update(status="error", message=str(exc))
    finally:
        with LOCK:
            ACTIVE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if self.path.startswith("/api/jobs/"):
            return
        super().log_message(fmt, *args)

    def send(self, data, status=200, content_type="application/json; charset=utf-8", download=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)

    def export(self, format="jsonl", scope="examples", ids=None):
        from server.storage import Library
        data, status, content_type, download = Library(DATA).export(format, scope, ids)
        return self.send(data, status, content_type, download)

    def do_GET(self):
        global HARDWARE_PREVIEW
        route = urlparse(self.path)
        path = route.path
        try:
            if path == "/api/config":
                # Keep the UI immediately available, even during the first torch import.
                return self.send({"models": model_catalog(), "active_job": ACTIVE,
                                  "data_path": str(DATA), "model_cache_path": str(CACHE)})
            if path == "/api/hardware":
                if ENGINE is not None:
                    return self.send(ENGINE.hardware.describe())
                # Detection imports torch only in this request, keeping static pages fast.
                from hardware import Hardware
                with HARDWARE_LOCK:
                    if HARDWARE_PREVIEW is None:
                        HARDWARE_PREVIEW = Hardware().describe()
                    return self.send(HARDWARE_PREVIEW)
            if path == "/api/library":
                with LOCK:
                    all_records = records("examples")
                    history = records("runs")
                key = lambda r: r.get("created_at", "")
                return self.send({"examples": sorted(all_records, key=key, reverse=True),
                                  "history": sorted(history, key=key, reverse=True)})
            if path.startswith("/api/jobs/"):
                with LOCK:
                    job = JOBS.get(path.split("/")[-1])
                    return self.send(job if job else {"error": "Job not found."}, 200 if job else 404)
            if path == "/api/export":
                query = parse_qs(route.query)
                return self.export(query.get("format", ["jsonl"])[0],
                                   query.get("scope", ["examples"])[0])
            assets = ROOT / "web" if path in ("/token-matrix.js", "/token-matrix.css") else STATIC
            file = (assets / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if file.parent != assets or not file.is_file():
                return self.send({"error": "Not found"}, 404)
            return self.send(file.read_bytes(), content_type=(mimetypes.guess_type(file.name)[0] or "application/octet-stream") + "; charset=utf-8")
        except (OSError, ValueError) as exc:
            self.send({"error": str(exc)}, 500)

    def do_POST(self):
        global ACTIVE
        # Accept mutations only from this local UI.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send({"error": "Use the local Plateau Lab page for this action."}, 403)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size > 100_000:
                raise ValueError("The request is too large.")
            data = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Invalid request format.")
            path = urlparse(self.path).path
            if path == "/api/export":
                return self.export(data.get("format", "jsonl"), data.get("scope", "examples"),
                                   data.get("ids") if data.get("ids") is not None else [])
            if path == "/api/run":
                with LOCK:
                    if ACTIVE:
                        return self.send({"error": "Another experiment is running. Wait for it to finish or stop it first."}, 409)
                    job_id = uuid.uuid4().hex
                    JOBS[job_id] = {"id": job_id, "status": "running", "progress": 0,
                                    "message": "Preparing the model…", "request": data}
                    ACTIVE = job_id
                threading.Thread(target=run_job, args=(job_id, data), daemon=True).start()
                return self.send({"id": job_id}, 202)
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                with LOCK:
                    job_id = path.split("/")[-2]
                    if job_id not in JOBS:
                        return self.send({"error": "Job not found."}, 404)
                    if JOBS[job_id]["status"] == "running":
                        JOBS[job_id].update(cancelled=True, message="Stopping after the current computation or download step…")
                    return self.send({"ok": True})
            if path == "/api/examples":
                job_id = str(data.get("id", ""))
                if not re_full_id(job_id):
                    raise ValueError("Invalid example ID.")
                with LOCK:
                    result = read_json(DATA / "runs" / f"{job_id}.json")
                    result["tag"] = str(data.get("tag", "Unclassified"))[:80]
                    result["notes"] = str(data.get("notes", ""))[:4000]
                    result["saved_at"] = datetime.now(timezone.utc).isoformat()
                    write_json(DATA / "examples" / f"{job_id}.json", result)
                return self.send({"ok": True, "id": job_id})
            return self.send({"error": "Not found"}, 404)
        except (ValueError, OSError) as exc:
            self.send({"error": str(exc)}, 400)


def re_full_id(value):
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA,
                        help="Collection directory (use a temporary directory for application checks).")
    args = parser.parse_args()
    DATA = args.data_dir.resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"MARS V · Plateau Lab → {url}\nExamples are stored in {DATA}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
