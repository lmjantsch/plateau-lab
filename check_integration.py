"""Offline cross-backend measurement and FastAPI compatibility checks."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from plateau.core.experiment import run_experiment
from plateau.core.math import arc_segments, effect_metrics
from plateau.core.results import arc_from_segments
from server.storage import Library
import server.app as server


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, **kwargs):
        return [1 if text == "A" else 2, 3]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class Backend:
    """Known vector paths; exercise real experiment orchestration and reduction."""
    def __init__(self, points, oom=False):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.natural_calls = self.calls = 0
        self.oom = oom

    def load(self, *args):
        return SimpleNamespace(tokenizer=Tokenizer(), context_limit=256, blocks=[None],
                               config=SimpleNamespace(model_type="synthetic"))

    def natural(self, *args):
        side = self.natural_calls % 2
        self.natural_calls += 1
        return {"source": torch.tensor([[float(side), 1.0]]),
                "layer_states": self.points[[0 if side == 0 else -1]],
                "logits": self.points[0 if side == 0 else -1], "dtype": "float32",
                "tokens": [1, 2, 3], "probabilities": [.5, .5, .5]}

    def path(self, lm, ids, layer, start, sources, ts, method, record_layers,
             include_reference_logits=False, previous=None, cancelled=lambda: False):
        self.calls += 1
        if self.oom and self.calls == 2:
            raise RuntimeError("out of memory after a partial path")
        indices = (ts * (len(self.points) - 1)).round().long()
        samples = self.points[indices]
        values, gap = effect_metrics(samples, self.points[0], self.points[-1])
        keys = [str(layer) for layer in record_layers] + ["logits"]
        arcs = {key: arc_segments(samples, (previous or {}).get(key)) for key in keys}
        return {"values": {key: {name: value.tolist() for name, value in values.items()} for key in keys},
                "gaps": {key: float(gap) for key in keys}, "endpoint_l2": [float(gap)],
                "step_lengths": {key: arc[0].tolist() for key, arc in arcs.items()},
                "last_vectors": {key: arc[1] for key, arc in arcs.items()},
                "reference_tokens": [0, 1], "tokens": [0] * len(ts),
                "candidate_ids": [[[0, 1, 2]] * len(ts)] * 3,
                "candidate_probs": [[[.5, .3, .1]] * len(ts)] * 3,
                "first_error": 0.0, "last_error": 0.0, "reference_logits": self.points[[0, -1]]}

    def describe(self):
        return {"remote": False, "device": "synthetic"}


REQUEST = dict(model="synthetic", sequence_a="A", sequence_b="B", steps=5,
               interpolation="linear", patch_layer=0)
MODELS = {"synthetic": {"repo": "no-network", "label": "Synthetic"}}
POINTS = [[0, 0], [.5, 1], [1, -1], [1.5, 1], [2, 0]]


def run(backend, batch=2, progress=lambda *_: None, cancelled=lambda: False):
    return run_experiment(backend, REQUEST, batch, progress, cancelled, MODELS)


class Measurements(unittest.TestCase):
    def test_actual_segments_at_every_batch_split(self):
        expected = [0.] + [math.dist(a, b) for a, b in zip(POINTS, POINTS[1:])]
        total = math.fsum(expected)
        for batch in (1, 2, 3, 5):
            result = run(Backend(POINTS), batch)
            self.assertEqual(result["schema_version"], 7)
            for row in result["effect"]["rows"]:
                for actual, target in zip(row["step_lengths"], expected):
                    self.assertAlmostEqual(actual, target, places=12)
                self.assertAlmostEqual(row["total_length"], total, places=12)
                self.assertEqual(row["values"]["c"][-1], 1)
            json.dumps(result, allow_nan=False)

    def test_loop_stationary_and_nonfinite(self):
        cases = [([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], "ok"),
                 ([[0, 0]] * 5, "stationary"),
                 ([[0, 0], [1, 0], [float("nan"), 1], [1, 0], [0, 0]], "nonfinite")]
        for points, status in cases:
            result = run(Backend(points))
            row = result["effect"]["rows"][-1]
            self.assertEqual(row["c_status"], status)
            self.assertFalse(row["defined_metrics"]["relative_l2_shinkle"])
            self.assertEqual(row["defined_metrics"]["c"], status == "ok")
            json.dumps(result, allow_nan=False)

    def test_tiny_arc_and_float64_before_subtraction(self):
        points = torch.tensor([[1e20, 0], [-1e20, 1e-20], [1e20, 0]], dtype=torch.float32)
        lengths, _ = arc_segments(points)
        self.assertTrue(torch.isfinite(lengths).all())
        tiny = arc_from_segments([0, 1e-30, 2e-30])
        self.assertEqual(tiny["c_status"], "ok")
        self.assertEqual(tiny["c"][-1], 1)

    def test_oom_discards_partial_path(self):
        result = run(Backend(POINTS, oom=True), 2)
        clean = run(Backend(POINTS), 1)
        self.assertEqual(result["settings"]["batch_retries"], 1)
        self.assertEqual(result["curves"], clean["curves"])
        self.assertEqual(result["effect"], clean["effect"])
        self.assertEqual(result["path_predictions"], clean["path_predictions"])

    def test_cancel_after_final_chunk_and_reuse(self):
        stopped = False
        def progress(message, fraction):
            nonlocal stopped
            if message.startswith("Measuring c(t)"):
                stopped = True
        backend = Backend(POINTS)
        with self.assertRaises(InterruptedError):
            run(backend, 5, progress, lambda: stopped)
        self.assertEqual(run(backend)["curves"], run(Backend(POINTS))["curves"])


class API(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.library = Library(self.directory.name)
        self.patches = [patch.object(server, "REMOTE", False), patch.object(server, "LIBRARY", self.library),
                        patch.object(server, "JOBS", {}), patch.object(server, "LOCAL_ENGINE", None)]
        for item in self.patches:
            item.start()
        self.client = TestClient(server.app)
        self.record = run(Backend(POINTS))
        self.record["id"] = "a" * 32
        self.library.write("runs", self.record)

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def test_local_library_examples_selected_exports(self):
        self.assertTrue(self.client.get("/api/config").json()["local_library"])
        self.assertEqual(self.client.get("/classic/").status_code, 200)
        self.assertIn('src="app.js"', self.client.get("/classic/").text)
        self.assertIn("function renderCurves", self.client.get("/classic/app.js").text)
        self.assertIn("function renderEffect", self.client.get("/app.js").text)
        response = self.client.post("/api/examples", json={"id": self.record["id"], "tag": "loop", "notes": 'Unicode α\n"quoted", note'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.client.get("/api/library").json()["examples"]), 1)
        for fmt in ("jsonl", "csv"):
            exported = self.client.post("/api/export", json={"format": fmt, "scope": "examples", "ids": [self.record["id"]]})
            self.assertEqual(exported.status_code, 200, exported.text)
            self.assertIn("cumulative_length", exported.text)
            self.assertIn("α", exported.text)
        for ids, status in (([], 400), (["all"], 400), (["b" * 32], 404)):
            self.assertEqual(self.client.post("/api/export", json={"ids": ids}).status_code, status)
        self.assertEqual(self.client.post("/api/examples", json={"id": "../bad"}).status_code, 400)
        self.assertEqual(self.client.post("/api/examples", headers={"Origin": "https://elsewhere.example"}, json={}).status_code, 403)

    def test_remote_has_no_local_data_and_requires_key(self):
        with patch.object(server, "REMOTE", True), patch.object(server, "SERVER_KEY", False):
            for path in ("/api/library", "/api/export", "/api/hardware"):
                self.assertEqual(self.client.get(path).status_code, 404)
            self.assertEqual(self.client.post("/api/examples", json={}).status_code, 404)
            self.assertEqual(self.client.post("/api/run", json=REQUEST).status_code, 401)

    def test_server_cancellation_does_not_write_result(self):
        job_id = "b" * 32
        server.JOBS[job_id] = {"status": "queued"}
        def finish(*args, **kwargs):
            server.JOBS[job_id]["cancelled"] = True
            return dict(self.record)
        with patch.object(server, "LOCAL_ENGINE", SimpleNamespace(run=finish)):
            server.run_job(job_id, REQUEST, MODELS, None)
        self.assertEqual(server.JOBS[job_id]["status"], "cancelled")
        self.assertEqual(len(self.library.records("runs")), 1)

    def test_default_model_and_active_job_restore(self):
        with patch.object(server.WORKER, "submit"):
            request = dict(REQUEST)
            request.pop("model")
            response = self.client.post("/api/run", json=request)
            self.assertEqual(response.status_code, 202)
            job = self.client.get("/api/jobs/" + response.json()["id"]).json()
            self.assertEqual(job["request"]["model"], "gpt2")
            self.assertEqual(self.client.post("/api/run", json=request).status_code, 409)


if __name__ == "__main__":
    unittest.main(verbosity=2)
