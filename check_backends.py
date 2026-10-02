"""Real cached Pythia parity and independent vectors through nnsight traces.

Runs nnsight locally: no NDIF key or remote inference is used. Pass --output to
save a schema-7 fixture outside the user library for browser/HTTP validation.
"""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from engine import Engine
from models import cached_snapshot
import plateau.backends.nnsight_backend as nb
from plateau.core.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = Engine()
    engine.load("pythia-70m", lambda *_: None)
    backend = nb.NnsightBackend(remote=False, device=engine.device)
    catalog = {"pythia-70m": {"repo": str(cached_snapshot("pythia-70m")), "label": "Pythia 70M"}}
    request = dict(model="pythia-70m", sequence_a="The house was big", sequence_b="The house was in",
                   context="a", patch_layer=0, patch_position="different_suffix", steps=7)
    original = nb.arc_segments
    for method in ("linear", "slerp"):
        local = engine.run(dict(request, interpolation=method), all_layers=True)
        # Capture exactly the tensors the nnsight trace supplies to arc measurement;
        # calculate distances independently using Python math.dist / math.fsum.
        captured = []
        def capture(vectors, previous=None):
            captured.append(vectors.detach().cpu().double().tolist())
            return original(vectors, previous)
        with patch.object(nb, "arc_segments", capture):
            remote_contract = run_experiment(backend, dict(request, interpolation=method), 3, models=catalog)
        readouts = remote_contract["effect"]["rows"]
        for index, row in enumerate(readouts):
            points = [point for batch in captured[index::len(readouts)] for point in batch]
            assert len(points) == request["steps"]
            steps = [0.] + [math.dist(a, b) for a, b in zip(points, points[1:])]
            cumulative = [math.fsum(steps[:i + 1]) for i in range(len(steps))]
            for actual, expected in zip(row["step_lengths"], steps):
                assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), (row["key"], actual, expected)
            assert math.isclose(row["total_length"], cumulative[-1], rel_tol=1e-10, abs_tol=1e-9)
            torch.testing.assert_close(torch.tensor(row["values"]["c"]), torch.tensor([v / cumulative[-1] for v in cumulative]))
            peer = next(r for r in local["effect"]["rows"] if r["key"] == row["key"])
            for metric in ("c", "relative_l2_shinkle", "relative_l2_janiak", "relative_l2_projected"):
                torch.testing.assert_close(torch.tensor(row["values"][metric]), torch.tensor(peer["values"][metric]),
                                           atol=2e-3, rtol=2e-3)
        json.dumps(remote_contract, allow_nan=False)
        for sample, peer in zip(remote_contract["path_predictions"], local["path_predictions"]):
            assert sample["token_id"] == peer["token_id"]
            matrix, reference = sample["token_matrix"], peer["token_matrix"]
            assert matrix["stop_reason"] == reference["stop_reason"]
            assert len(matrix["steps"]) == len(reference["steps"])
            for candidates, expected in zip(matrix["steps"], reference["steps"]):
                assert [t["id"] for t in candidates] == [t["id"] for t in expected]
                torch.testing.assert_close(torch.tensor([t["probability"] for t in candidates]),
                                           torch.tensor([t["probability"] for t in expected]), atol=2e-5, rtol=2e-3)
        print(f"PASS {engine.device} {method}: all-layer local/nnsight parity and independent trace-vector arc norms, "
              "including batch boundaries, singleton final batch and all three top-3 prediction columns.", flush=True)
    if args.output:
        local["id"] = "c" * 32
        args.output.write_text(json.dumps(local, allow_nan=False))


if __name__ == "__main__":
    main()
