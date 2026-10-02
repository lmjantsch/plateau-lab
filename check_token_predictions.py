"""Offline probability and patched autoregression checks with a tiny real model."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from engine import Engine
from plateau.core.math import top_candidates
from plateau.core.predictions import sample_prediction


class Tokenizer:
    eos_token_id = None
    prompts = {"A": [1, 2, 3], "B": [1, 4, 5], "Long B": [6, 7, 8, 9]}

    def encode(self, text, **kwargs):
        return self.prompts[text].copy()

    def decode(self, ids, **kwargs):
        return "".join(f" word{token}" for token in ids)


class Predictions(unittest.TestCase):
    def test_probabilities_are_full_vocabulary_and_greedy_ties_match(self):
        scores = torch.tensor([[4., 3., 2., 1., 0.], [1., 1., 1., 1., 1.]])
        ids, probabilities = top_candidates(scores)
        self.assertEqual(ids[0].tolist(), [0, 1, 2])
        self.assertEqual(ids[1, 0].item(), scores[1].argmax().item())
        for row in ids:
            self.assertEqual(len(set(row.tolist())), 3)
        expected = scores.double().exp() / scores.double().exp().sum(-1, keepdim=True)
        torch.testing.assert_close(probabilities.double(), expected.gather(-1, ids), atol=1e-7, rtol=1e-6)
        self.assertLess(float(probabilities[0].sum()), 1)

    def test_eos_and_nonfinite_do_not_invent_future_predictions(self):
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = 9
        ids = [[1, 2, 9], [9, 3, 4], [5, 6, 7]]
        probs = [[.5, .2, .1]] * 3
        matrix = sample_prediction(tokenizer, .5, ids, probs)["token_matrix"]
        self.assertEqual(matrix["stop_reason"], "eos")
        self.assertEqual(len(matrix["steps"]), 2)
        self.assertTrue(matrix["steps"][0][2]["is_eos"])
        probs[1] = [float("nan")] * 3
        record = sample_prediction(tokenizer, .5, ids, probs)
        self.assertEqual(record["token_matrix"]["stop_reason"], "nonfinite")
        self.assertEqual(len(record["token_matrix"]["steps"]), 1)
        json.dumps(record, allow_nan=False)

    def setUp(self):
        torch.manual_seed(812)
        torch.set_num_threads(2)
        self.engine = Engine()
        self.engine.model = GPT2LMHeadModel(GPT2Config(vocab_size=23, n_positions=128,
            n_embd=16, n_layer=2, n_head=2, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
            bos_token_id=0, eos_token_id=None, attn_implementation="eager")).eval()
        self.engine.tokenizer = Tokenizer()
        self.engine.hardware = SimpleNamespace(device="cpu", batch_size=lambda *args: 2,
            clear_cache=lambda: None, describe=lambda: {"name": "test CPU"})
        self.engine.load = lambda *args: None
        self.request = dict(model="gpt2", sequence_a="A", sequence_b="B", patch_layer=0,
            patch_position="different_suffix", context="a", steps=5, interpolation="linear")

    @torch.inference_mode()
    def independent(self, request, t):
        """Full-prefix oracle: manually patch fixed positions and append raw argmax.

        No production interpolation, candidate-ranking, or prediction-format helper
        is used here. The generated positions are deliberately outside the patch.
        """
        engine = self.engine
        a, b = [engine.tokenizer.encode(request[key]) for key in ("sequence_a", "sequence_b")]
        if request["patch_position"] == "different_suffix":
            start = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            starts = [start, start]
        else:
            starts = [len(a)-1, len(b)-1]
        block = engine.model.transformer.wte if request["patch_layer"] == -1 else engine.model.transformer.h[request["patch_layer"]]
        captured = []
        def capture(_, args, output):
            captured.append((output[0] if isinstance(output, tuple) else output).clone())
        handle = block.register_forward_hook(capture)
        try:
            for tokens in (a, b):
                engine.model(torch.tensor([tokens]), use_cache=False)
        finally:
            handle.remove()
        replacement = (1-t)*captured[0][:, starts[0]:] + t*captured[1][:, starts[1]:]
        side = 0 if request["context"] == "a" else 1
        tokens = (a if side == 0 else b).copy()
        prompt_length = len(tokens)
        def intervene(_, args, output):
            hidden = (output[0] if isinstance(output, tuple) else output).clone()
            hidden[:, starts[side]:prompt_length] = replacement
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        handle = block.register_forward_hook(intervene)
        steps = []
        try:
            for _ in range(3):
                scores = engine.model(torch.tensor([tokens]), use_cache=False).logits[0, -1].double()
                probabilities = (scores - scores.max()).exp()
                probabilities /= probabilities.sum()
                ordered = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))[:3]
                steps.append([(i, float(probabilities[i])) for i in ordered])
                tokens.append(ordered[0])
        finally:
            handle.remove()
        return steps

    def test_each_column_matches_independent_patched_autoregression(self):
        for layer, position, context, b in [(0, "different_suffix", "a", "B"),
                (0, "last_token", "b", "Long B"), (-1, "different_suffix", "b", "B")]:
            request = dict(self.request, patch_layer=layer, patch_position=position, context=context, sequence_b=b)
            record = self.engine.run(request, all_layers=True)
            self.assertEqual(len(record["path_predictions"]), 5)
            for sample in record["path_predictions"]:
                expected = self.independent(request, sample["t"])
                columns = sample["token_matrix"]["steps"]
                self.assertEqual(len(columns), 3)
                self.assertEqual(sample["token_id"], columns[0][0]["id"])
                for column, target in zip(columns, expected):
                    self.assertEqual([c["id"] for c in column], [i for i, _ in target])
                    for actual, (_, probability) in zip(column, target):
                        self.assertAlmostEqual(actual["probability"], probability, places=6)
            self.assertTrue(all(not module._forward_hooks for module in self.engine.model.modules()))

    def test_oom_during_continuation_and_cancel_remove_hooks(self):
        failed = False
        def fail_once(_, args):
            nonlocal failed
            if args[0].shape == (2, 4) and not failed:
                failed = True
                raise torch.OutOfMemoryError("OOM on the second prediction column")
        handle = self.engine.model.register_forward_pre_hook(fail_once)
        try:
            actual = self.engine.run(self.request)
        finally:
            handle.remove()
        with patch.object(self.engine.hardware, "batch_size", return_value=1):
            expected = self.engine.run(self.request)
        self.assertEqual(actual["settings"]["batch_retries"], 1)
        self.assertEqual(actual["path_predictions"], expected["path_predictions"])
        self.assertEqual(actual["curves"], expected["curves"])
        stopped = False
        def stop_after_column(_, args):
            nonlocal stopped
            if args[0].shape == (2, 4):
                stopped = True
        handle = self.engine.model.register_forward_pre_hook(stop_after_column)
        try:
            with self.assertRaises(InterruptedError):
                self.engine.run(self.request, cancelled=lambda: stopped)
        finally:
            handle.remove()
        self.assertTrue(all(not module._forward_hooks for module in self.engine.model.modules()))
        with patch.object(self.engine.hardware, "batch_size", return_value=1):
            self.assertEqual(self.engine.run(self.request)["path_predictions"], expected["path_predictions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
