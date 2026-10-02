"""Real-model arc check using independently captured vectors and Python math.dist.

No user records are read or written. --output writes a fixture for isolated UI
checks. Uses the normal cache; a missing checkpoint downloads on first use.
"""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from engine import Engine
from models import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pythia-70m', choices=list(MODELS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    engine = Engine()
    engine.load(args.model, lambda *_: None)
    request = dict(model=args.model, sequence_a='The house was big', sequence_b='The house was in',
                   context='a', patch_layer=0, patch_position='different_suffix', steps=7)
    measure = engine._measure_path
    observed, references = {}, {}

    def independently_capture(*args):
        blocks, layers, ts, batch_size = args[4], args[5], args[6], args[9]
        context_length = args[2].shape[1]
        latest, handles = {}, []
        observed.clear()
        references.clear()
        calls = 0
        reference_calls = 2 if batch_size == 1 else 1

        def capture_layer(key):
            def hook(_, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                latest[key] = hidden[:, -1, :].detach().float().cpu().clone()
            return hook

        def capture_forward(_, inputs, output):
            nonlocal calls
            # The matrix's later columns append generated tokens. Only forwards
            # over the original context belong to the measured trajectory.
            if inputs[0].shape[1] != context_length:
                return
            calls += 1
            latest['logits'] = output.logits[:, -1, :].detach().float().cpu().clone()
            target = references if calls <= reference_calls else observed
            remaining = 2-sum(len(x) for x in target.get('logits', [])) if target is references else len(ts)-sum(len(x) for x in observed.get('logits', []))
            for key, values in latest.items():
                target.setdefault(key, []).append(values[:min(batch_size, remaining)].clone())

        try:
            for layer in layers:
                handles.append(blocks[layer].register_forward_hook(capture_layer(str(layer))))
            handles.append(engine.model.register_forward_hook(capture_forward))
            return measure(*args)
        finally:
            for handle in handles:
                handle.remove()

    results = []
    for method in ('linear', 'slerp'):
        # Production MPS policy uses one sample for Pythia/Qwen. CPU also tests
        # padded final batches and every transition between inference batches.
        batches = (1,) if engine.device == 'mps' else (1, 2, 4)
        same_method = []
        for batch in batches:
            with patch.object(engine, '_measure_path', independently_capture), \
                 patch.object(engine.hardware, 'batch_size', return_value=batch):
                result = engine.run(dict(request, interpolation=method))
            json.dumps(result, allow_nan=False)
            for curve in result['curves']:
                vectors = torch.cat(observed[curve['key']])
                assert len(vectors) == request['steps']
                points = vectors.double().tolist()
                segments = [0] + [math.dist(a, b) for a, b in zip(points, points[1:])]
                cumulative = [math.fsum(segments[:i+1]) for i in range(len(points))]
                expected = [length/cumulative[-1] for length in cumulative]
                for field, values in [('step_lengths', segments), ('cumulative_length', cumulative), ('c', expected)]:
                    torch.testing.assert_close(torch.tensor(curve[field], dtype=torch.float64),
                                               torch.tensor(values, dtype=torch.float64), atol=1e-9, rtol=1e-10)
                endpoints = torch.cat(references[curve['key']])
                da = (vectors-endpoints[0]).norm(dim=-1)
                db = (vectors-endpoints[1]).norm(dim=-1)
                torch.testing.assert_close(torch.tensor(curve['d']), da/(da+db), atol=2e-6, rtol=2e-6)
                assert abs(curve['total_length']-cumulative[-1]) < 1e-8
            assert all(not m._forward_hooks for m in engine.model.modules())
            same_method.append(result)
            print(f'PASS: {args.model} {engine.device}, {method}, batch {batch}: independent vector capture, arc lengths, d arithmetic, padding and JSON.', flush=True)
        for other in same_method[1:]:
            for first, curve in zip(same_method[0]['curves'], other['curves']):
                for metric in ('c', 'd', 'cumulative_length'):
                    torch.testing.assert_close(torch.tensor(first[metric]), torch.tensor(curve[metric]), atol=2e-4, rtol=2e-4)
        results.append(same_method[0])

    equal_endpoints = engine.run(dict(request, sequence_b='The cat was big', patch_position='last_token',
                                      patch_layer=-1, interpolation='linear'))
    for curve in equal_endpoints['curves']:
        assert curve['d_status'] == 'coincident_endpoints'
        # Even identical sources can pick up roundoff in (1-t)*a+t*b. The
        # zero arc tolerance must count any actual, however tiny, movement.
        assert curve['c_status'] == ('stationary' if curve['total_length'] == 0 else 'ok')
    json.dumps(equal_endpoints, allow_nan=False)
    print('PASS: equal endpoints keep d undefined while c follows actual measured movement.', flush=True)

    # Zero only the final block's readout. Earlier measured layers still move;
    # the final residual and logits are stationary. One undefined readout must
    # not suppress other valid curves or crash summary creation.
    blocks, _ = engine.architecture()
    def stationary_final(_, inputs, output):
        hidden = (output[0] if isinstance(output, tuple) else output).clone()
        hidden[:, -1, :] = 0
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    handle = blocks[-1].register_forward_hook(stationary_final)
    try:
        mixed = engine.run(dict(request, interpolation='linear'))
    finally:
        handle.remove()
    assert mixed['curves'][0]['c_status'] == 'ok'
    assert mixed['curves'][-1]['c_status'] == 'stationary'
    assert mixed['metrics']['c']['max_abs_slope'] is None
    json.dumps(mixed, allow_nan=False)
    print('PASS: stationary readouts coexist with valid layer curves; undefined summaries serialize safely.', flush=True)
    if args.output:
        args.output.write_text(json.dumps(results[1], allow_nan=False))


if __name__ == '__main__':
    main()
