"""Real HTTP/CLI application smoke test in a temporary collection. Not a GUI test."""
import argparse
import copy
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pythia-70m')
    parser.add_argument('--fastapi', action='store_true', help='Check the integrated server and schema 7.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='plateau-http-') as directory:
        data = Path(directory).resolve() / 'data'
        log = Path(directory) / 'server.log'
        with log.open('w') as output:
            command = [sys.executable, '-m', 'server.launch'] if args.fastapi else [sys.executable, str(root/'app.py')]
            server = subprocess.Popen(command + ['--port', '0',
                                       '--data-dir', str(data)], stdout=output, stderr=output, cwd=root)
        try:
            deadline = time.monotonic()+20
            url = None
            while time.monotonic() < deadline:
                match = re.search(r'http://127.0.0.1:\d+', log.read_text())
                if match:
                    url = match.group()
                    break
                assert server.poll() is None, log.read_text()
                time.sleep(.1)
            assert url, log.read_text()

            def request(path, body=None, raw=False):
                req = Request(url+path, data=None if body is None else json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json'})
                with urlopen(req, timeout=60) as response:
                    payload = response.read()
                    return payload if raw else json.loads(payload)

            assert request('/api/config')['data_path'] == str(data)
            page = request('/classic/' if args.fastapi else '/', raw=True).decode()
            assert page.index('id="c-section"') < page.index('id="d-section"')
            if args.fastapi:
                assert b'id="effect-section"' in request('/', raw=True)
            assert b'PlateauTokenMatrix' in request('/token-matrix.js', raw=True)
            assert b'.token-matrix-table' in request('/token-matrix.css', raw=True)
            for method in ('linear', 'slerp'):
                job_id = request('/api/run', dict(model=args.model, sequence_a='The house was big',
                    sequence_b='The house was in', patch_layer=0, interpolation=method, steps=21,
                    patch_position='different_suffix', context='a'))['id']
                deadline = time.monotonic()+180
                while time.monotonic() < deadline:
                    job = request('/api/jobs/'+job_id)
                    assert job['status'] not in ('error', 'cancelled'), job
                    if job['status'] == 'done':
                        break
                    time.sleep(.1)
                assert job['status'] == 'done', job
                record = job['result']
                assert record['schema_version'] == (7 if args.fastapi else 4)
                assert len(record['path_predictions']) == 21
                assert all(len(sample['token_matrix']['steps']) == 3 or sample['token_matrix']['stop_reason']
                           for sample in record['path_predictions'])
                assert all(len(column) == 3 for sample in record['path_predictions']
                           for column in sample['token_matrix']['steps'])
                if args.fastapi:
                    assert len(record['effect']['rows']) == len(record['l2_distances']['layers']) + 1
                assert all(c['c'][0] == 0 and c['c'][-1] == 1 for c in record['curves'])
                assert json.loads((data/'runs'/f'{job_id}.json').read_text()) == record
                request('/api/examples', dict(id=job_id, tag='Plateau', notes='HTTP arc round trip'))
                saved = next(r for r in request('/api/library')['examples'] if r['id'] == job_id)
                assert saved['curves'] == record['curves']
                assert saved['path_predictions'] == record['path_predictions']
                assert saved['metric_definitions'] == record['metric_definitions']
                print(f'PASS HTTP: {method}, real {args.model} on {record["device"]}, run/poll/save/reload.', flush=True)

            # Seed a d-only record solely in this disposable collection. Reading
            # and exporting it must not add c or rewrite any of its bytes.
            legacy = copy.deepcopy(record)
            legacy['id'], legacy['schema_version'] = 'f'*32, 3
            legacy.pop('effect', None)
            legacy.pop('metric_definitions')
            legacy.pop('primary_metric')
            legacy['metrics'].pop('c')
            legacy['metrics'].pop('d')
            for curve in legacy['curves']:
                for key in list(curve):
                    if key not in ('key', 'title', 't', 'd'):
                        del curve[key]
            legacy_path = data/'runs'/f'{legacy["id"]}.json'
            legacy_path.write_text(json.dumps(legacy, allow_nan=False))
            original = legacy_path.read_bytes()
            history = request('/api/library')['history']
            assert next(r for r in history if r['id'] == legacy['id']) == legacy
            assert request('/api/config')['active_job'] is None
            for scope in ('history', 'examples'):
                ids = [record['id'], legacy['id']] if scope == 'history' else [record['id']]
                payload = dict(scope=scope, ids=ids)
                exported = request('/api/export', dict(payload, format='jsonl'), raw=True)
                decoded = [json.loads(line) for line in exported.splitlines()]
                assert decoded[0]['curves'] == record['curves']
                assert decoded[0]['path_predictions'] == record['path_predictions']
                assert decoded[0]['metric_definitions'] == record['metric_definitions']
                if scope == 'history':
                    assert decoded[1] == legacy
                csv_body = request('/api/export', dict(payload, format='csv'), raw=True)
                rows = list(csv.DictReader(io.StringIO(csv_body.decode('utf-8-sig'))))
                samples = len(record['curves']) * record['settings']['steps']
                assert len(rows) == samples * len(ids)
                for index, row in enumerate(rows[:samples]):
                    curve = record['curves'][index//record['settings']['steps']]
                    i = index % record['settings']['steps']
                    for column in ('c', 'd', 'cumulative_length'):
                        assert float(row[column]) == curve[column][i]
                    assert float(row['total_length']) == curve['total_length']
                assert all(row['c'] == '' and row['c_status'] == 'not_recorded' for row in rows[samples:])
            assert legacy_path.read_bytes() == original
            assert 'Traceback' not in log.read_text(), log.read_text()
            print('PASS HTTP: selected CSV/JSONL, both collections, metadata, raw lengths, legacy preservation and no server errors. Browser checks not performed.', flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    main()
