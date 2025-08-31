"""Benchmark streaming vs in-memory extract_features runs on a small synthetic dataset.

Usage: run inside project venv with PYTHONPATH set to repo root.
"""
import time
import os
import resource
import tempfile
import pandas as pd
from user_tools.nnTraining2 import extractFeatures as ef


def make_large_flattened_csv(path, n_events=100, rows_per_event=2):
    cols = ['eventId','dataTime','userId','typeStr','type','osdAlarmState','osdSpecPower','osdRoiPower','hr','o2sat']
    for prefix in ['M','X','Y','Z']:
        for i in range(125):
            cols.append(f"{prefix}{i:03d}")

    rows = []
    for e in range(n_events):
        for r in range(rows_per_event):
            row = {}
            for c in cols:
                if c == 'eventId':
                    row[c] = f'E{e}'
                elif c == 'dataTime':
                    row[c] = r * 125
                elif c == 'userId':
                    row[c] = 'U1'
                elif c == 'typeStr':
                    row[c] = 'T'
                elif c == 'type':
                    row[c] = 1 if (e % 10 == 0) else 0
                elif c in ['osdAlarmState','osdSpecPower','osdRoiPower','hr','o2sat']:
                    row[c] = 0
                else:
                    row[c] = float(r)
            rows.append(row)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def measure(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in kilobytes on Linux
    return end - start, usage.ru_maxrss / 1024.0


def run_bench():
    fd, tmp = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        make_large_flattened_csv(tmp, n_events=200, rows_per_event=2)
        config = {
            'dataProcessing': {'window':125, 'step':125, 'features':[]},
            'dataFileNames': {}
        }

        # In-memory run (will load full CSV)
        t_mem, m_mem = measure(lambda: ef.extract_features(pd.read_csv(tmp), config))
        print(f"In-memory: time={t_mem:.2f}s, peak_mem={m_mem:.1f}MB")

        # Streaming run (pass filename)
        t_stream, m_stream = measure(lambda: ef.extract_features(tmp, config))
        print(f"Streaming: time={t_stream:.2f}s, peak_mem={m_stream:.1f}MB")
    finally:
        os.remove(tmp)


if __name__ == '__main__':
    run_bench()
