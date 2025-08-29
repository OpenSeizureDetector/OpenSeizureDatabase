import os
import tempfile
import pandas as pd
from user_tools.nnTraining2 import io_utils


def make_flattened_csv(path):
    # small CSV with two events: eventA (3 rows), eventB (2 rows)
    rows = []
    for eventId, nrows in [('A', 3), ('B', 2)]:
        for i in range(nrows):
            row = {'eventId': eventId, 'dataTime': i, 'type': 1 if eventId == 'A' else 0, 'M000': 0.1 * i}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_stream_events_from_flattened_csv():
    fd, tmp = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        make_flattened_csv(tmp)
        events = list(io_utils.stream_events_from_flattened_csv(tmp, event_col='eventId', chunksize=2))
        assert len(events) == 2
        assert events[0][0] == 'A'
        assert len(events[0][1]) == 3
        assert events[1][0] == 'B'
        assert len(events[1][1]) == 2
    finally:
        os.remove(tmp)
