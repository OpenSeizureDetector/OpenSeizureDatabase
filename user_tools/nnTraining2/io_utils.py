"""IO utility helpers for streaming flattened event CSVs and writing results

These helpers let us read the large flattened CSV without loading it all
into memory and write results incrementally so downstream processing can
be parallelised and memory-efficient.
"""
import pandas as pd
from typing import Generator, Tuple


def stream_events_from_flattened_csv(inFname: str, event_col: str = 'eventId', chunksize: int = 20000) -> Generator[Tuple[object, pd.DataFrame], None, None]:
    """Yield (eventId, dataframe) for each contiguous event in a flattened CSV.

    The input CSV is assumed to be ordered by rows belonging to each event
    (i.e. rows for an event appear together). We read by chunks and group
    inside each chunk while keeping any event that spans chunk boundaries
    in a persistent buffer.
    """
    reader = pd.read_csv(inFname, chunksize=chunksize)
    current_event_id = None
    buffered_frames = []

    for chunk in reader:
        # Ensure consistent dtypes for grouping
        for event_id, grp in chunk.groupby(event_col, sort=False):
            if current_event_id is None:
                current_event_id = event_id
                buffered_frames = [grp]
            elif event_id == current_event_id:
                buffered_frames.append(grp)
            else:
                # finish previous event
                yield current_event_id, pd.concat(buffered_frames, ignore_index=True)
                current_event_id = event_id
                buffered_frames = [grp]

    # yield last buffered event
    if current_event_id is not None and buffered_frames:
        yield current_event_id, pd.concat(buffered_frames, ignore_index=True)


def write_rows_batch_csv(outFname: str, rows, header: bool = False, mode: str = 'a'):
    """Write a batch of rows (list of dicts or DataFrame) to CSV.

    The function is deliberately minimal: it constructs a DataFrame then
    uses pandas' CSV writer. Calling code is responsible for choosing
    header/mode and keeping writes atomic enough for its use.
    """
    if rows is None or len(rows) == 0:
        return
    if not hasattr(rows, 'columns'):
        df = pd.DataFrame(rows)
    else:
        df = rows
    df.to_csv(outFname, mode=mode, header=header, index=False)
