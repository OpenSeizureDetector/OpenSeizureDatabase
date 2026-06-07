"""output_folders.py – Numbered run-folder management for testRunner.

Each test run produces its own sequentially numbered subdirectory inside
``<outDir>/testRun/``, e.g. ``./output/testRun/1/``, ``./output/testRun/2/``.

Pass ``rerun=N`` to open an *existing* folder rather than creating a new one.
"""
import os


def getLatestOutputFolder(outPath="./output", prefix="testRun"):
    """Return the path of the highest-numbered subdirectory inside
    ``outPath/prefix/``, or ``None`` if none exist yet.
    """
    folder = os.path.join(outPath, prefix)
    if not os.path.exists(folder):
        return None
    latest_num = 0
    latest_path = None
    try:
        for entry in os.scandir(folder):
            if entry.is_dir():
                try:
                    num = int(entry.name)
                    if num > latest_num:
                        latest_num = num
                        latest_path = entry.path
                except ValueError:
                    pass
    except OSError:
        pass
    return latest_path


def getOutputPath(outPath="./output", rerun=0, prefix="testRun"):
    """Return (and optionally create) a numbered output folder.

    * ``rerun=0``  → create the next sequential folder (``outPath/prefix/N+1``).
    * ``rerun=N``  → reuse existing folder ``outPath/prefix/N``
      (raises ``FileNotFoundError`` if missing).
    """
    if rerun == 0:
        latest = getLatestOutputFolder(outPath, prefix)
        latest_num = int(os.path.basename(latest)) if latest is not None else 0
        new_num = latest_num + 1
        new_path = os.path.join(outPath, prefix, str(new_num))
        os.makedirs(new_path, exist_ok=False)
    else:
        new_path = os.path.join(outPath, prefix, str(rerun))
        if not os.path.exists(new_path):
            raise FileNotFoundError(
                f"Rerun folder not found: {new_path}. "
                f"Expected structure: {outPath}/{prefix}/{rerun}"
            )
    print(f"getOutputPath() → {new_path}")
    return new_path
