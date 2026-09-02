# 🤖 Agent Instructions for makeOsdDb

## 🚀 Core Workflow & Setup
**CRITICAL:** Before any update/processing task involving local edits, ensure schema migration and change detection are handled.

1.  **Schema Migration (One-time/Required after updates):**
    `python3 src/schema_migration_v2.py --db <path_to_db>`
2.  **Preserve Local Edits:** If you have existing edits, identify and mark them:
    *   Dry run: `python3 src/detect_local_changes.py --db <path_to_db> --json-dir <path_to_json> --dry-run`
    *   Apply: `python3 src/detect_local_changes.py --db <path_to_db> --json-dir <path_to_json>`

## 🛠️ Essential Commands

### Database Update (Download & Process)
*   **Standard Update:** `python3 makeOsdDb_refactored_wrapper.py --osdb-dir <osdb_directory>`
*   **Date Filtered:** Use `--start YYYY-MM-DD --end YYYY-MM-DD`
*   **Debug Mode:** Enable with `--debug`

### Database Management (`manage_events.py`)
*Always provide `--db <path_to_db>`. No database path = failure.*
*   **Stats:** `python3 manage_events.py stats --db <path_to_db>`
*   **Validation:** `python3 manage_events.py validate --db <path_to_db>` (Run this after any destructive operation or schema change)
*   **List/Show:** `list` and `show --event-id <ID>` for auditing changes.

### Publication (Export to JSON)
*   **Standard Export:** `python3 makeOsdDb_refactored_wrapper.py --osdb-dir <osdb_dir> --publish`
*   **Full Package:** `--publish --generate-index --generate-graphs` (Generates CSVs and summary charts)

## ⚙️ Configuration & Rules
*   **`osdb.cfg`**: The source of truth for processing logic.
    *   Check `groupingPeriod` (e.g., `3min`)—it defines the sliding window for event merging.
    *   Check `excludeDataSources` and `invalidEvents` before assuming data is missing.
*   **`client.cfg`**: Ensure API credentials are valid; otherwise, downloads will fail silently or with errors.
*   **Grouping Logic:** Events are merged if they are within the `groupingPeriod`. **NDA events are NOT grouped.**
*   **Data Integrity:** Uses SQLite foreign keys and `CASCADE DELETE`. Manual deletions/edits must be validated via `manage_events.py validate`.

## 🧪 Testing
*   **Run all tests:** `pytest tests/` or `python3 tests/test_wrapper_integration.py`
*   **Key Areas to Verify:** Database utility integrity, integration of the wrapper, and event preservation logic.
