# Role of makeOsdDb.py Script in curator_tools

## Purpose

The `makeOsdDb.py` script is responsible for creating and maintaining the **OpenSeizureDatabase (OSDB)** distribution files. These are anonymized JSON collections of seizure-like events derived from the live OSD Data Sharing system database, packaged into separate categories (general seizures, tonic-clonic seizures, falls, false alarms, normal activity, and unknown) by type and grouped together into time windows to ensure uniqueness.

## How It Works - The Main Pipeline

The script executes two principal pipelines that serve distinct use cases: a one-time creation mode for generating fresh database files, followed by an update mode that progressively enriches those installations with new seizure-like events from the server while preserving existing entries.

### Creation Mode (`--create`)

When run with the `--create` flag the script bypasses any pre-existing OSDB installation and builds entirely new database files from scratch using different event categories. It first retrieves all raw events from the server, then applies a series of filtering steps — removing warnings (unless they are tagged as seizures or falls), discarding test records, dropping entries whose descriptions contain 'test', applying date range filters if specified, and excluding data sources marked in the configuration (like Phone and AndroidWear). After that it groups events by user ID and time period so everything within a 3-minute window is treated as one event, picks the "best" record from each group (preferring manual alarms first, then tagged descriptions, then OSD-generated alarms), and categorizes them.

Next it downloads all datapoints for these events back onto the server. After that it tidies the retrieved data — expanding any embedded JSON strings into flat objects, pruning skipped elements to save space, adding seizure start/end times from a separate CSV file if available, correcting event alarm states based on their datapoints alarm states, and validating all events for sufficient datapoints before saving them to final JSON files.

### Update Mode (Default)

When no flags are passed the update pipeline takes over for maintaining an existing OSDB installation by progressively enriching it with new seizure-like events from the server while preserving existing entries. It begins exactly like creation mode — retrieving raw events, filtering and deduplicating them into unique event lists grouped by user ID and time period categorized as All Seizures (Seizure), Tonic Clonic Seizures (Tonic-Clonic), False Alarms (False Alarm), Falls (Fall), Normal Daily Activities (NDA), or Unknown. Then it compares this list against the current OSDB files stored locally, identifying which unique events are genuinely new versus ones not currently available. For any new seizure-like events (excluding those marked invalid in configuration) it downloads and tidies them through the same process as creation mode but specifically for the newly added events rather than rebuilding everything from scratch. Then it appends these fresh events to the existing database along with their associated datapoints, applies all the necessary tidying operations including data expansion, alarm state correction, seizure time updates if applicable, removal of invalid entries, and validation — writing both a comprehensive OSDB file containing all categories plus an index file for quick lookup.

After each update cycle it performs thorough validation on all events to ensure they have adequate datapoints and meets quality standards before adding them back into the distribution files along with necessary tidying operations such as extracting metadata like app versions, phone source data, watch SD version information, etc. The final output is a clean collection of JSON files organized by the different categories (seizures, tonic-clonic seizures, falls, false alarms, normal daily activities), with each category containing complete event and datapoint records ready for analysis or research purposes.

### Graphs Command (`graphs` sub-command)

Additionally, when invoked with the `graphs` command it generates summary graphs from JSON database files based on user-defined parameters such as threshold counts and output directory paths — though this is optional functionality rather than a core requirement of OSDB generation. It processes JSON event database files, computes descriptive statistics across all events including seizure durations for each seizure category (allSeizures, tonic-clonic seizures, falls) if applicable per the configuration file, calculates temporal distribution summaries by user ID and day of week along with their counts to identify peak activity days and frequencies, generates visualizations — typically 10 subplots per JSON file — covering daily seizure rates, user-specific seizure rate trends over time (25 points), total monthly event counts broken down by type (seizures, tonic-clonic seizures, falls, normal daily activities) with trend lines and labels, temporal density plots based on timestamps, average seizure duration metrics (mean, standard deviation, median) across different seizure categories, and false alarm rates for each distinct user ID (showing the proportion of warnings relative to total events), then saves these figures as PNG files in the specified output directory.

## Configuration

The script is fully controlled by an `osdb.cfg` config file that specifies:
- **Grouping period**: how long a time window counts before events are grouped
- **Data source inclusion/exclusion**: which data sources to include or exclude (e.g., Phone, AndroidWear)
- **Warning state handling**: whether warning-level alerts without seizure tags should be kept or dropped and why certain event categories exist in the final output, plus invalid events list.

## Output Structure

The script produces separate JSON files for each category of interest:
- `{prefix}_{groupingPeriod}_tcSeizures.json`: All tonic-clonic seizures (separate grouping)
- `{prefix}_{groupingPeriod}_allSeizures.json`: All seizure-like events (grouped together)
- `{prefix}_{groupingPeriod}_falseAlarms.json`: Events flagged as false alarms during processing
- `{prefix}_{groupingPeriod}_fallEvents.json`: Fall-related seizures/events (also separate grouping for clarity)

Each file contains arrays of event records with standard fields: id, dataTime, type, subType, userId, osdAlarmState, phoneAppVersion, watchAppVersion, dataSource. Each record includes full datapoint objects representing accelerometer readings at 50Hz sampling rates plus derived metrics like mean, SD, skewness, kurtosis, max frequency value and location, raw waveform data (truncated for storage efficiency), user-supplied descriptions where available, source metadata identifying the phone/watch/device origin, app version numbers for reference, and additional diagnostic information including alarm state details from individual datapoints.

## Important Notes

- The script does **not** generate an `unknownEvents.json` file — it intentionally skips this during creation because unknown events are typically normal daily activity or user-defined labels that don't require separate grouping treatment since they're expected to be contiguous in time and should stay ungrouped as-is
- When the script is run without any flags, it defaults to the "update" mode which enriches an existing installation by retrieving only new seizure-like events from the server rather than rebuilding the entire database — this means existing published event IDs are always preserved, with all newly added data thoroughly validated before inclusion in final files alongside necessary tidying operations such as extracting metadata and ensuring quality standards
- The script performs extensive validation of all downloaded events to ensure they have adequate datapoints; events with insufficient or corrupted data (like event 20595 which was identified as test data, event 61879 marked as bad) are automatically excluded based on configuration rules defined in osdb.cfg