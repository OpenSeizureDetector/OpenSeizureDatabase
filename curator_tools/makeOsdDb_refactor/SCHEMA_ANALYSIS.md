# Schema Comparison: JSON vs SQLite

## Executive Summary

**Risk Assessment: MEDIUM** - Current SQLite schema is missing several fields that exist in JSON files. Need to update schema before data migration.

## Detailed Field-by-Field Comparison

### Event-Level Fields

| Field Name | JSON | SQLite | Storage Location | Notes |
|------------|------|--------|------------------|-------|
| **Core Identity** |
| id | ✅ | ✅ | events.id | PRIMARY KEY |
| userId | ✅ | ✅ | events.userId | NOT NULL |
| dataTime | ✅ | ✅ | events.dataTime | ISO 8601 format |
| dataTimeEnd | ⚠️ | ✅ | events.dataTimeEnd | Computed from datapoints |
| type | ✅ | ✅ | events.type | Seizure/Fall/False Alarm/etc |
| subType | ✅ | ✅ | events.subType | Tonic-Clonic/Other/etc |
| desc | ✅ | ✅ | events.desc | User description |
| **Alarm State** |
| osdAlarmState | ✅ | ✅ | events.osdAlarmState | 0=OK, 1=WARN, 2=ALARM, 3=FALL |
| alarmPhrase | ✅ | ✅ | events.alarmPhrase | Text version of alarm |
| alarmRationale | ✅ | ✅ | events.alarmRationale | Why alarm was raised |
| alarmTime | ✅ | ✅ | events.alarmTime | When alarm triggered |
| **Device Information** |
| dataSourceName | ✅ | ✅ | events.dataSourceName | Watch/Phone/AndroidWear |
| phoneAppVersion | ✅ | ✅ | events.phoneAppVersion | Phone app version |
| watchSdVersion | ✅ | ✅ | events.watchSdVersion | Watch software version |
| watchSdName | ✅ | ✅ | events.watchSdName | Watch app name |
| watchPartNo | ✅ | ✅ | events.watchPartNo | Watch part number |
| watchSerialNo | ✅ | ✅ | events.watchSerialNo | Watch serial number |
| watchFwVersion | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| **Data Availability Flags** |
| has3dData | ✅ | ✅ | events.has3dData | ≥50% datapoints have 3D |
| hasHrData | ✅ | ✅ | events.hasHrData | ≥50% datapoints have HR |
| hasO2SatData | ✅ | ✅ | events.hasO2SatData | ≥50% datapoints have O2 |
| **Alarm Configuration** |
| alarmThresh | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| alarmRatioThresh | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| alarmFreqMin | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| alarmFreqMax | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| **HR/O2Sat Configuration** |
| hrThreshMin | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| hrThreshMax | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| o2SatAlarmActive | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| o2SatAlarmStanding | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| o2SatThreshMin | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| **Battery & System** |
| batteryPc | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| sdMode | ✅ | ❌ | metadata (JSON) | Deprecated, low priority |
| **Manual Corrections** |
| seizureTimes | ✅ | ❌ | **MISSING** | **⚠️ CRITICAL - manual edits** |
| **Merge Metadata** |
| merged_from_events | ✅ | ✅ | events.merged_from_events | Event IDs that were merged |
| merged_event_count | ✅ | ✅ | events.merged_event_count | Count of merged events |
| **Statistics** |
| duration_seconds | ⚠️ | ✅ | events.duration_seconds | Computed field |
| datapoint_count | ⚠️ | ✅ | events.datapoint_count | Count of datapoints |
| **Internal** |
| metadata | ❌ | ✅ | events.metadata | JSON blob for extras |
| last_modified | ❌ | ✅ | events.last_modified | Tracking field |

### Datapoint-Level Fields

| Field Name | JSON | SQLite | Storage Location | Notes |
|------------|------|--------|------------------|-------|
| **Core** |
| id | ✅ | ✅ | datapoints.id | Auto-increment |
| event_id | ✅ | ✅ | datapoints.event_id | FOREIGN KEY |
| dataTime | ✅ | ✅ | datapoints.dataTime | Timestamp |
| alarmState | ✅ | ✅ | datapoints.alarmState | 0/1/2/3/5 |
| **Sensor Data** |
| hr | ✅ | ✅ | datapoints.hr | Heart rate (bpm) |
| o2Sat | ✅ | ✅ | datapoints.o2Sat | Oxygen saturation (%) |
| rawData | ✅ | ✅ | datapoints.rawData | JSON array of 125 values |
| rawData3D | ✅ | ✅ | datapoints.rawData3D | JSON array of 375 values |
| **Analysis Fields** |
| specPower | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| roiPower | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| roiRatio | ✅ | ❌ | **MISSING** | **⚠️ Data Loss Risk** |
| maxVal | ✅ | ❌ | **MISSING** | Usually in skipElements |
| maxFreq | ✅ | ❌ | **MISSING** | Usually in skipElements |

## Summary of Missing Fields

### CRITICAL (Must Add - Manual User Data)
1. **seizureTimes** - Manually corrected seizure start/end times from CSV file
   - Type: Array of 2 floats [start_seconds, end_seconds]
   - Usage: Manual corrections to auto-detected seizure boundaries
   - **Impact: Loss of manual curation work**

### HIGH PRIORITY (Configuration & Analysis)
2. **alarmThresh** - Threshold for alarm detection
3. **alarmRatioThresh** - Ratio threshold for alarm
4. **alarmFreqMin** - Minimum frequency for alarm
5. **alarmFreqMax** - Maximum frequency for alarm
6. **hrThreshMin** - HR alarm lower threshold
7. **hrThreshMax** - HR alarm upper threshold
8. **o2SatThreshMin** - O2 saturation alarm threshold
9. **o2SatAlarmActive** - Whether O2 alarm is enabled
10. **o2SatAlarmStanding** - O2 alarm standing status
11. **batteryPc** - Watch battery percentage
12. **watchFwVersion** - Watch firmware version
13. **specPower** (datapoint) - Spectrum power analysis
14. **roiPower** (datapoint) - Region of interest power
15. **roiRatio** (datapoint) - ROI to spectrum ratio

### MEDIUM PRIORITY (Useful but Computed/Deprecated)
16. **maxVal** (datapoint) - Maximum acceleration value (usually skipped)
17. **maxFreq** (datapoint) - Frequency of max power (usually skipped)
18. **sdMode** - System mode (deprecated)

## Recommended Schema Updates

### Add to `events` table:
```sql
ALTER TABLE events ADD COLUMN watchFwVersion TEXT;
ALTER TABLE events ADD COLUMN alarmThresh REAL;
ALTER TABLE events ADD COLUMN alarmRatioThresh REAL;
ALTER TABLE events ADD COLUMN alarmFreqMin REAL;
ALTER TABLE events ADD COLUMN alarmFreqMax REAL;
ALTER TABLE events ADD COLUMN hrThreshMin INTEGER;
ALTER TABLE events ADD COLUMN hrThreshMax INTEGER;
ALTER TABLE events ADD COLUMN o2SatThreshMin INTEGER;
ALTER TABLE events ADD COLUMN o2SatAlarmActive INTEGER;
ALTER TABLE events ADD COLUMN o2SatAlarmStanding INTEGER;
ALTER TABLE events ADD COLUMN batteryPc INTEGER;
ALTER TABLE events ADD COLUMN seizureTimes TEXT;  -- JSON array: [start, end]
```

### Add to `datapoints` table:
```sql
ALTER TABLE datapoints ADD COLUMN specPower REAL;
ALTER TABLE datapoints ADD COLUMN roiPower REAL;
ALTER TABLE datapoints ADD COLUMN roiRatio REAL;
ALTER TABLE datapoints ADD COLUMN maxVal REAL;
ALTER TABLE datapoints ADD COLUMN maxFreq REAL;
```

## Data Loss Assessment

| Category | Risk Level | Fields Affected | Mitigation |
|----------|------------|-----------------|------------|
| Manual Curation | **CRITICAL** | seizureTimes | Must add to schema |
| Alarm Configuration | **HIGH** | alarmThresh, alarmRatioThresh, alarmFreq* | Add to schema or store in metadata JSON |
| HR/O2 Configuration | **HIGH** | hrThresh*, o2SatThresh*, o2SatAlarm* | Add to schema or store in metadata JSON |
| Device Metadata | **MEDIUM** | watchFwVersion, batteryPc | Add to schema |
| Analysis Results | **MEDIUM** | specPower, roiPower, roiRatio | Add to datapoints or recompute |
| Deprecated Fields | **LOW** | sdMode, maxVal, maxFreq | Store in metadata JSON or skip |

## Recommendations

1. **IMMEDIATE ACTION REQUIRED**: Add `seizureTimes` field to events table
   - This contains manual curation work that cannot be reconstructed
   - Format: JSON string "[start_seconds, end_seconds]"

2. **HIGH PRIORITY**: Add alarm and HR/O2 configuration fields
   - These document the algorithm settings used for each event
   - Important for reproducing analysis results

3. **MEDIUM PRIORITY**: Add device metadata fields
   - Useful for filtering and analysis
   - watchFwVersion helps track firmware-related issues
   - batteryPc can correlate with data quality

4. **CONSIDER**: Add datapoint analysis fields
   - specPower, roiPower, roiRatio are useful for analysis
   - Can be recomputed from rawData if needed
   - Add if space permits, otherwise document as "recomputable"

5. **METADATA JSON USAGE**: Use events.metadata field for:
   - Rarely-used fields
   - Deprecated fields (sdMode)
   - Future extensibility without schema changes

6. **SCHEMA VERSIONING**: Add schema_version tracking
   - Create `database_info` table with version number
   - Document schema changes across versions
   - Enable migration scripts for upgrades

## Implementation Plan

1. Update `init_database.py` CREATE TABLE statements
2. Add schema version table and tracking
3. Create migration script for existing databases
4. Update `osdb_sqlite.py` import/export to handle new fields
5. Add tests to verify no data loss in roundtrip (JSON → DB → JSON)
6. Document which fields go to database vs metadata JSON
