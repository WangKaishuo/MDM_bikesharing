# Capital Bikeshare Data Analysis

## Project Structure

```
MDM_bikesharing/
├── data_cleaning.py            # step1
├── data/                   
│   ├── raw/                    # Store raw data
│   ├── data/trips_20xx.parquet      # Trips data in 20xx
│   └── data/stations_20xx.parquet   # Stations data in 20xx
├── docs/                       # Store Documents
└── README.md
```

---

# 1. Data Cleaning Script

### This script cleans Capital Bikeshare raw trip data and outputs structured Parquet files for subsequent analysis.

### Main functions:
- Merge 12 monthly CSV files
- Parse datetime fields
- Calculate trip distance (only when station info is complete)
- Optimize data types to reduce memory usage
- Generate trip & station list

## Input Format

Raw CSV files with the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| ride_id | Trip ID | 63DF1EFB0216674C |
| rideable_type | Bike type | electric_bike |
| started_at | Start time | 2025/1/18 16:14 |
| ended_at | End time | 2025/1/18 16:20 |
| start_station_name | Start station name | 8th & F St NE |
| start_station_id | Start station ID | 31631 |
| end_station_name | End station name | 1st & K St NE |
| end_station_id | End station ID | 31662 |
| start_lat | Start latitude | 38.897274 |
| start_lng | Start longitude | -76.994749 |
| end_lat | End latitude | 38.902386 |
| end_lng | End longitude | -77.005649 |
| member_casual | User type | member |

**Note**: `start_station_name`, `start_station_id`, `end_station_name`, `end_station_id` may be empty.

## Output Files

### 1. data/trips.parquet

| Field | Type | Description |
|-------|------|-------------|
| id | string | Trip ID |
| start_time | datetime | Start time |
| end_time | datetime | End time |
| start_station | category | Start station name (nullable) |
| start_station_id | category | Start station ID (nullable) |
| end_station | category | End station name (nullable) |
| end_station_id | category | End station ID (nullable) |
| user_type | category | User type: member / casual |
| distance_m | Int32 | Trip distance in meters (null if station info incomplete) |

### 2. data/stations.parquet

| Field | Type | Description |
|-------|------|-------------|
| name | category | Station name |
| id | category | Station ID |
| lat | float32 | Latitude |
| lng | float32 | Longitude |

### Data Summary

**Data in 2024**

| Metric | Value |
|--------|-------|
| Total trip records | 6,662,647 |
| Complete station info | 4,541,599 (68.2%) |
| Incomplete station info | 2,121,048 (31.8%) |
| Number of stations | 848 |

**Data in 2025**

| Metric | Value |
|--------|-------|
| Total trip records | 6,114,323 |
| Complete station info | 4,410,121 (72.1%) |
| Incomplete station info | 1,704,202 (27.9%) |
| Number of stations | 820 |

### Loading Data

```python
import pandas as pd

trips = pd.read_parquet('data/trips.parquet')
stations = pd.read_parquet('data/stations.parquet')
```

---
