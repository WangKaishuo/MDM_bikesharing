"""
Capital Bikeshare Data Cleaning Script

input: CSV files（in data/raw/）
output:
  - data/trips.parquet: Riding Records
  - data/stations.parquet: Stations List
"""

import pandas as pd
import math
from pathlib import Path

def distance_calculation(row):
    """
    Calculate the distance according to the starting and ending coordinates (return meters),
    Return None when the station information is incomplete
    """
    if not (row['has_start_station'] and row['has_end_station']):
        return None

    lat1, lng1 = row['start_lat'], row['start_lng']
    lat2, lng2 = row['end_lat'], row['end_lng']

    radius = 6371  # earth radius (km)
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    lat_diff = lat2 - lat1
    lon_diff = lng2 - lng1
    a = math.sin(lat_diff/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(lon_diff/2)**2
    b = 2 * math.asin(math.sqrt(a))
    
    return round(radius * b, 3)

def main():
    print("Data cleaning starts...")
    
    # Read and Merge data
    csv_files = sorted(Path('data/raw').glob('*.csv'))
    print(f"\n{len(csv_files)} CSV files found")

    all_data = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nNumber of original riding records: {len(df):,}")

    # Resolve time format (example: 2025/1/18 16:14)
    df['started_at'] = pd.to_datetime(df['started_at'], format='mixed')
    df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed')

    # Mark whether the station information is complete
    df['has_start_station'] = df['start_station_name'].notna() & (df['start_station_name'] != '')
    df['has_end_station'] = df['end_station_name'].notna() & (df['end_station_name'] != '')

    total = len(df)
    complete = (df['has_start_station'] & df['has_end_station']).sum()
    print(f"Complete station information: {complete:,} ({complete / total * 100:.1f}%)")
    print(f"Incomplete station information: {total - complete:,} ({(total - complete) / total * 100:.1f}%)")

    # Calculate distance
    df['distance_km'] = df.apply(distance_calculation, axis=1)

    # Create DataFrame
    trips = df[[
        'ride_id',
        'started_at',
        'ended_at', 
        'start_station_name',
        'start_station_id',
        'end_station_name',
        'end_station_id',
        'distance_km',
        'member_casual'
    ]].copy()
    
    # Rename
    trips.columns = [
        'id',
        'start_time',
        'end_time',
        'start_station',
        'start_station_id',
        'end_station',
        'end_station_id',
        'distance_km',
        'user_type'
    ]

    # Build stations list
    start_stations = df[df['start_station_name'].notna() & (df['start_station_name'] != '')][
        ['start_station_name', 'start_station_id', 'start_lat', 'start_lng']
    ].drop_duplicates(subset=['start_station_name']).rename(columns={
        'start_station_name': 'name',
        'start_station_id': 'id', 
        'start_lat': 'lat',
        'start_lng': 'lng'
    })

    end_stations = df[df['end_station_name'].notna() & (df['end_station_name'] != '')][
        ['end_station_name', 'end_station_id', 'end_lat', 'end_lng']
    ].drop_duplicates(subset=['end_station_name']).rename(columns={
        'end_station_name': 'name',
        'end_station_id': 'id',
        'end_lat': 'lat', 
        'end_lng': 'lng'
    })

    stations = pd.concat([start_stations, end_stations]).drop_duplicates(
        subset=['name']
    ).sort_values('name').reset_index(drop=True)

    # Optimize data types to reduce memory consumption
    trips['user_type'] = trips['user_type'].astype('category')
    trips['start_station'] = trips['start_station'].astype('category')
    trips['end_station'] = trips['end_station'].astype('category')
    trips['start_station_id'] = trips['start_station_id'].astype('category')
    trips['end_station_id'] = trips['end_station_id'].astype('category')

    trips['distance_m'] = (trips['distance_km'] * 1000).round().astype('Int32') # Convert to integer type
    trips = trips.drop(columns=['distance_km'])

    stations['name'] = stations['name'].astype('category')
    stations['id'] = stations['id'].astype('category')
    stations['lat'] = stations['lat'].astype('float32')
    stations['lng'] = stations['lng'].astype('float32')

    # Export data
    trips_path = 'data/trips.parquet'
    stations_path = 'data/stations.parquet'

    trips.to_parquet(trips_path, index=False)
    stations.to_parquet(stations_path, index=False)

    print("\nData cleaning completed!")
    print(f"\nOutput: data/trips.json, data/stations.json")
    print(f"\nNumber of riding records: {len(trips):,}")
    print(f"\nNumber of stations: {len(stations)}")

if __name__ == '__main__':
    main()
