"""
Clean and Prepare Weather Data for Prophet Model

Input: data/weather_2022_2025.csv
Output: data/weather_cleaned.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_weather_data(filepath):
    """Load weather data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def handle_missing_values(df):
    """Handle missing values in weather data"""
    print("\n" + "=" * 60)
    print("Handling Missing Values")
    print("=" * 60)
    
    # Check missing values
    missing_before = df.isnull().sum()
    print("\nMissing values before cleaning:")
    for col, count in missing_before.items():
        if count > 0:
            print(f"{col}: {count}")
    
    if missing_before.sum() == 0:
        print("No missing values found!")
        return df
    
    # 1. Forward fill for temperature (yesterday's temp is good estimate)
    temp_cols = ['temp_mean', 'temp_max', 'temp_min', 'feels_like']
    for col in temp_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # 2. Fill precipitation with 0 (missing usually means no rain)
    precip_cols = ['precipitation', 'rain', 'snowfall']
    for col in precip_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 3. Fill wind with median
    if 'wind_speed_max' in df.columns:
        df['wind_speed_max'] = df['wind_speed_max'].fillna(df['wind_speed_max'].median())
    
    # Check remaining missing values
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print("\nMissing values after cleaning:")
        for col, count in missing_after.items():
            if count > 0:
                print(f"  {col}: {count}")
    else:
        print("\nAll missing values handled!")
    
    return df


def create_derived_features(df):
    """Create derived features for better prediction"""
    print("\n" + "=" * 60)
    print("Creating Derived Features")
    print("=" * 60)
    
    # Temperature squared (for non-linear effect)
    # Optimal biking temp is around 20-25°C
    # Both cold and hot reduce ridership - this captures the inverted U-shape
    df['temp_squared'] = df['temp_mean'] ** 2
    print("Created: temp_squared (for non-linear temperature effect)")

    return df


def save_cleaned_data(df, output_path):
    """Save cleaned weather data"""
    # Select only the columns we actually need for the model
    output_columns = [
        'date',
        'temp_mean',         # Daily mean temperature
        'precipitation',     # Daily precipitation
        'temp_squared',      # Derived: for non-linear temperature effect
        'temp_max',          # For data exploration
        'temp_min',          # For data exploration
    ]

    output_columns = [col for col in output_columns if col in df.columns]
    
    df_output = df[output_columns].copy()
    df_output.to_csv(output_path, index=False)
    
    print(f"Saved to: {output_path}")
    print(f"Columns: {len(output_columns)}")
    print(f"Records: {len(df_output)}")
    
    print("\nColumns saved:")
    for col in output_columns:
        used = "(used in model)" if col in ['temp_mean', 'precipitation', 'temp_squared'] else "(reference only)"
        print(f"  - {col} {used}")
    
    return df_output


def main():
    print("Weather Data Cleaning...")

    df = load_weather_data('data/weather_2022_2025.csv')
    # Clean and prepare
    df = handle_missing_values(df)
    df = create_derived_features(df)

    df_cleaned = save_cleaned_data(df, 'data/weather_cleaned.csv')
    print("Data Cleaning Complete!")

    return df_cleaned


if __name__ == "__main__":
    df = main()
