"""
Capital Bikeshare Station Predictability Analysis
Using Prophet with Weather Data

Input:
  - data/trips_2022.parquet, trips_2023.parquet, trips_2024.parquet, trips_2025.parquet
  - data/weather_cleaned.csv
  
Output:
  - results/predictability_ranking.csv
  - results/mape_distribution.pdf
  - results/prediction_plots/
  - results/summary_report.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from prophet import Prophet

# Set plot style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_trip_data():
    """Load trip data from all years"""
    print("1. Loading Trip Data")
    
    years = [2022, 2023, 2024, 2025]
    trips_by_year = {}
    
    for year in years:
        filepath = f'data/trips_{year}.parquet'
        try:
            df = pd.read_parquet(filepath)
            trips_by_year[year] = df
            print(f"{year}: {len(df):,} records")
        except FileNotFoundError:
            print(f"{year}: File not found - {filepath}")
    
    return trips_by_year

def load_weather_data():
    """Load cleaned weather data"""
    print("\n2. Loading Weather Data")
    
    weather = pd.read_csv('data/weather_cleaned.csv')
    weather['date'] = pd.to_datetime(weather['date'])
    
    print(f"Records: {len(weather)}")
    print(f"Date range: {weather['date'].min().date()} to {weather['date'].max().date()}")
    print(f"Columns: {list(weather.columns)}")
    
    return weather

def prepare_daily_station_data(trips_by_year):
    """Prepare daily departure counts by station"""
    print("\n3. Preparing Daily Station Data")
    
    all_daily = []
    
    for year, trips in trips_by_year.items():
        # Filter to valid stations
        valid_trips = trips[trips['start_station'].notna()].copy()
        valid_trips['date'] = valid_trips['start_time'].dt.date
        
        # Count departures by station and date
        daily = valid_trips.groupby(['start_station', 'date']).size().reset_index(name='departures')
        daily['date'] = pd.to_datetime(daily['date'])
        daily['year'] = year
        
        all_daily.append(daily)
        print(f"{year}: {len(daily):,} station-day records")
    
    combined = pd.concat(all_daily, ignore_index=True)
    print(f"\nTotal: {len(combined):,} station-day records")
    
    return combined

def find_common_stations(daily_data):
    """Find stations that exist in all years"""
    print("\n4. Finding Common Stations")
    
    # Get stations by year
    stations_by_year = {}
    for year in daily_data['year'].unique():
        stations = set(daily_data[daily_data['year'] == year]['start_station'].unique())
        stations_by_year[year] = stations
        print(f"{year}: {len(stations)} stations")
    
    # Find intersection
    common_stations = set.intersection(*stations_by_year.values())
    print(f"\nCommon stations (exist in all years): {len(common_stations)}")
    
    return list(common_stations)

def select_stations_for_analysis(daily_data, common_stations, min_daily_avg=10):
    """Select stations with sufficient data"""
    print("\n5. Selecting Stations for Analysis")
    
    # Filter to common stations
    daily_common = daily_data[daily_data['start_station'].isin(common_stations)]
    
    # Calculate daily average for each station
    station_stats = daily_common.groupby('start_station').agg({
        'departures': ['mean', 'sum', 'count']
    }).reset_index()
    station_stats.columns = ['station', 'daily_avg', 'total', 'days']
    
    # Filter by minimum daily average
    qualified = station_stats[station_stats['daily_avg'] >= min_daily_avg].copy()
    qualified = qualified.sort_values('daily_avg', ascending=False)
    
    print(f"Minimum daily average: {min_daily_avg}")
    print(f"Qualified stations: {len(qualified)}")
    
    return qualified

def create_prophet_dataframe(daily_data, weather_data, station_name):
    """Create Prophet-compatible dataframe for a station"""
    # Filter to this station
    station_data = daily_data[daily_data['start_station'] == station_name].copy()
    
    # Create date range covering all years
    date_range = pd.date_range(
        start=station_data['date'].min(),
        end=station_data['date'].max(),
        freq='D'
    )
    
    # Create complete dataframe
    df = pd.DataFrame({'ds': date_range})
    
    # Merge with station departures
    station_data = station_data.rename(columns={'date': 'ds', 'departures': 'y'})
    df = df.merge(station_data[['ds', 'y']], on='ds', how='left')
    df['y'] = df['y'].fillna(0)
    
    # Merge with weather data (make a copy to avoid modifying original)
    weather_copy = weather_data.copy()
    weather_copy = weather_copy.rename(columns={'date': 'ds'})
    df = df.merge(weather_copy, on='ds', how='left')
    
    return df

def fit_prophet_and_predict(train_df, test_df, weather_regressors):
    """
    Fit Prophet model and make predictions
    
    Parameters:
        train_df: Training data (2022-2024)
        test_df: Test data (2025)
        weather_regressors: List of weather columns to use as regressors
    
    Returns:
        forecast: Predictions
        metrics: Evaluation metrics
        model: Fitted model
    """
    try:
        # Check for NaN in regressors
        for reg in weather_regressors:
            if reg in train_df.columns:
                nan_count = train_df[reg].isna().sum()
                if nan_count > 0:
                    # Fill NaN with median
                    train_df[reg] = train_df[reg].fillna(train_df[reg].median())
            if reg in test_df.columns:
                nan_count = test_df[reg].isna().sum()
                if nan_count > 0:
                    test_df[reg] = test_df[reg].fillna(test_df[reg].median())
        
        # Initialize Prophet
        model = Prophet(
            yearly_seasonality=True,    # Capture yearly patterns
            weekly_seasonality=True,    # Capture weekly patterns
            daily_seasonality=False,    # Not needed for daily data
            seasonality_mode='multiplicative',  # Better for varying amplitude
            changepoint_prior_scale=0.05,  # Flexibility for trend changes
        )
        
        # Add weather regressors
        for regressor in weather_regressors:
            if regressor in train_df.columns:
                model.add_regressor(regressor, mode='multiplicative')

        model.fit(train_df)
        forecast = model.predict(test_df)
        forecast['yhat'] = np.maximum(forecast['yhat'], 0)
        
        # Calculate metrics
        actual = test_df['y'].values
        predicted = forecast['yhat'].values
        
        # MAPE
        mask = actual > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAE
        mae = np.mean(np.abs(actual - predicted))
        
        # Correlation
        if np.std(actual) > 0 and np.std(predicted) > 0:
            correlation = np.corrcoef(actual, predicted)[0, 1]
        else:
            correlation = np.nan
        
        metrics = {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation
        }
        
        return forecast, metrics, model
        
    except Exception as e:
        return None, {
            'mape': np.nan, 'rmse': np.nan, 'mae': np.nan, 
            'correlation': np.nan, 'error': str(e)
        }, None

def analyze_all_stations(daily_data, weather_data, qualified_stations, max_stations=None):
    """Run Prophet analysis for all qualified stations"""
    print("\n6. Running Prophet Predictions")
    
    stations_to_analyze = qualified_stations['station'].tolist()
    if max_stations:
        stations_to_analyze = stations_to_analyze[:max_stations]
    
    print(f"Analyzing {len(stations_to_analyze)} stations...")
    weather_regressors = [
        'temp_mean',         # Linear temperature effect
        'temp_squared',      # Non-linear temperature effect (inverted U-shape)
        'precipitation',     # Rain reduces ridership
    ]
    print(f"Weather regressors: {weather_regressors}")

    results = []
    errors = []
    
    for i, station in enumerate(stations_to_analyze):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Progress: {i + 1}/{len(stations_to_analyze)} - {station[:40]}...")
        
        try:
            # Create Prophet dataframe
            prophet_df = create_prophet_dataframe(daily_data, weather_data.copy(), station)
            
            # Split into train (2022-2024) and test (2025)
            train_df = prophet_df[prophet_df['ds'].dt.year <= 2024].copy()
            test_df = prophet_df[prophet_df['ds'].dt.year == 2025].copy()
            
            if len(train_df) < 365 or len(test_df) < 30:
                print(f"Skipping {station}: insufficient data (train={len(train_df)}, test={len(test_df)})")
                continue
            
            # Fit and predict
            forecast, metrics, model = fit_prophet_and_predict(
                train_df, test_df, weather_regressors
            )

            # Get station stats
            station_info = qualified_stations[qualified_stations['station'] == station].iloc[0]
            
            results.append({
                'station': station,
                'daily_avg': station_info['daily_avg'],
                'total_trips': station_info['total'],
                'mape': metrics['mape'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'correlation': metrics['correlation'],
                'forecast': forecast,
                'train_df': train_df,
                'test_df': test_df,
                'model': model
            })
        except Exception as e:
            errors.append({'station': station, 'error': str(e)})
            if len(errors) <= 3:  # Print first 3 errors
                print(f"  Error for {station}: {str(e)[:100]}")
    
    if errors:
        print(f"\nTotal errors: {len(errors)}")
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df['mape_rank'] = results_df['mape'].rank()
        results_df = results_df.sort_values('mape')
    
    print("\nAnalysis complete!")
    
    return results_df

def generate_summary_report(results_df, output_dir):
    """Generate analysis report"""
    print("\n7. Generating Report")
    
    # Filter out rows with NaN MAPE for ranking
    valid_results = results_df[results_df['mape'].notna()].copy()
    valid_results['mape_rank'] = valid_results['mape'].rank()
    valid_results = valid_results.sort_values('mape')

    report_lines = []
    report_lines.append("Capital Bikeshare Station Predictability Analysis Report")
    report_lines.append(f"  Number of stations analyzed: {len(results_df)}")
    report_lines.append(f"  Stations with valid results: {len(valid_results)}")
    report_lines.append(f"  Stations failed: {len(results_df) - len(valid_results)}")
    report_lines.append(f"  Model: Prophet with weather regressors")
    report_lines.append(f"  Seasonality: Yearly + Weekly (multiplicative)")
    report_lines.append("")
    
    report_lines.append("[WEATHER REGRESSORS USED]")
    report_lines.append("  - temp_mean: Daily mean temperature (°C)")
    report_lines.append("  - temp_squared: Temperature squared (captures non-linear effect)")
    report_lines.append("  - precipitation: Daily precipitation (mm)")
    report_lines.append("")

    report_lines.append("[MODEL RATIONALE]")
    report_lines.append("  The model uses only scientifically justified features:")
    report_lines.append("  - Prophet automatically captures weekly and yearly seasonality")
    report_lines.append("  - Temperature has an inverted U-shape effect on biking")
    report_lines.append("    (both too cold and too hot reduce ridership)")
    report_lines.append("  - Precipitation directly discourages biking")
    report_lines.append("  No event-specific features (holidays, festivals) are included")
    report_lines.append("  to maintain model generalizability and interpretability.")
    report_lines.append("")
    
    # MAPE statistics
    valid_mape = valid_results['mape']
    report_lines.append("[MAPE STATISTICS]")
    report_lines.append(f"  Mean MAPE: {valid_mape.mean():.2f}%")
    report_lines.append(f"  Median MAPE: {valid_mape.median():.2f}%")
    report_lines.append(f"  Min MAPE: {valid_mape.min():.2f}%")
    report_lines.append(f"  Max MAPE: {valid_mape.max():.2f}%")
    report_lines.append(f"  Std MAPE: {valid_mape.std():.2f}%")
    report_lines.append("")
    
    # Predictability classification
    excellent = (valid_mape < 20).sum()
    good = ((valid_mape >= 20) & (valid_mape < 30)).sum()
    moderate = ((valid_mape >= 30) & (valid_mape < 50)).sum()
    poor = (valid_mape >= 50).sum()
    
    report_lines.append("[PREDICTABILITY DISTRIBUTION]")
    report_lines.append(f"  Excellent (MAPE < 20%):     {excellent} stations ({excellent/len(valid_mape)*100:.1f}%)")
    report_lines.append(f"  Good (20% <= MAPE < 30%):   {good} stations ({good/len(valid_mape)*100:.1f}%)")
    report_lines.append(f"  Moderate (30% <= MAPE < 50%): {moderate} stations ({moderate/len(valid_mape)*100:.1f}%)")
    report_lines.append(f"  Poor (MAPE >= 50%):         {poor} stations ({poor/len(valid_mape)*100:.1f}%)")
    report_lines.append("")
    
    # Most predictable stations
    report_lines.append("[TOP 10 MOST PREDICTABLE STATIONS] (Lowest MAPE)")
    top10 = valid_results.head(10)
    for _, row in top10.iterrows():
        report_lines.append(f"  {int(row['mape_rank']):3d}. {row['station'][:40]:<40} MAPE={row['mape']:.2f}%")
    report_lines.append("")
    
    # Least predictable stations
    report_lines.append("[TOP 10 LEAST PREDICTABLE STATIONS] (Highest MAPE)")
    bottom10 = valid_results.tail(10).iloc[::-1]
    for _, row in bottom10.iterrows():
        report_lines.append(f"  {int(row['mape_rank']):3d}. {row['station'][:40]:<40} MAPE={row['mape']:.2f}%")
    report_lines.append("")
    
    # Correlation with usage
    report_lines.append("[PREDICTABILITY VS USAGE RELATIONSHIP]")
    correlation = valid_results['mape'].corr(valid_results['daily_avg'])
    report_lines.append(f"  Correlation between MAPE and daily average: {correlation:.3f}")
    if correlation < -0.3:
        report_lines.append("  Interpretation: High-frequency stations tend to be more predictable")
    elif correlation > 0.3:
        report_lines.append("  Interpretation: High-frequency stations tend to be less predictable")
    else:
        report_lines.append("  Interpretation: No strong relationship between usage and predictability")
    
    report_text = "\n".join(report_lines)
    
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {report_path}")
    
    return report_text


def plot_prediction_comparison(results_df, output_dir):
    """Plot prediction comparison charts"""
    print("\n8. Generating Prediction Plots")
    
    plot_dir = output_dir / 'prediction_plots'
    plot_dir.mkdir(exist_ok=True)
    
    # Filter to valid results
    valid_results = results_df[results_df['mape'].notna()].copy()
    valid_results = valid_results.sort_values('mape')
    valid_results['mape_rank'] = range(1, len(valid_results) + 1)

    best_station = valid_results.head(1)
    worst_station = valid_results.tail(1)
    stations_to_plot = pd.concat([best_station, worst_station])
    
    for _, row in stations_to_plot.iterrows():
        if row['forecast'] is None:
            continue

        fig, ax = plt.subplots(figsize=(12, 4))
        
        station_name = row['station']
        test_df = row['test_df']
        forecast = row['forecast']

        ax.plot(test_df['ds'], test_df['y'], 'b-', alpha=0.7, linewidth=1.2, label='Actual')
        ax.plot(forecast['ds'], forecast['yhat'], 'r-', alpha=0.7, linewidth=1.2, label='Predicted')
        
        # Add confidence interval
        ax.fill_between(
            forecast['ds'], 
            forecast['yhat_lower'], 
            forecast['yhat_upper'],
            alpha=0.2, color='red', label='95% CI'
        )
        
        ax.set_title(f'{station_name} | MAPE: {row["mape"]:.1f}%', fontsize=12)
        ax.set_xlabel('Date (2025)')
        ax.set_ylabel('Daily Departures')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in station_name)[:50]
        rank = int(row['mape_rank'])
        fig_path = plot_dir / f'rank_{rank:03d}_{safe_name}.pdf'
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(stations_to_plot)} prediction plots to: {plot_dir}")

def plot_mape_distribution(results_df, output_dir):
    """Plot MAPE distribution - histogram only"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter to valid results
    valid_results = results_df[results_df['mape'].notna()]
    valid_mape = valid_results['mape']
    
    if len(valid_mape) == 0:
        print("No valid MAPE values to plot")
        plt.close()
        return
    
    # Histogram
    ax.hist(valid_mape, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(valid_mape.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {valid_mape.median():.1f}%')
    ax.axvline(valid_mape.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {valid_mape.mean():.1f}%')
    ax.set_xlabel('MAPE (%)', fontsize=11)
    ax.set_ylabel('Number of Stations', fontsize=11)
    ax.set_title('Distribution of Prediction Error (MAPE)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = output_dir / 'mape_distribution.pdf'
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"MAPE distribution plot saved to: {fig_path}")


def plot_top_bottom_comparison(results_df, output_dir):
    """Plot comparison of most vs least predictable stations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter to valid results
    valid_results = results_df[results_df['mape'].notna()].sort_values('mape')
    
    # Top 10
    top10 = valid_results.head(10)
    axes[0].barh(range(len(top10)), top10['mape'].values, color='forestgreen', alpha=0.7)
    axes[0].set_yticks(range(len(top10)))
    axes[0].set_yticklabels([s[:30] + '...' if len(s) > 30 else s for s in top10['station']])
    axes[0].set_xlabel('MAPE (%)')
    axes[0].set_title('Top 10 Most Predictable Stations')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Bottom 10
    bottom10 = valid_results.tail(10).iloc[::-1]
    axes[1].barh(range(len(bottom10)), bottom10['mape'].values, color='crimson', alpha=0.7)
    axes[1].set_yticks(range(len(bottom10)))
    axes[1].set_yticklabels([s[:30] + '...' if len(s) > 30 else s for s in bottom10['station']])
    axes[1].set_xlabel('MAPE (%)')
    axes[1].set_title('Top 10 Least Predictable Stations')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig_path = output_dir / 'top_bottom_comparison.pdf'
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {fig_path}")


def plot_seasonality_components(results_df, output_dir):
    """Plot seasonality components for a sample station"""
    # Pick the most predictable station that has a model
    for _, row in results_df.iterrows():
        if row['model'] is not None:
            model = row['model']
            station_name = row['station']
            break
    else:
        print("No valid model found for seasonality plot")
        return
    
    fig = model.plot_components(row['forecast'])
    fig.suptitle(f'Prophet Components: {station_name}', fontsize=12)
    
    fig_path = output_dir / 'seasonality_components.pdf'
    fig.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Seasonality components plot saved to: {fig_path}")


def analyze_predictability_factors(results_df, output_dir):
    """Analyze what factors explain predictability differences"""
    print("\n9. Analyzing Predictability Factors")
    
    # Filter to valid results
    valid_results = results_df[results_df['mape'].notna()].copy()
    
    if len(valid_results) == 0:
        print("No valid results to analyze")
        return None
    
    analysis_lines = []
    analysis_lines.append("\n" + "=" * 70)
    analysis_lines.append("PREDICTABILITY ANALYSIS: Why are some stations more predictable?")
    analysis_lines.append("=" * 70)
    
    # Calculate variability metrics for each station
    station_variability = []
    
    for _, row in valid_results.iterrows():
        station = row['station']
        train_df = row['train_df']
        
        if train_df is None:
            continue
        
        y = train_df['y'].values
        
        # Coefficient of Variation (CV) - measures relative variability
        cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else np.nan
        
        # Weekend vs Weekday ratio
        train_df_copy = train_df.copy()
        train_df_copy['dow'] = train_df_copy['ds'].dt.dayofweek
        weekday_avg = train_df_copy[train_df_copy['dow'] < 5]['y'].mean()
        weekend_avg = train_df_copy[train_df_copy['dow'] >= 5]['y'].mean()
        weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else np.nan
        
        # Summer vs Winter ratio (seasonality strength)
        train_df_copy['month'] = train_df_copy['ds'].dt.month
        summer_avg = train_df_copy[train_df_copy['month'].isin([6, 7, 8])]['y'].mean()
        winter_avg = train_df_copy[train_df_copy['month'].isin([12, 1, 2])]['y'].mean()
        season_ratio = summer_avg / winter_avg if winter_avg > 0 else np.nan
        
        # Spike frequency (days with >3x mean)
        mean_y = np.mean(y)
        spike_days = np.sum(y > 3 * mean_y)
        spike_pct = spike_days / len(y) * 100
        
        station_variability.append({
            'station': station,
            'mape': row['mape'],
            'daily_avg': row['daily_avg'],
            'cv': cv,
            'weekend_ratio': weekend_ratio,
            'season_ratio': season_ratio,
            'spike_pct': spike_pct
        })
    
    var_df = pd.DataFrame(station_variability)
    
    # Correlation between MAPE and various factors
    analysis_lines.append("\n[CORRELATION WITH MAPE]")
    analysis_lines.append("(Positive = higher value → less predictable)")
    analysis_lines.append("")
    
    factors = {
        'cv': 'Coefficient of Variation (variability)',
        'weekend_ratio': 'Weekend/Weekday ratio',
        'season_ratio': 'Summer/Winter ratio',
        'spike_pct': 'Spike frequency (%)',
        'daily_avg': 'Daily average usage'
    }
    
    for factor, description in factors.items():
        corr = var_df['mape'].corr(var_df[factor])
        analysis_lines.append(f"  {description}: r = {corr:.3f}")
    
    # Compare top vs bottom stations
    analysis_lines.append("\n[TOP 10 vs BOTTOM 10 COMPARISON]")
    analysis_lines.append("")
    
    top10 = var_df.head(10)
    bottom10 = var_df.tail(10)
    
    comparison_metrics = ['daily_avg', 'cv', 'weekend_ratio', 'season_ratio', 'spike_pct']
    
    analysis_lines.append(f"{'Metric':<35} {'Most Predictable':<20} {'Least Predictable':<20}")
    analysis_lines.append("-" * 75)
    
    for metric in comparison_metrics:
        top_val = top10[metric].mean()
        bot_val = bottom10[metric].mean()
        
        metric_names = {
            'daily_avg': 'Avg Daily Departures',
            'cv': 'Coefficient of Variation',
            'weekend_ratio': 'Weekend/Weekday Ratio',
            'season_ratio': 'Summer/Winter Ratio',
            'spike_pct': 'Spike Days (%)'
        }
        
        analysis_lines.append(f"{metric_names[metric]:<35} {top_val:<20.2f} {bot_val:<20.2f}")

    # Station type inference
    analysis_lines.append("\n[STATION TYPE INFERENCE]")
    analysis_lines.append("\n  Most Predictable Stations (likely COMMUTER stations):")
    for _, row in top10.head(5).iterrows():
        analysis_lines.append(f"    - {row['station'][:45]}")
        analysis_lines.append(f"      CV={row['cv']:.2f}, Weekend/Weekday={row['weekend_ratio']:.2f}")
    
    analysis_lines.append("\n  Least Predictable Stations (likely TOURIST stations):")
    for _, row in bottom10.tail(5).iloc[::-1].iterrows():
        analysis_lines.append(f"    - {row['station'][:45]}")
        analysis_lines.append(f"      CV={row['cv']:.2f}, Spike%={row['spike_pct']:.1f}%")
    
    analysis_text = "\n".join(analysis_lines)
    
    # Append to summary report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(analysis_text)
    
    print(analysis_text)
    
    # Save variability analysis
    var_df.to_csv(output_dir / 'station_variability_analysis.csv', index=False)
    print(f"\nVariability analysis saved to: {output_dir / 'station_variability_analysis.csv'}")
    
    return var_df

def main():
    """Main function"""
    print("Capital Bikeshare Station Predictability Analysis")

    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # 1: Load trip data
    trips_by_year = load_trip_data()
    
    # 2: Load weather data
    weather_data = load_weather_data()
    
    # 3: Prepare daily station data
    daily_data = prepare_daily_station_data(trips_by_year)
    
    # 4: Find common stations
    common_stations = find_common_stations(daily_data)
    
    # 5: Select stations for analysis
    qualified_stations = select_stations_for_analysis(
        daily_data, common_stations, min_daily_avg=10
    )
    
    # 6: Run Prophet predictions
    results_df = analyze_all_stations(
        daily_data,
        weather_data,
        qualified_stations,
        max_stations=None
    )
    
    # 7: Generate report
    generate_summary_report(results_df, output_dir)
    
    # 8: Generate plots
    plot_mape_distribution(results_df, output_dir)
    plot_prediction_comparison(results_df, output_dir)
    plot_top_bottom_comparison(results_df, output_dir)

    # 9: Analyze predictability factors
    analyze_predictability_factors(results_df, output_dir)
    
    # Save ranking CSV
    results_export = results_df.drop(columns=['forecast', 'train_df', 'test_df', 'model'])
    results_export.to_csv(output_dir / 'predictability_ranking.csv', index=False, encoding='utf-8-sig')
    print(f"\nComplete ranking saved to: {output_dir / 'predictability_ranking.csv'}")
    print("\nAnalysis Complete!")
    
    return results_df

if __name__ == "__main__":
    results = main()
