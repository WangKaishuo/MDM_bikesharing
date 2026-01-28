# Capital Bikeshare Station Predictability Analysis

## Research Question

**Which stations are most and least predictable, and why?**

## Method

| Item | Description |
|------|-------------|
| Model | Prophet (Facebook) with weather regressors |
| Training | 2022-2024 (3 years) |
| Testing | 2025 (1 year) |
| Stations | 278 (present in all 4 years, daily avg ≥ 10) |

### Weather Regressors

| Variable | Rationale |
|----------|-----------|
| temp_mean | Temperature affects biking willingness |
| temp_squared | Non-linear effect (too cold/hot reduces ridership) |
| precipitation | Rain discourages biking |

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| Median MAPE | 40.59% |
| Mean MAPE | 45.21% |
| Best | 18.48% |
| Worst | 137.97% |

### Predictability Distribution

| Category | Stations | Percentage |
|----------|----------|------------|
| Excellent (< 20%) | 1 | 0.4% |
| Good (20-30%) | 65 | 23.4% |
| Moderate (30-50%) | 141 | 50.7% |
| Poor (≥ 50%) | 71 | 25.5% |

### Most vs Least Predictable

| Most Predictable | MAPE | Least Predictable | MAPE |
|------------------|------|-------------------|------|
| 14th & Belmont St NW | 18.5% | National Arboretum | 138.0% |
| Georgia & New Hampshire Ave NW | 20.5% | Gravelly Point | 131.2% |
| Columbia Rd & Belmont St NW | 21.7% | 34th & Water St NW | 128.6% |

## Key Findings

### Strongest Predictor of Unpredictability

**Coefficient of Variation (CV)**: r = 0.638

| Station Type | CV | Weekend/Weekday | Predictability |
|--------------|-----|-----------------|----------------|
| Commuter | ~0.4 | ~1.0 | High |
| Tourist | ~0.8 | ~1.7 | Low |

### Why Some Stations Are Unpredictable

1. **High variability** - Irregular usage patterns
2. **Event-driven demand** - Spikes from festivals, tours (e.g., Cherry Blossom)
3. **Strong seasonality** - Large summer/winter difference

### Conclusion

1. **~75% of stations have MAPE < 50%** - Reasonable predictability
2. **Commuter stations are predictable** - Regular patterns, low CV
3. **Tourist stations are inherently unpredictable** - Event-driven demand
4. **CV is the best predictor** of station predictability

