# PJM Zone Price Forecast + Spike Alerts (Day-Ahead)

Forecast next-day hourly zonal LMPs and flag high-risk (spike) hours across PJM using scikit-learn. End-to-end: tidy the raw EIA PJM datasets

**Outline:**
  - Data wrangling: Convert wide Day-Ahead LMP tables (one column per zone) to long
  - Features & targets: Calendar features + lag/rolling stats, and two labels:
      - Regression: LMP 24 hours ahead
      - Classification: 1 if next-day LMP > 0.9 percentile of the zone
  - Modeling: scikit-learn Pipelines with HistGradientBoosting + TimeSeriesSplit CV
  - Outputs: Saved models and simple reports/CSV with metrics and feature importance


**Data courtesy of EIA (PJM wholesale market datasets)**

**Author:**
  - Developed by Behnam Jabbari Marand, Ph.D. Student, NC State University
  - Focus: Optimization, integer programming, and power systems applications.
