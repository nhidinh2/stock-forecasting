# Crypto Forecasting: Time Series Analysis

A comprehensive time series analysis project for cryptocurrency price forecasting, featuring both **R** (RMarkdown) and **Python** implementations. This project demonstrates advanced econometric techniques including ARIMAX modeling and GARCH volatility forecasting for Bitcoin price prediction.

For the most updated R-based report (including narrative, methodology, and results), see the published document on RPubs: [bitcoin-predict by mee](https://rpubs.com/eloise21/1387325).

## Overview

This project performs a complete time series analysis workflow on Bitcoin (BTC) price data, from exploratory data analysis through advanced forecasting models. The analysis includes:

- **Data exploration** and visualization
- **Stationarity testing** (ADF, KPSS)
- **Mean modeling** with OLS regression and ARIMAX
- **Volatility modeling** with eGARCH
- **5-day ahead forecasting** with confidence intervals

## Key Features

- **Dual Implementation**: Available in both R (RMarkdown) and Python
- **Comprehensive Analysis**: Full pipeline from data cleaning to forecasting
- **Advanced Models**: 
  - Seasonal ARIMAX for mean dynamics
  - eGARCH(1,1) with skewed Student-t distribution for volatility
- **Visualizations**: Rich plots for diagnostics and forecasts
- **Multiple Cryptocurrencies**: Dataset includes 50+ cryptocurrencies in the `archive/` folder

## Project Structure

```
stock-forecasting/
├── archive/                    # Historical cryptocurrency data (CSV files)
│   ├── bitcoin.csv            # Bitcoin OHLCV data (primary dataset)
│   ├── ethereum.csv
│   ├── solana.csv
│   └── ...                    # 50+ other cryptocurrencies
├── time_series_analysis.Rmd   # R Markdown analysis 
├── time_series_analysis.html  # Rendered HTML report
├── time_series_analysis.py    # Python implementation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### Python Version

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis**
   ```bash
   python time_series_analysis.py
   ```

   The script will:
   - Load and clean Bitcoin data from `archive/bitcoin.csv`
   - Perform exploratory data analysis
   - Test for stationarity
   - Fit OLS and ARIMAX models
   - Model volatility with eGARCH
   - Generate 5-day ahead forecasts
   - Display plots and summary statistics

### R Version

1. **Install Required R Packages**
   ```r
   install.packages(c("dplyr", "ggplot2", "gridExtra", "forecast", 
                     "lmtest", "tseries", "zoo", "lubridate", 
                     "knitr", "rugarch", "FinTS"))
   ```

2. **Render the RMarkdown**
   ```r
   rmarkdown::render("time_series_analysis.Rmd")
   ```

   Or open `time_series_analysis.Rmd` in RStudio and click "Knit".

## Methodology

### 1. Data Preparation
- Load OHLCV (Open, High, Low, Close, Volume) data
- Clean and validate dataset
- Handle missing values and duplicates

### 2. Exploratory Data Analysis
- Visualize price and volume trends
- Calculate summary statistics
- Identify trading patterns

### 3. Stationarity Analysis
- Transform prices to log returns: $r_t = \log(C_t) - \log(C_{t-1})$
- Test stationarity using:
  - **ADF Test** (Augmented Dickey-Fuller)
  - **KPSS Test** (Kwiatkowski-Phillips-Schmidt-Shin)
- Visual diagnostics (ACF, rolling statistics)

### 4. Mean Modeling

#### Model 1: OLS Regression
$$r_t = \beta_0 + \beta_1 r_{t-1} + \beta_2 \Delta\log(Vol_t) + \beta_3 \log(H_t/L_t) + \varepsilon_t$$

#### Model 2: Seasonal ARIMAX
- Captures autocorrelation in residuals
- Includes weekly seasonality (period = 7 days)
- External regressors: lagged returns, volume changes, intraday range

### 5. Volatility Modeling
- **ARCH Effects Testing**: Detect volatility clustering
- **eGARCH(1,1) Model**: 
  - Captures asymmetric volatility (leverage effect)
  - Uses skewed Student-t distribution for heavy tails

### 6. Forecasting
- Combine mean (ARIMAX) and volatility (eGARCH) forecasts
- Generate 5-day ahead predictions
- Provide 95% confidence intervals

## Dependencies

### Python
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `statsmodels` - Statistical models
- `pmdarima` - ARIMA model selection
- `arch` - GARCH models

### R
- `dplyr`, `ggplot2`, `gridExtra` - Data manipulation and visualization
- `forecast` - Time series forecasting
- `tseries` - Time series analysis
- `rugarch` - GARCH models
- `lmtest`, `FinTS` - Diagnostic tests

## Data Format

The CSV files in `archive/` follow this structure:

```csv
Date,Close,High,Low,Open,Volume
2014-09-17,457.33,468.17,452.42,465.86,21056800
2014-09-18,424.44,456.86,413.10,456.86,34483200
...
```

- **Date**: Trading date (YYYY-MM-DD)
- **Close**: Closing price (USD)
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Open**: Opening price
- **Volume**: Trading volume

## Key Findings

- Bitcoin price levels are **non-stationary** (trending mean, changing variance)
- **Log returns are stationary** (confirmed by ADF/KPSS tests)
- Strong evidence of **volatility clustering** (ARCH effects)
- **Seasonal ARIMAX** outperforms non-seasonal model (lower BIC)
- **eGARCH** captures asymmetric volatility response (leverage effect)

## Output

The analysis produces:

1. **Summary Statistics**: Mean, median, standard deviation, min/max prices
2. **Diagnostic Plots**: 
   - Price and volume trends
   - ACF/PACF plots
   - Stationarity diagnostics
   - Residual analysis
3. **Model Results**: 
   - OLS regression coefficients
   - ARIMAX model parameters
   - GARCH volatility estimates
4. **Forecasts**: 
   - 5-day ahead return predictions
   - Volatility forecasts
   - 95% confidence intervals

## Customization

To analyze a different cryptocurrency:

**Python:**
```python
# Edit line 551 in time_series_analysis.py
csv_file = "archive/ethereum.csv"  # Change to desired cryptocurrency
```

**R:**
```r
# Edit line 128 in time_series_analysis.Rmd
csv_file <- "archive/ethereum.csv"
```

## References

- **ARIMA Models**: Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*
- **GARCH Models**: Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation
- **eGARCH**: Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach
