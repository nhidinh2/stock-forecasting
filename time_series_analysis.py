"""
Time Series Analysis: Bitcoin Price Forecasting (Python Version)
----------------------------------------------------------------

This script mirrors the workflow in `time_series_analysis.Rmd` using Python.
It performs:

- Data loading and cleaning for Bitcoin daily OHLCV data
- Exploratory data analysis (price and volume trends, summary statistics)
- Stationarity analysis (log returns, ADF and KPSS tests, ACF)
- Mean modeling with:
  - OLS regression on log returns
  - ARIMAX / seasonal ARIMAX with exogenous regressors
- Volatility modeling with an eGARCH(1,1) model (Student-t innovations)
- 5-day ahead forecasting of returns and volatility with confidence intervals

Requirements (install with pip, e.g. `pip install -r requirements.txt`):
    pandas
    numpy
    matplotlib
    seaborn
    statsmodels
    pmdarima
    arch
"""

import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch import arch_model
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm


warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


# ---------------------------------------------------------------------------
# 2. Preliminary Analysis
# ---------------------------------------------------------------------------

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean Bitcoin OHLCV data from CSV.

    Steps mirror the R function `load_and_clean_data()`:
    - Read CSV
    - Drop second row if it contains currency metadata (BTC-USD / USD)
    - Parse Date
    - Convert numeric columns
    - Sort by Date (ascending)
    - Remove duplicate dates
    - Drop rows with missing Date / Close / Volume
    """
    df = pd.read_csv(file_path)

    # Remove second row if it contains currency metadata in the second column
    if df.shape[0] > 1:
        second_row_val = str(df.iloc[0, 1])
        if ("BTC-USD" in second_row_val) or ("USD" in second_row_val.upper()):
            df = df.iloc[1:].reset_index(drop=True)

    # Parse Date column
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")

    # Convert numeric columns
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by date (ascending)
    df = df.sort_values("Date")

    # Remove duplicate dates
    df = df.drop_duplicates(subset="Date", keep="first")

    # Remove rows with missing key values
    df = df.dropna(subset=["Date", "Close", "Volume"])

    df = df.reset_index(drop=True)
    return df


def check_trading_days(df: pd.DataFrame) -> None:
    """
    Check basic trading day information (weekday distribution).

    This mirrors the R `check_trading_days()` helper.
    """
    weekdays = df["Date"].dt.weekday  # Monday=0, Sunday=6
    has_weekends = weekdays.isin([5, 6]).any()
    total_days = len(df)
    weekend_count = weekdays.isin([5, 6]).sum()

    print("Trading Days Info:")
    print(f"- Total observations: {total_days}")
    print(f"- Contains weekend data (Sat/Sun): {has_weekends}")
    print(f"- Weekend count: {weekend_count}")


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for Close price and Volume.
    """
    stats = pd.DataFrame(
        {
            "Mean_Price": [df["Close"].mean()],
            "Median_Price": [df["Close"].median()],
            "SD_Price": [df["Close"].std()],
            "Min_Price": [df["Close"].min()],
            "Max_Price": [df["Close"].max()],
            "Mean_Volume": [df["Volume"].mean()],
        }
    )
    return stats


def plot_price_and_volume(df: pd.DataFrame) -> None:
    """
    Plot Close price and Volume over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Close price
    axes[0].plot(df["Date"], df["Close"], color="steelblue", linewidth=0.8)
    axes[0].set_title("Bitcoin Close Price Over Time", fontweight="bold")
    axes[0].set_ylabel("Close Price (USD)")

    # Volume
    axes[1].plot(df["Date"], df["Volume"], color="darkgreen", linewidth=0.8)
    axes[1].set_title("Bitcoin Trading Volume Over Time", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Volume")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2.3 Stationarity Analysis
# ---------------------------------------------------------------------------

def prepare_stationarity_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed close and volume, log returns, and rolling stats
    to the dataframe, mirroring the R code.
    """
    df = df.copy()
    df["log_Close"] = np.log(df["Close"])
    df["log_Volume"] = np.log(df["Volume"])

    # Log returns: r_t = log(C_t) - log(C_{t-1})
    df["returns"] = df["log_Close"].diff()

    # Rolling window
    window = min(60, max(2, df.shape[0] // 10))
    df["Close_rolling_mean"] = df["Close"].rolling(window).mean()
    df["Close_rolling_sd"] = df["Close"].rolling(window).std()
    return df


def plot_stationarity_diagnostics(df: pd.DataFrame) -> None:
    """
    Plot price with rolling mean ± sd and ACF of close prices.
    """
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Plot 1: Close price with rolling stats
    axes[0].plot(df["Date"], df["Close"], color="steelblue", linewidth=0.7, label="Close")
    axes[0].plot(
        df["Date"],
        df["Close_rolling_mean"],
        color="red",
        linewidth=1.2,
        linestyle="--",
        label="Rolling Mean",
    )
    upper = df["Close_rolling_mean"] + df["Close_rolling_sd"]
    lower = df["Close_rolling_mean"] - df["Close_rolling_sd"]
    axes[0].fill_between(
        df["Date"],
        lower,
        upper,
        color="red",
        alpha=0.2,
        label="Rolling Mean ± SD",
    )
    axes[0].set_title("Close Price with Rolling Mean ± SD (approx. 60-day window)", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Close Price")
    axes[0].legend()

    # Plot 2: ACF of Close Prices
    plot_acf(df["Close"].dropna(), ax=axes[1], lags=40, zero=False)
    axes[1].set_title("ACF of Close Price", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_returns(df: pd.DataFrame) -> None:
    """
    Plot log returns over time.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["returns"], color="steelblue", linewidth=0.6)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Log Returns: r(t) = log(C_t) - log(C_{t-1})", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    plt.tight_layout()
    plt.show()


def stationarity_tests(returns: pd.Series) -> None:
    """
    Run ADF (H0: unit root) and KPSS (H0: stationarity) tests on returns.
    """
    returns_clean = returns.dropna()
    print("ADF Test on log returns (H0: unit root / non-stationary):")
    adf_stat, adf_p, _, _, crit_vals, _ = adfuller(returns_clean, autolag="AIC")
    print(f"  Test statistic: {adf_stat:.4f}, p-value: {adf_p:.4g}")
    print(f"  Critical values: {crit_vals}")

    print("\nKPSS Test on log returns (H0: trend-stationary):")
    kpss_stat, kpss_p, _, kpss_crit = kpss(returns_clean, regression="ct", nlags="auto")
    print(f"  Test statistic: {kpss_stat:.4f}, p-value: {kpss_p:.4g}")
    print(f"  Critical values: {kpss_crit}")


# ---------------------------------------------------------------------------
# 3. Model Creation
# ---------------------------------------------------------------------------

def prepare_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create regression dataset as in the R `prepare_regression_data()` function:
    - Dependent variable: log_returns
    - Regressors:
        lag_return
        intraday_range = log(High / Low)
        log_Volume
        delta_log_Volume
    """
    df = df.copy()
    df["log_returns"] = np.log(df["Close"]).diff()
    df["lag_return"] = df["log_returns"].shift(1)
    df["intraday_range"] = np.log(df["High"] / df["Low"])
    df["log_Volume"] = np.log(df["Volume"])
    df["delta_log_Volume"] = df["log_Volume"].diff()

    df = df.dropna(subset=["log_returns", "lag_return", "delta_log_Volume", "intraday_range"])
    return df


def fit_ols_model(df_reg: pd.DataFrame):
    """
    Fit OLS regression:
        r_t = beta0 + beta1 * r_{t-1} + beta2 * Δlog(Vol_t) + beta3 * log(H_t/L_t) + e_t
    """
    y = df_reg["log_returns"]
    X = df_reg[["lag_return", "delta_log_Volume", "intraday_range"]]
    X = add_constant(X)
    model = OLS(y, X).fit()
    print(model.summary())
    return model


def ols_residual_diagnostics(residuals: pd.Series) -> None:
    """
    Residual diagnostics similar to the R code:
    - ACF and PACF of residuals
    - Ljung-Box test
    """
    res = residuals.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(res, ax=axes[0], lags=40, zero=False)
    axes[0].set_title("ACF of OLS Residuals", fontweight="bold")
    plot_pacf(res, ax=axes[1], lags=40, zero=False, method="ywunbiased")
    axes[1].set_title("PACF of OLS Residuals", fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("Ljung-Box test for residual autocorrelation (lag=10):")
    lb = acorr_ljungbox(res, lags=[10], return_df=True)
    print(lb)


def fit_arimax_models(df_reg: pd.DataFrame):
    """
    Fit non-seasonal and seasonal ARIMAX models using pmdarima.auto_arima,
    analogous to the R `auto.arima` calls (with and without weekly seasonality).
    """
    y = df_reg["log_returns"].values
    xreg = df_reg[["lag_return", "delta_log_Volume", "intraday_range"]].values

    print("Fitting non-seasonal ARIMAX model...")
    fit_ns = pm.auto_arima(
        y,
        X=xreg,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    print("Fitting seasonal ARIMAX model with weekly seasonality (m=7)...")
    fit_s = pm.auto_arima(
        y,
        X=xreg,
        seasonal=True,
        m=7,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )

    comparison = pd.DataFrame(
        {
            "Model": ["Non-Seasonal ARIMAX", "Seasonal ARIMAX (s=7)"],
            "AIC": [fit_ns.aic(), fit_s.aic()],
            "BIC": [fit_ns.bic(), fit_s.bic()],
        }
    )
    print("\nModel selection criteria:")
    print(comparison.to_string(index=False))

    # Choose the seasonal model to mirror the R analysis
    best = fit_s

    print("\nLjung-Box test on residuals of selected seasonal ARIMAX model:")
    resid = pd.Series(best.resid())
    lb = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
    print(lb)

    return fit_ns, fit_s


# ---------------------------------------------------------------------------
# 3.4 Volatility Modeling (eGARCH)
# ---------------------------------------------------------------------------

def volatility_diagnostics_and_garch(residuals: pd.Series):
    """
    Test for ARCH effects and fit an eGARCH(1,1) model with Student-t innovations.

    This approximates the eGARCH modeling from the R script using the `arch` package.
    """
    res = residuals.dropna()

    # ACF of squared residuals
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(res ** 2, ax=ax, lags=40, zero=False)
    ax.set_title("ACF of Squared Residuals", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # ARCH LM-type tests
    print("ARCH LM test for up to 7, 14, 21 lags (approximate to R ArchTest):")
    for lags in [7, 14, 21]:
        stat, pvalue, _, _ = het_arch(res, nlags=lags)
        print(f"  Lags={lags}: LM stat={stat:.4f}, p-value={pvalue:.4g}")

    # Fit eGARCH(1,1) with Student-t distribution
    print("\nFitting eGARCH(1,1) with Student-t innovations...")
    egarch = arch_model(
        res,
        vol="EGARCH",
        p=1,
        o=1,  # asymmetric term
        q=1,
        dist="t",
        mean="Zero",
    )
    egarch_res = egarch.fit(update_freq=20, disp="off")
    print(egarch_res.summary())

    return egarch_res


# ---------------------------------------------------------------------------
# 4. Forecasting
# ---------------------------------------------------------------------------

def combined_forecast(
    fit_s: pm.ARIMA,
    egarch_res,
    df_reg: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Generate 5-day ahead forecasts combining:
    - Mean forecasts from seasonal ARIMAX (fit_s)
    - Volatility forecasts from eGARCH(1,1) (egarch_res)
    """
    h = horizon
    last_row = df_reg.iloc[-1]
    xreg_future = np.tile(
        last_row[["lag_return", "delta_log_Volume", "intraday_range"]].values,
        (h, 1),
    )

    # Mean forecasts from ARIMAX
    mu_hat, _ = fit_s.predict(n_periods=h, X=xreg_future, return_conf_int=True)
    mu_hat = np.asarray(mu_hat).ravel()

    # Volatility forecasts from eGARCH
    garch_forecast = egarch_res.forecast(horizon=h)
    sigma_hat = np.sqrt(garch_forecast.variance.values[-1, :])

    last_date = df_reg["Date"].max()
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, h + 1)]

    lower_95 = mu_hat - 1.96 * sigma_hat
    upper_95 = mu_hat + 1.96 * sigma_hat

    forecast_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecasted_Return": mu_hat,
            "Forecasted_Volatility": sigma_hat,
            "Lower_95": lower_95,
            "Upper_95": upper_95,
        }
    )

    print("\nForecast results (returns and volatility with 95% intervals):")
    print(forecast_df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    return forecast_df


def plot_forecasts(forecast_df: pd.DataFrame) -> None:
    """
    Plot return and volatility forecasts, similar to the R markdown.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Return forecasts with CI
    axes[0].fill_between(
        forecast_df["Date"],
        forecast_df["Lower_95"],
        forecast_df["Upper_95"],
        color="steelblue",
        alpha=0.3,
        label="95% CI",
    )
    axes[0].plot(
        forecast_df["Date"],
        forecast_df["Forecasted_Return"],
        color="steelblue",
        marker="o",
        linewidth=1.2,
        label="Forecasted Return",
    )
    axes[0].axhline(0.0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("5-Day Log Return Forecast", fontweight="bold")
    axes[0].set_ylabel("Forecasted Log Return")
    axes[0].legend()

    # Volatility forecasts
    axes[1].plot(
        forecast_df["Date"],
        forecast_df["Forecasted_Volatility"],
        color="darkgreen",
        marker="o",
        linewidth=1.2,
    )
    axes[1].fill_between(
        forecast_df["Date"],
        0,
        forecast_df["Forecasted_Volatility"],
        alpha=0.3,
        color="darkgreen",
    )
    axes[1].set_title("5-Day Volatility Forecast", fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Forecasted Volatility (σ)")

    plt.tight_layout()
    plt.show()


def plot_historical_with_forecast(df_reg: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """
    Plot recent history of returns together with forecasts and 95% CI.
    """
    recent_window = 30
    recent_data = df_reg.tail(recent_window)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Historical returns
    ax.plot(
        recent_data["Date"],
        recent_data["log_returns"],
        color="steelblue",
        linewidth=0.8,
        label="Historical",
    )

    # Forecasted returns
    ax.plot(
        forecast_df["Date"],
        forecast_df["Forecasted_Return"],
        color="coral",
        linewidth=0.8,
        marker="o",
        label="Forecast",
    )

    # Forecast confidence band
    ax.fill_between(
        forecast_df["Date"],
        forecast_df["Lower_95"],
        forecast_df["Upper_95"],
        color="coral",
        alpha=0.2,
        label="95% CI (Forecast)",
    )

    ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Recent History and 5-Day Forecast (Log Returns)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main execution to mirror the RMarkdown flow
# ---------------------------------------------------------------------------

def main():
    # 2.1 Data Loading and Cleaning
    csv_file = "archive/bitcoin.csv"
    data = load_and_clean_data(csv_file)

    print("Dataset Summary:")
    print(f"- Date Range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"- Total Observations: {len(data)}")
    check_trading_days(data)

    # 2.2 Exploratory Data Analysis
    print("\nSummary Statistics for Bitcoin:")
    print(summary_statistics(data).round(4).to_string(index=False))
    plot_price_and_volume(data)

    # 2.3 Stationarity Analysis
    data_sta = prepare_stationarity_series(data)
    plot_stationarity_diagnostics(data_sta)
    plot_returns(data_sta)
    stationarity_tests(data_sta["returns"])

    # 3.1 Data Preparation for Modeling
    data_reg = prepare_regression_data(data)
    print(f"\nObservations for modeling: {len(data_reg)}")

    # 3.2 Model 1: OLS Regression
    print("\nOLS Regression on log returns:")
    ols_model = fit_ols_model(data_reg)
    ols_residual_diagnostics(ols_model.resid)

    # 3.3 Model 2: ARIMAX (Regression with ARMA Errors)
    fit_ns, fit_s = fit_arimax_models(data_reg)

    # 3.4 Volatility Modeling (eGARCH)
    seasonal_resid = pd.Series(fit_s.resid())
    egarch_res = volatility_diagnostics_and_garch(seasonal_resid)

    # 4. Forecasting
    forecast_df = combined_forecast(fit_s, egarch_res, data_reg, horizon=5)
    plot_forecasts(forecast_df)
    plot_historical_with_forecast(data_reg, forecast_df)


if __name__ == "__main__":
    main()


