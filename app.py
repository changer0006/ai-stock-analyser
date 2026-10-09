import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="AI Stock Analyser",
    page_icon=":material/analytics:",
    layout="wide",
)

st.title("AI Stock Analyser", icon=":material/query_stats:")
st.caption(
    "Professional-grade historical market data analysis and machine learning forecasts."
)

with st.sidebar:
    st.header("Analysis Settings", icon=":material/settings:")
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    period = st.selectbox(
        "Historical period",
        ["1y", "2y", "5y", "10y"],
        index=1,
    )
    analyse = st.button("Analyse Stock", type="primary")
    st.caption("Note: Market data is historical and not guaranteed to be real-time.")


@st.cache_data(ttl=900)
def download_stock_data(symbol, selected_period):
    data = yf.download(
        symbol,
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("No historical data was returned. Check the ticker symbol.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = ["Close", "Volume"]

    if not all(column in data.columns for column in required_columns):
        raise ValueError("The downloaded data is missing required columns.")

    data = data[["Close", "Volume"]].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]

    if len(data) < 100:
        raise ValueError("Not enough historical observations for this analysis.")

    return data


def create_features(data):
    frame = data.copy()

    frame["Return"] = frame["Close"].pct_change()
    frame["MA_5"] = frame["Close"].rolling(5).mean()
    frame["MA_20"] = frame["Close"].rolling(20).mean()
    frame["MA_50"] = frame["Close"].rolling(50).mean()

    for lag in [1, 2, 3, 5, 10]:
        frame[f"Close_Lag_{lag}"] = frame["Close"].shift(lag)

    frame["Target"] = frame["Close"].shift(-1)

    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def make_price_chart(data):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Closing Price",
            line=dict(color="#00B4D8", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA_20"],
            name="20-Day Average",
            line=dict(color="#FFD166", width=1.5, dash="dot"),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MA_50"],
            name="50-Day Average",
            line=dict(color="#EF476F", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Adjusted Price (USD)",
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return fig


def make_prediction_chart(dates, actual, predicted):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            name="Actual Price",
            line=dict(color="#00B4D8", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted,
            name="Predicted Price",
            line=dict(color="#FFD166", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Adjusted Price (USD)",
        hovermode="x unified",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return fig


if analyse:
    if not ticker:
        st.error("Enter a stock ticker symbol.", icon=":material/error:")
        st.stop()

    with st.spinner(f"Downloading and processing data for {ticker}..."):
        try:
            raw_data = download_stock_data(ticker, period)
        except Exception as error:
            st.error(f"Unable to load stock data: {error}", icon=":material/error:")
            st.stop()

    st.badge("Data retrieved successfully", icon=":material/check_circle:", color="green")
    
    latest_close = float(raw_data["Close"].iloc[-1])
    previous_close = float(raw_data["Close"].iloc[-2])

    daily_change = (
        (latest_close - previous_close) / previous_close * 100
        if previous_close != 0
        else 0.0
    )

    # Key Metrics
    with st.container(horizontal=True):
        st.metric(
            "Latest Available Close",
            f"${latest_close:,.2f}",
            f"{daily_change:+.2f}%",
            border=True,
        )
        st.metric(
            "Historical Observations",
            f"{len(raw_data):,}",
            border=True,
        )
        st.metric(
            "Latest Available Date",
            raw_data.index[-1].strftime("%d %b %Y"),
            border=True,
        )

    chart_data = raw_data.copy()
    chart_data["MA_20"] = chart_data["Close"].rolling(20).mean()
    chart_data["MA_50"] = chart_data["Close"].rolling(50).mean()

    # Main Price Chart
    with st.container(border=True):
        st.subheader("Historical Price and Moving Averages", icon=":material/show_chart:")
        st.plotly_chart(
            make_price_chart(chart_data.dropna())
        )

    # Returns Analysis
    with st.container(border=True):
        st.subheader("Daily Returns Distribution", icon=":material/bar_chart:")
        returns = raw_data["Close"].pct_change().dropna() * 100
        return_fig = go.Figure(
            data=[
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    marker_color="#00B4D8",
                    opacity=0.8,
                    name="Daily Returns",
                )
            ]
        )
        return_fig.update_layout(
            xaxis_title="Daily Return (%)",
            yaxis_title="Number of Observations",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(return_fig)

    st.header("Machine Learning Forecast Evaluation", icon=":material/model_training:")

    feature_data = create_features(raw_data)

    feature_columns = [
        "Close",
        "Return",
        "MA_5",
        "MA_20",
        "MA_50",
        "Close_Lag_1",
        "Close_Lag_2",
        "Close_Lag_3",
        "Close_Lag_5",
        "Close_Lag_10",
    ]

    X = feature_data[feature_columns]
    y = feature_data["Target"]

    split_index = int(len(feature_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Not enough data to create training and test sets.", icon=":material/error:")
        st.stop()

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    baseline_predictions = X_test["Close"].to_numpy()

    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Model Performance (Random Forest)**")
            st.metric("Model MAE", f"{mae:.3f}")
            st.metric("Model RMSE", f"{rmse:.3f}")
    with col2:
        with st.container(border=True):
            st.markdown("**Baseline Comparison (Previous-Close)**")
            st.metric("Baseline MAE", f"{baseline_mae:.3f}")
            st.metric("Baseline RMSE", f"{baseline_rmse:.3f}")

    st.caption(
        "Lower MAE and RMSE indicate smaller prediction errors. "
        "The baseline predicts the next close using the current close. "
        "Test performance does not guarantee future accuracy."
    )

    with st.container(border=True):
        st.subheader("Held-Out Test Set: Actual vs Predicted", icon=":material/multiline_chart:")
        st.plotly_chart(
            make_prediction_chart(
                X_test.index,
                y_test.to_numpy(),
                predictions,
            )
        )

    final_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X, y)

    latest_features = X.iloc[[-1]]
    next_estimate = float(final_model.predict(latest_features)[0])

    with st.container(border=True):
        st.subheader("Next Trading Day Estimate", icon=":material/online_prediction:")
        st.metric(
            "Estimated Closing Price",
            f"${next_estimate:,.2f}",
        )
        st.warning(
            "This is an experimental model estimate, not a guaranteed price "
            "or a recommendation to buy or sell. Market conditions can change "
            "substantially, and the model may perform poorly.",
            icon=":material/warning:"
        )

    with st.expander("View Recent Historical Data", icon=":material/table_rows:"):
        st.dataframe(
            raw_data.tail(20)
        )

else:
    st.info(
        "Choose a stock ticker and historical period in the sidebar, "
        "then click **Analyse Stock**.",
        icon=":material/info:"
    )

    st.markdown(
        """
        ### Explore the market

        This application provides:
        - Historical price analysis
        - Moving averages and daily returns
        - Machine learning forecasts
        - Model evaluation and baseline comparison
        """
    )