# 📈 AI-Based Stock Analyser

An interactive stock analysis dashboard built with Python, Streamlit,
and machine learning. The application retrieves historical stock market
data, visualizes price trends, and evaluates a machine learning model
for next-trading-day closing-price prediction.

> **Disclaimer:** This project is for educational purposes only.
> Stock price predictions are uncertain and should not be considered
> financial advice.

## 🎯 Project Overview

The AI-Based Stock Analyser explores how machine learning can be used
to analyze historical stock market data.

Users can select a stock ticker, examine historical price movements,
view technical indicators, and evaluate model predictions against
actual stock prices.

The project combines financial data analysis, data visualization,
and machine learning in an interactive web application.

## ✨ Features

- Retrieve historical stock market data.
- Visualize historical closing prices.
- Analyze moving averages and daily returns.
- Train and evaluate a machine learning model.
- Compare predicted prices with actual prices.
- Evaluate prediction errors using MAE and RMSE.
- Explore stock data through an interactive dashboard.

*Keep only the features that are implemented and working in your current version.*

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Streamlit | Interactive web dashboard |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| yfinance | Historical market data retrieval |
| Scikit-learn | Machine learning and model evaluation |
| Plotly | Interactive visualizations |

## 🧠 Machine Learning Approach

The project uses historical stock data to explore next-trading-day
closing-price prediction.

The workflow consists of:

1. Retrieve historical stock market data.
2. Clean and prepare the dataset.
3. Generate features from historical prices and returns.
4. Split the data chronologically into training and testing sets.
5. Train the machine learning model using training data.
6. Generate predictions for the test period.
7. Evaluate predictions using Mean Absolute Error (MAE)
   and Root Mean Squared Error (RMSE).

### Evaluation Metrics

- **MAE:** Measures the average absolute difference between predicted
  and actual prices.
- **RMSE:** Measures prediction error while penalizing larger errors
  more heavily.

Model performance should be compared against a simple baseline,
such as predicting that the next closing price will equal the
current closing price.

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Git.
- An internet connection for retrieving market data.

### 1. Clone the repository

```bash
git clone https://github.com/changer0006/ai-stock-analyser.git
cd ai-stock-analyser
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 📊 How to Use

1. Launch the application.
2. Enter a supported stock ticker.
3. Retrieve historical market data.
4. Explore the available charts and indicators.
5. Review the model's predictions and evaluation metrics.

Examples of ticker symbols:

- `RELIANCE.NS` — Reliance Industries on NSE.
- `TCS.NS` — Tata Consultancy Services on NSE.
- `AAPL` — Apple Inc. on the US market.

Availability depends on the data provider.

## ⚠️ Limitations

- Historical price patterns do not reliably predict future prices.
- Market data availability depends on the external data provider.
- Model performance can change across stocks and time periods.
- Prediction errors do not capture every source of financial risk.
- This project is not intended to provide investment recommendations.

## 🔮 Future Improvements

- Compare multiple machine learning models.
- Add feature importance and model explainability.
- Introduce financial risk metrics.
- Store historical data in a database.
- Add portfolio analysis.
- Develop a backtesting framework.
- Add automated tests and continuous integration.

*Move each item to the implemented features section once it has been completed and tested.*

## 👨‍💻 Author

**Sree Ram Kumar**

B.Tech Computer Science Engineering | 2027

- GitHub: [changer0006](https://github.com/changer0006)

## 📄 License

No license has been specified yet.
