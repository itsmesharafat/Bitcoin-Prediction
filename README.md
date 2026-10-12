# Bitcoin Price Prediction Model

A simple and interactive Bitcoin price prediction app built with Python, TensorFlow, scikit-learn, and Streamlit.

Created by: Sharafat

## Overview

This project predicts Bitcoin prices using a Long Short-Term Memory (LSTM) neural network trained on historical BTC-USD data from Yahoo Finance. The app downloads recent market data, prepares the time series, runs the model, and visualizes the forecast.

## Features

- Real-time Bitcoin market data via Yahoo Finance
- LSTM-based price prediction model
- Data visualization with Streamlit charts
- Historical price table and prediction comparison
- Future price forecast window
- Clean dark UI for a modern look

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Streamlit
- yfinance

## Project Structure

```bash
Bitcoin-Prediction/
├── app.py
├── Bitcoin_ML_Model.ipynb
├── Bitcoin_Price_prediction_Model.keras
├── LICENSE
└── README.md
```

## Setup

1. Clone the project
2. Open the project folder
3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

4. Install dependencies

```bash
pip install numpy pandas yfinance tensorflow scikit-learn streamlit
```

5. Run the app

```bash
streamlit run app.py
```

## Usage

Once the app starts, it will:

- fetch Bitcoin price data
- display the dataset
- plot the price trend
- run predictions against the trained LSTM model
- show predicted vs original values
- visualize the future forecast

## Example

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Notes

- The project uses a trained `.keras` model stored in the repository.
- Yahoo Finance formatting may vary slightly by version, so data normalization is handled in the app for compatibility.
- The model is designed for experimentation and educational demonstration.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Sharafat

---

Made with Python and a passion for data-driven forecasting.
