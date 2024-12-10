# Stock Predictive Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_dark.svg)](https://your-streamlit-app-url.streamlit.app/)


This Streamlit app provides stock price predictions and sentiment analysis using a combination of historical data, advanced time series forecasting with the Prophet model, and real-time news sentiment analysis.  It's designed to offer users a comprehensive view of a stock's potential future performance by combining quantitative analysis with qualitative insights derived from news articles.


## Features

* **Interactive Stock Selection:** Easily input the stock ticker symbol of interest.
* **Flexible Time Periods:** Choose from various historical data periods (e.g., 1 year, 5 years) to tailor the analysis.
* **Prophet-based Forecasting:** Leverages the powerful Prophet time series forecasting model, specifically designed for handling seasonality and trend changes in financial data.
* **Next-Day Price Prediction:**  Provides a prediction for the expected stock price on the next trading day.
* **Weekly Change Forecast:**  Estimates the potential percentage change in the stock price over the next week.
* **Real-Time News Sentiment:** Analyzes news headlines and descriptions related to the selected stock to gauge overall market sentiment (positive, negative, or neutral).  This helps contextualize the quantitative predictions.
* **News Aggregation and Display:**  Fetches and displays relevant news articles, providing users with direct access to the information driving the sentiment analysis.
* **Interactive Charts:**  Visualizes historical stock prices, predicted future prices, and uncertainty bands using interactive Plotly charts. This provides a clear and intuitive representation of the analysis.
* **Company Name Resolution:** Automatically retrieves the full company name associated with the provided ticker symbol.



## How it Works

This app utilizes several key technologies and methodologies to provide its predictive and analytical capabilities:


1. **Data Acquisition:**
   - Uses the `yfinance` library to fetch historical stock price data from Yahoo Finance.  The user-specifed time period determines the range of data retrieved.

2. **Price Forecasting with Prophet:**
   - The historical stock data is processed and formatted for use with the Prophet forecasting model.
   - Prophet, a robust time series forecasting library developed by Meta, is employed to generate future price predictions.  Prophet excels at handling the complexities of financial time series, including seasonality and trend changes.
   - The model outputs predictions for a specified future period (including the next day and the following week), along with uncertainty bounds.

3. **News Sentiment Analysis:**
   - The app uses the NewsAPI to fetch news articles relevant to the selected company. The company name is resolved using the Google Gemini generative AI model to ensure accurate news retrieval.
   - The Natural Language Toolkit (NLTK) library, combined with the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer, is used to analyze the sentiment expressed in news headlines and descriptions.  VADER assigns a sentiment score to each piece of text, indicating its positivity, negativity, or neutrality.  
   - The sentiment scores from multiple articles are aggregated to provide an overall sentiment trend for the stock.

4. **Visualization and Display:**
   - The Streamlit framework powers the interactive web application.
   - Plotly is used to create dynamic and visually appealing charts that display the historical stock prices, Prophet forecasts, and uncertainty bands.
   - The app presents the predicted next-day price, weekly change forecast, and overall news sentiment score in a clear and concise format.
   - Links to the relevant news articles are provided to allow users to delve deeper into the information driving the sentiment analysis.


## Core Code Snippets Explained

**1. Data Retrieval:**

```python
import yfinance as yf

def get_stock_data(ticker, period="5y"):
    data = yf.download(ticker, period=period)  # Downloads historical data
    return data['Close']  # Returns closing prices
```

This function uses `yfinance` to download historical stock data for a given ticker and period. It returns the closing prices.

**2. Prophet Forecasting:**

```python
from prophet import Prophet
import pandas as pd

def predict_prophet(data, forecast_days=7):
    df = pd.DataFrame({'ds': data.index, 'y': data.values}) # Formats data for Prophet
    model = Prophet()
    model.fit(df) # Trains the Prophet model
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future) # Makes predictions
    return forecast # Returns the forecast DataFrame
```

This function takes the historical data, formats it for Prophet, trains a Prophet model, and generates future predictions.

**3. Sentiment Analysis:**

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text) # Calculates sentiment scores
    return scores
```

This function uses VADER to analyze the sentiment of a given text and returns the polarity scores.

**4. News Retrieval and Display:**

```python
import requests as req

def NEWSrequest_newsapi(api_url):
    newsapi_req = req.get(api_url).json()
    # ... (Code to extract title, description, source, URL)
    return TITLE_list, DESCRIPTION_list, SOURCE_list, URL_list
# ... (Code to display news in Streamlit)
```

This snippet retrieves news data from NewsAPI, extracts relevant information, and then uses Streamlit's functionalities (e.g., `st.write`, `st.link_button`) to display the news articles within the app.


**5. Plotly Charting:**

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name='Historical Data'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
# ... (Code to add other traces like upper and lower bounds)
st.plotly_chart(fig) # Displays the chart in Streamlit
```
This section demonstrates how Plotly is used to create interactive charts.  It adds traces for historical data, the forecast, and optionally, the upper and lower confidence bounds of the forecast.  The `st.plotly_chart()` function then renders the chart in the Streamlit app.


## Technologies Used

* **Streamlit:**  Framework for building interactive web apps.
* **yfinance:**  Library for retrieving stock market data from Yahoo Finance.
* **Prophet (from Meta):** Time series forecasting library optimized for business scenarios.
* **Plotly:**  Charting library for creating interactive visualizations.
* **NewsAPI:**  API for accessing real-time news articles.
* **Google Gemini Generative AI:** Model for resolving company names from ticker symbols.
* **NLTK (Natural Language Toolkit):**  Library for natural language processing tasks.
* **VADER (Valence Aware Dictionary and sEntiment Reasoner):**  Lexicon and rule-based sentiment analysis tool.
* **Pandas:** Data manipulation and analysis library.
* **Requests:**  Library for making HTTP requests to APIs.



## Disclaimer

This app is for educational and informational purposes only and should not be considered financial advice.  The predictions generated by the model are not guaranteed to be accurate and should not be used as the sole basis for investment decisions. Stock market predictions are inherently uncertain, and past performance is not indicative of future results. Always consult with a qualified financial advisor before making any investment decisions.
