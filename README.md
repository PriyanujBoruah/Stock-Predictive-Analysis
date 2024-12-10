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


## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/PriyanujBoruah/Stock-Predictive-Analysis.git
   cd Stock-Predictive-Analysis
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv .venv  # Use a descriptive name
   source .venv/bin/activate  # Activate (Linux/macOS)
   .venv\Scripts\activate      # Activate (Windows)
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Where `requirements.txt` contains:
   ```
   google-generative-ai>=0.2.0
   nltk>=3.8.1
   plotly>=5.0.0
   prophet>=1.1.1
   requests>=2.28.1
   streamlit>=1.0.0
   yfinance>=0.2.4
   pandas>=2.0.0
   ```

4. **Download NLTK Resources:**
   ```bash
   python
   >>> import nltk
   >>> nltk.download('vader_lexicon')
   >>> exit()
   ```

5. **Set API Keys (In `app.py`):**

   * **Google Generative AI API Key:** Replace `"YOUR_GOOGLE_API_KEY"`.
   * **NewsAPI Key:** Replace `"YOUR_NEWSAPI_KEY"`.


## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Enter Stock Ticker and Period:** Input the desired stock ticker and select the data period in the app interface.

3. **Click "Predict":** The app will fetch data, generate predictions, analyze sentiment, and display the results.


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


## Get API Keys

**1. Google Generative AI API Key:**

* **Obtain the key:**
    * Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (you might need to be logged in to your Google account).
    * Create a new API key.  Give it a descriptive name so you can recognize it later.  Copy the generated key.
* **Add to your code:** In your Python script (e.g., `your_script_name.py`), you'll need to set the environment variable `GOOGLE_API_KEY` or pass your api key directly to the `genai.configure()` method.  Here are both methods. Make sure to choose only one:

   ```python
   import google.generativeai as genai
   import os
   # Method 1. Using environment variables:
   os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # Replace with your actual key
   genai.configure()
   # ---- OR ----
   # Method 2: Passing the API key directly
   genai.configure(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your actual key


   ```
   Replace `"YOUR_GOOGLE_API_KEY"` with the API key you copied from the Google AI Studio website.


**2. NewsAPI Key:**

* **Obtain the key:**
    * Go to [https://newsapi.org/account](https://newsapi.org/account) (you'll need to create an account if you don't already have one).
    * Subscribe to a plan (there's a free tier available).
    * Obtain your API key.
* **Add to your code:**  There are two main ways to integrate your NewsAPI key:

    * **Directly in the API URL:**  This is the simplest but least secure option for local development.
        ```python
        api_url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey=YOUR_NEWSAPI_KEY"  # Replace with your key
        response = requests.get(api_url)
        ```

    * **As an environment variable (recommended):** This is more secure, especially for deployed applications.
        ```python
        import os
        import requests

        NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")  # Get the key from the environment

        if NEWSAPI_KEY is None:
            raise ValueError("NewsAPI key not found. Set the NEWSAPI_KEY environment variable.")


        api_url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={NEWSAPI_KEY}"
        response = requests.get(api_url)


        ```

        Then, before running your app, you would set the environment variable in your terminal:

        ```bash
        export NEWSAPI_KEY="YOUR_NEWSAPI_KEY"  # Linux/macOS
        set NEWSAPI_KEY="YOUR_NEWSAPI_KEY"   # Windows
        # Or if running from a .env file use. Example is for windows:
        set -o allexport; source .env; set +o allexport
        streamlit run your_script_name.py

        ```
        

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
