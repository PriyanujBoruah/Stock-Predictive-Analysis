import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import date, timedelta
import streamlit as st
import plotly.graph_objects as go
import requests as req
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.set_page_config(page_title="Stock Predictive Analysis by Priyanuj", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

def get_stock_data(ticker, period="5y"):
    """Retrieves historical stock data from Yahoo Finance."""
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=int(period[:-1]) * 365)
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        return data['Close']  # Return closing prices
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def predict_prophet(data, forecast_days=7):
    """Predicts future prices using Prophet."""
    try:
        df = pd.DataFrame({'ds': data.index, 'y': data.values})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        next_day_price = forecast['yhat'].iloc[-forecast_days]
        weekly_change = (forecast['yhat'].iloc[-1] / data[-1] - 1) * 100
        return forecast, next_day_price, weekly_change # Return the forecast DataFrame

    except Exception as e:
        print(f"Prophet prediction error: {e}")
        return None, None, None

def NEWSrequest_newsdata(api_url):
    newsdata_req = req.get(api_url)
    newsdata_req = newsdata_req.json()
    TITLE_list, DESCRIPTION_list, SOURCE_list, URL_list = JSONtoTEXT_newsdata(newsdata_req)
    return TITLE_list, DESCRIPTION_list, SOURCE_list, URL_list



def NEWSrequest_newsapi(api_url):
    newsapi_req = req.get(api_url)
    newsapi_req = newsapi_req.json()
    TITLE_list, DESCRIPTION_list, SOURCE_list, URL_list = JSONtoTEXT_newsapi(newsapi_req)
    return TITLE_list, DESCRIPTION_list, SOURCE_list, URL_list

def JSONtoTEXT_newsapi(news):
    i = 0
    TITLE_list = []
    DESCRIPTION_list = []
    SOURCE_list = []
    URL_list = []
    while True:
        try:
            TITLE_list.append(str(news['articles'][i]['title']))
            DESCRIPTION_list.append(str(news['articles'][i]['description']))
            SOURCE_list.append(str(news['articles'][i]['source']['name']))
            URL_list.append(str(news['articles'][i]['url']))

        except:
            break
        i = i + 1
    return TITLE_list, DESCRIPTION_list, SOURCE_list, URL_list

def JSONtoTEXT_newsdata(news):
    i = 0
    TITLE_list = []
    DESCRIPTION_list = []
    SOURCE_list = []
    URL_list = []
    while True:
        try:
            TITLE_list.append(str(news['results'][i]['title']))
            DESCRIPTION_list.append(str(news['results'][i]['description']))
            SOURCE_list.append(str(news['results'][i]['source_name']))
            URL_list.append(str(news['results'][i]['link']))
        except:
            break
        i = i + 1
    return TITLE_list, DESCRIPTION_list, SOURCE_list, URL_list

def NAME(TICKER):
    NAME = model.generate_content(f"Give only the name of the company of this ticker symbol {TICKER}")
    return NAME.text

def analyze_sentiment(text):

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores


#with open( "style.css" ) as css:
#    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

nltk.download('vader_lexicon')

GOOGLE_API_KEY = st.secrets["GOOGLE_API_key"]
NEWS_API_KEY = st.secrets["NEWS_API_key"]


genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')



# Streamlit app
st.html(f"<h1 style='text-align: center; color: #2f3133'>Stock<span style='color: #076EFF'> Predictive </span>Analysis <span style='font-size: medium;'>by <span style='color: #076EFF'>Priyanuj Boruah</span></span></h1>")

TOP_COL1, TOP_COL2, TOP_COL3, TOP_COL4 = st.columns(4, vertical_alignment="bottom")

ticker = TOP_COL2.text_input("Stock Ticker (e.g., AAPL, MSFT):")
period = TOP_COL3.selectbox("Data period:", ["1y", "2y", "5y", "10y"], index=2)  # Default to 5y

if TOP_COL4.button("Predict", type="primary"):
    
    STOCK, NEWS = st.columns([2,1])

    data = get_stock_data(ticker, period)

    if data is not None:
        forecast, next_day_prophet, weekly_change_prophet = predict_prophet(data)


        if next_day_prophet is not None:
            ticker_org = NAME(ticker)
            STOCK.html(f"<h2 style='color: #076EFF'>{ticker_org}</h2>")
            STOCK.html(f"<h4>Expected Value Tomorrow: <b style='color: #076EFF'>{next_day_prophet:.2f}</b></h4>")
            WEEK_COL, MEDIA_COL = STOCK.columns(2)
            if weekly_change_prophet > 0:
                WEEK_COL.html(f"<p style='font-size: large'>Expected Weekly Change: <b style='color: #076EFF'>{weekly_change_prophet:.2f}%</b></p>")
            else:
                WEEK_COL.html(f"<p style='font-size: large'>Expected Weekly Change: <b style='color: red'>{weekly_change_prophet:.2f}%</b></p>")


            api_url1 = f"https://newsapi.org/v2/everything?q={ticker_org}&apiKey={NEWS_API_KEY}"
            TITLE_list1, DESCRIPTION_list1, SOURCE_list1, URL_list1 = NEWSrequest_newsapi(api_url1)
            
            SENTIMENT_LIST = []
        
            for i in range(len(TITLE_list1)):
                if TITLE_list1[i] != "[Removed]" and DESCRIPTION_list1[i] != "[Removed]":
                    sentiment = analyze_sentiment(TITLE_list1[i]+DESCRIPTION_list1[i])
                    SENTIMENT_LIST.append(sentiment["compound"]) 
            
            mean_SENTIMENT = sum(SENTIMENT_LIST) / len(SENTIMENT_LIST)

            if mean_SENTIMENT > 0:
                MEDIA_COL.html(f"<p style='font-size: large'>Media Sentiment Score: <b style='color: #076EFF'>{mean_SENTIMENT:.4f}</b></p>")
            else:
                MEDIA_COL.html(f"<p style='font-size: large'>Media Sentiment Score: <b style='color: red'>{mean_SENTIMENT:.4f}</b></p>")
            
            # Create Plotly figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(color='DarkBlue'), name='Lower Bound', fill='tonexty', fillcolor='lightblue'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='DodgerBlue'), name='Upper Bound', fill='tonexty', fillcolor='lightblue'))
            fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', line=dict(color='DeepPink'), name='Historical Data'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line=dict(color='black'), name='Forecast'))
            
            fig.update_layout(title=f"{ticker} Stock Price Forecast",
                              xaxis_title="Date",
                              yaxis_title="Price")
            STOCK.plotly_chart(fig)

    
            
            NEWS_CONTAINER = NEWS.container(border=True, height=600)

            NEWS_CONTAINER.html(f"<h3>Latest news related to {ticker_org}</h3>")

            news_count = 0

            for i in range(len(TITLE_list1)):
                if news_count < 10:
                    if TITLE_list1[i] not in ["None", "[Removed]"] and DESCRIPTION_list1[i] not in ["None", "[Removed]"]:
                        NEWS_CONTAINER.html(f"<h3 style='font-size: large'>{TITLE_list1[i]}</h3>")
                        NEWS_CONTAINER.html(f"<p style='font-size: large'>{DESCRIPTION_list1[i]}</p>")
                        SOURCE_COL, URL_COL = NEWS_CONTAINER.columns(2, vertical_alignment="center")
                        SOURCE_COL.html(f"<p><b>{SOURCE_list1[i]}</b></p>")
                        URL_COL.link_button("See News", URL_list1[i])
                        NEWS_CONTAINER.write("---")
                        news_count += 1

    else:
        st.write("Error fetching stock data. Please check the ticker symbol and period.")

    
