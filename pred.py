from sqlalchemy import over
import streamlit as st
from plotly import graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

from  prophet import Prophet
from  prophet.plot import plot_plotly


filename = 'predictionFinal.pkl'
regressor = pickle.load(open(filename, 'rb'))

# def load_model():
#     with open('E:/predictionFinal.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return data

# data = load_model()

# regressor = data["model"]

def show_predict_page():
    st.title('Cricket Score Prediction')

    ds = ("CHOOSE TEAM","ACCENTURE","ROYALENFIELD")
    battingTeam= st.selectbox('BATTING TEAM', ds)
    bowlingTeam= st.selectbox('BOWLING TEAM', ds)

    overs = st.text_input('OVERS', )
    st.write('Want to predicted :', overs)

    wickets = st.slider('WICKETS', 0, 10)
    

    ok = st.button("PREDICT SCORE")

    if ok:
        temp_array=[]

        batting_team = battingTeam
        if batting_team == 'ACCENTURE':
            temp_array = temp_array + [1,0]
        else:
            temp_array = temp_array + [0,1]
        
        bowl_team = bowlingTeam
        if bowl_team == 'ACCENTURE':
            temp_array = temp_array + [1,0]
        else:
            temp_array = temp_array + [0,1]
    
        temp_array = temp_array + [overs, wickets]
    
        


        data = np.array([temp_array])
        data= data.astype(float)
        # overs = int(overs)
        overs= float(overs)
        predict = regressor.predict(data)
        balls = int(regressor.predict(data)[0][0])
        finalScore = int(regressor.predict(data)[0][1])
        # runRate = float(regressor.predict(data)[0][2])
        runRate = finalScore / overs

        st.subheader(f"Predicted Score :  {finalScore}")
        st.subheader(f"Total RunRate For {battingTeam} In {overs} Overs :  {runRate:.2f}")
        st.subheader(f"Total balls For Predicted Score :  {balls}")



        # m = Prophet

        # # Show and plot forecast
        # st.subheader('Forecast data')
        # st.write(predict)
        
        encoded_df = pd.read_csv("streamlitForecastPredict.csv")
        # Predict forecast with Prophet.
        

        # df_train = encoded_df.drop(['ball_counts','runs','runrate_ball'],axis=1)
        df_train = encoded_df[['Date','runs']]
        # df_train['ds'] = pd.to_datetime(df_train['date'])
        df_train['Date'] = pd.to_datetime(df_train['Date'])
        # df_train["Date"] = pd.to_datetime(df_train["Date"], errors = 'coerce')
        # df_train['Date'] = df_train['Date'].values.astype(float)
        df_train = df_train.rename(columns={"Date": "ds", "runs": "y"})
        # df_train= df_train.astype(int)
        
       
        overs1 = round(overs)
        period = int(overs1)
       
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(forecast.tail())
        
        st.write(f'Forecast plot for {overs} Overs')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)








