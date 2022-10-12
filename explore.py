import streamlit as st
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np


def show_explore_page() :
 
    

    st.title('Cricket Score Dashboard')

    data_load_state = st.text('Loading data...')
    data =pd.read_csv("inningsI.csv")
    data1 =pd.read_csv("inningsII.csv")
    data_load_state.text('Loading data... done!')
        


    st.subheader('ROYAL ENFIELD VS ACCENTURE')
    st.write("INNINGS - I")
    st.write(data.tail())
    st.write("INNINGS - II")
    st.write(data1.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Overs'], y=data['run'], name="ACCENTURE"))
        fig.add_trace(go.Scatter(x=data1['Overs'], y=data1['run'], name="ROYALENFIELD"))
        fig.layout.update(title_text='Time Series data with Rangeslider(OVERS AND RUNS)', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()



    def barchart():
        df2  = data.groupby(["batsman"],as_index=False).sum()
        labels = df2["batsman"]
        size = df2["run"]
        fig = px.bar(x=labels, y=size, height=400)
        fig.layout.update(title_text='ACCENTURE BATSMAN WITH RUNS')

        # st.dataframe(df) # if need to display dataframe
        st.plotly_chart(fig)

    barchart()

    def barchart1():
        df2  = data1.groupby(["batsman"],as_index=False).sum()
        labels = df2["batsman"]
        size = df2["run"]
        fig = px.bar(x=labels, y=size, height=400)
        fig.layout.update(title_text='ROYALENFIELD BATSMAN WITH RUNS')

        # st.dataframe(df) # if need to display dataframe
        st.plotly_chart(fig)

    barchart1()

    def plot_runrate():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['runrate_ball'], y=data['Overs'], name="ACCENTURE"))
        fig.add_trace(go.Scatter(x=data1['runrate_ball'], y=data1['Overs'], name="ROYALENFIELD"))
        fig.layout.update(title_text='Time Series data with Rangeslider(RUNRATE)', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    plot_runrate()
    

    def regression1():
        fig =sns.lmplot(x='Overs',y='runs',data=data,aspect=2,height=6)
       
        plt.xlabel('OVERS')
        plt.ylabel('RUNS')
        plt.title('CRICKET SCORE PREDICTION INNING I');
        st.pyplot(fig)
    regression1()

    def regression2():
        fig =sns.lmplot(x='Overs',y='runs',data=data1,aspect=2,height=6)
        plt.xlabel('OVERS')
        plt.ylabel('RUNS')
        plt.title('CRICKET SCORE PREDICTION INNING II');
        st.pyplot(fig)
    regression2()

    
    def plot_WICKET_INNING1():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Overs'], y=data['run'], name="ACCENTURE RUNS"))
        # fig.add_trace(go.Scatter(x=data1['Overs'], y=data1['run'], name="ROYALENFIELD"))
        fig.add_trace(go.Scatter(x=data['Overs'], y=data['wickets'], name="WICKETS"))
        fig.layout.update(title_text='Wickets Timeseries For ACCENTURE', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_WICKET_INNING1()
    
    
    def plot_WICKET_INNING2():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Overs'], y=data['run'], name="ACCENTURE RUNS"))
        # fig.add_trace(go.Scatter(x=data1['Overs'], y=data1['run'], name="ROYALENFIELD"))
        fig.add_trace(go.Scatter(x=data1['Overs'], y=data1['wickets'], name="WICKETS"))
        fig.layout.update(title_text='Wickets Timeseries For ROYAL ENFIELD', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_WICKET_INNING2()

    
    fig = px.scatter(data, x="howout", y="Overs", color="HOWOUTS", symbol="HOWOUTS")
    fig.layout.update(title_text='ACCENTURE TEAM WICKETS', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    fig = px.scatter(data1, x="howout", y="Overs", color="HOWOUTS", symbol="HOWOUTS")
    fig.layout.update(title_text='ROYALENFIELD TEAM WICKETS', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


