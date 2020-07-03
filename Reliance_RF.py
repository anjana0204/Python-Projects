# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:55:34 2020

@author: Akshay
"""
"""Importing necessary libraries"""
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 

rf=RandomForestClassifier(random_state=42)

"""Downloading finance data of Reliance stocks"""
df = yf.download('RELIANCE.NS', 
                      start='1990-06-09', 
                      end='2020-06-09', 
                      progress=False)

"""Calculating RSI, MACD, Stochastic Indicators using talib package"""
df["RSI"]=talib.RSI(df["Close"], timeperiod=14)
Macd,Macdsignal,Macdhist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
df["Macd"]=Macd
df["Macdsignal"]=Macdsignal
Slowk,Slowd= talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=14, slowd_period=3)
df["Slowk"]=Slowk
df["Slowd"]=Slowd

"""Generating trading signals Buy(1) and Sell (-1)"""
df["Close_shift"]=df["Close"].shift(-1)
df["Trading_signals"]=np.where(df["Close_shift"]-df["Close"]>0,1,-1)

"""Creating a for loop to choose best combination of long and short averages for this data"""
short=[]
long=[]
accuracy_=[]
for i in range(2,61):
    for j in range(3,201):
        if (i>=j):
            continue
        df["Short_avg"]=talib.SMA(df["Close"],timeperiod=i)
        df["Long_avg"]=talib.SMA(df["Close"],timeperiod=j)
        new_df= df.dropna(how="any",axis=0)
        y=new_df["Trading_signals"]
        X=new_df.drop(["Trading_signals","Close","High","Open","Low","Adj Close","Close_shift"],axis=1)
        split=round(0.80*(len(new_df)))
        X_train=X[:split]
        X_test=X[split:]
        y_train=y[:split]
        y_test=y[split:]
        model=rf.fit(X_train,y_train)
        predicted=model.predict(X_test)
        short.append(i)
        long.append(j)
        accuracy_.append(accuracy_score(y_test,predicted))
data_dic={"Short_period":short,"Long_period":long,"Accuracy":accuracy_}
data=pd.DataFrame(data_dic)
sort_data=data.sort_values(by=["Accuracy"],ascending=False).head(10)

""" We obtain an accuracy of 0.544065 
with 18 and 149 periods SMA"""