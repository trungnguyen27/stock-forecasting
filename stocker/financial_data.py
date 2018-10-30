import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class FinancialData():
    def __init__(self):
        print('initialized')

    def load_data(ticker):
        dataset=pd.read_csv('./index_database/excel_vic.csv', parse_dates=[1])
        dataset.columns=["Ticker","Date","OpenFixed","HighFixed","LowFixed","CloseFixed","Volume","Open","High","Low","Close","VolumeDeal","VolumeFB","VolumeFS"]
        dataset.Timestamp=pd.to_datetime(dataset.Date, format='%d-%m-%Y %H:%M')
        dataset.index = dataset.Timestamp
        dataset = dataset.sort_values(by=['Date'], ascending=[True])
        dataset = dataset.resample('D').fillna(method='ffill')
        print(dataset.head())
        return dataset