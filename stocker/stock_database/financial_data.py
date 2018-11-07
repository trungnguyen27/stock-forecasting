import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
import matplotlib
from global_configs import configs
warnings.filterwarnings('ignore')

csv_path = configs['csv_path']

class FinancialData():
    currency = '000VND'
    def __init__(self, ticker = "VIC"):
        self.ticker = ticker.capitalize()
        data = pd.read_csv('%s/excel_vic.csv'%csv_path, parse_dates=[1])
        data.columns=["Ticker","Date","OpenFixed","HighFixed","LowFixed","CloseFixed","Volume","Open","High","Low","Close","VolumeDeal","VolumeFB","VolumeFS"]
        data.Timestamp=pd.to_datetime(data.Date, format='%d-%m-%Y %H:%M')
        data.index = data.Timestamp
        data = data.sort_values(by=['Date'], ascending=[True])
        data = data.resample('D').fillna(method='ffill')

        data['ds']=data.index
        data['y']=data['Close']

        if ('Adj. Close' not in data.columns):
            data['Adj. Close'] = data['Close']
            data['Adj. Open'] = data['Open']
        
        self.data = data
        # Minimum and maximum date in range
        self.min_date = min(data['Date'])
        self.max_date = max(data['Date'])

        self.years = (self.max_date - self.min_date).days/365
        
        self.max_price = np.max(self.data['y'])
        self.min_price = np.min(self.data['y'])
        
        self.min_price_date = self.data[self.data['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = self.data[self.data['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]
        
        # The starting price (starting with the opening price)
        self.starting_price = float(self.data.ix[0, 'Adj. Open'])
        
        # The most recent price
        self.most_recent_price = float(self.data.ix[len(self.data) - 1, 'y'])

    def get_data(self):
        if self.data.empty:
            print ('Stock data is empty')
            return
        else:
            print('Stock History Data of %s' %(self.ticker))
            return self.data

    def describe_stock(self):
        print(self.data.head())
        print('%d years of %s stock history data \n[%s To %s]' %(self.years, self.ticker, self.min_date, self.max_date))
        print('Lowest price on: %s with %d %s\nHighest price on: %s with %d %s' %(self.min_price_date, self.min_price, self.currency, self.max_price_date, self.max_price, self.currency))

    def get_moving_averages(self, lags = [30], columns=['Close']):
        # ing_averages= pd.DataFrame()
        moving_averages = dict()

        # with each columns, we calculate the moving average with lags
        for column in columns:
            try:
                for i, lag in enumerate(lags):
                    data = self.data[['ds', column]]
                    data = data.rename(columns = {column: 'y'})
                    data['y'] = data['y'].rolling(lag).mean()
                    moving_averages['MA_%d_%s' %(lag, column)] = data
            except Exception as e:
                print('An error occured:')
                print(e)

        self.lags = lags
        self.moving_averages = moving_averages

        lags_str =""
        for lag in lags:
            lags_str += str(lag) + ' '
        print('Moving averages generated: [{}] on columns [{}]'.format(lags_str, '-'.join(columns)))
        return moving_averages

    def plot_stock(self, columns=['Close'], show_data=True, show_volume=False, moving_averages = [1]):
        moving_averages = self.get_moving_averages(lags = moving_averages, columns=columns)
        self.reset_plot()
        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        plt.style.use('seaborn')
        
        if show_data:
            for i, column in enumerate(columns):
                plt.plot(self.data['ds'], self.data[column], color=colors[i])

        if show_volume:
            monthly_resampled_volume=self.data[['ds', 'Volume']].resample('M', on='ds').sum()
            min_max_scaler = MinMaxScaler()
            scaled_volume = min_max_scaler.fit_transform(np.array(monthly_resampled_volume).reshape(-1,1))
            scaled_volume *= max(self.data['Close'])/2
            plt.bar(monthly_resampled_volume.index, scaled_volume.flatten(), width=10)
        
        if moving_averages:
            for (col, avg) in moving_averages.items():
                plt.plot(avg['ds'], avg['y'] , ls='--', label = col)
                    
                
        #plt.plot(self.data['ds'], moving_avg, color='powderblue')
        plt.title('{} on {}'.format(self.ticker, ' '.join(columns)))
        plt.legend(loc='best')
        plt.show()

       # Reset the plotting parameters to clear style formatting
    # Not sure if this should be a static method
    @staticmethod
    def reset_plot():
        # Restore default parameters
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        
        # Adjust a few parameters to liking
        matplotlib.rcParams['figure.figsize'] = (8, 5)
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['text.color'] = 'k'