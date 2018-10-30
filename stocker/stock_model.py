import pandas as pd
import numpy as np
from fbprophet import Prophet
import pytrends
from pytrends.request import TrendReq
# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
import matplotlib
from financial_data import FinancialData as fd

class SModel():
    def __init__(self, ticker):
        df = fd.load_data('vic')
        self.symbol = ticker.capitalize()
        df['ds']=df.index
        df['y']=df['Close']
        self.stock = df.copy()
        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05 
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.quarterly_seasonality = True
        self.changepoints = None

    def build_model(self):
        model = Prophet(interval_width=0.2, daily_seasonality=self.daily_seasonality, weekly_seasonality=self.weekly_seasonality, yearly_seasonality=self.yearly_seasonality, changepoint_prior_scale=self.changepoint_prior_scale, changepoints=self.changepoints)
        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        if self.quarterly_seasonality:
            model.add_seasonality(name='quarterly', period=90, fourier_order=5)
        if self.yearly_seasonality:
            model.add_seasonality(name='yearly', period=365, fourier_order=5)
        
        return model

    def predict(self, use_moving_avg = False, days = 30, training_years = 1):
        
        # Use past self.training_years years for training
        training_sets = dict()
        futures= pd.DataFrame()
        if use_moving_avg:
            for i, lag in enumerate(self.lags):
                ma_set = pd.DataFrame(columns=['y', 'ds', 'label'])
                ma_set['y'] = self.moving_averages['ma_{}'.format(lag)]
                ma_set['ds'] = self.stock['ds']
                training_sets['ma_{}'.format(lag)] = ma_set
        else:
            self.stock['y'] = self.stock['Close']
            training_sets.append(self.stock)

        
        #train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=training_years)).date()]
        plt.style.use('seaborn')
        futures_dict = dict()
        predictions_dict = dict()
        for i, (key, training_set) in enumerate(training_sets.items()):
            model = self.build_model()
            model.fit(training_set)
        
            # Future dataframe with specified number of days to predict
            predicted = model.make_future_dataframe(periods=days, freq='D')
            predicted = model.predict(predicted)
            # Only concerned with future dates
            future = predicted[predicted['ds'] >= max(self.stock['Date']).date()]
        
            # Remove the weekends
            #future = self.remove_weekends(future)
            
            # Calculate whether increase or not
            future['diff'] = future['yhat'].diff()
        
            future = future.dropna()

            # Find the prediction direction and create separate dataframes
            future['direction'] = (future['diff'] > 0) * 1
            
            # Rename the columns for presentation
            future = future.rename(columns={'ds': 'Date', 'yhat': 'estimate', 'diff': 'change', 
                                            'yhat_upper': 'upper', 'yhat_lower': 'lower'})
            
            futures_dict[key] = future
            predictions_dict[key] = predicted
            
            self.future_increase = future[future['direction'] == 1]
            self.future_decrease = future[future['direction'] == 0]

            # # Print out the dates
            # print('\nPredicted Increase: \n')
            # print(self.future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])
            
            # print('\nPredicted Decrease: \n')
            # print(self.future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])
            self.futures = futures_dict
            self.predictions = predictions_dict
        self.plot_history_and_prediction()

        
    def plot_predicted_range(self):
        self.reset_plot()
        # Set up plot

        plt.style.use('fivethirtyeight')
        matplotlib.rcParams['axes.labelsize'] = 10
        matplotlib.rcParams['xtick.labelsize'] = 8
        matplotlib.rcParams['ytick.labelsize'] = 8
        matplotlib.rcParams['axes.titlesize'] = 12
        
        # Plot the predictions and indicate if increase or decrease
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot the estimates
        ax.plot(self.future_increase['Date'], self.future_increase['estimate'], 'g^', ms = 12, label = 'Pred. Increase')
        ax.plot(self.future_decrease['Date'], self.future_decrease['estimate'], 'rv', ms = 12, label = 'Pred. Decrease')

        # Plot errorbars
        ax.errorbar(self.future['Date'].dt.to_pydatetime(), self.future['estimate'], 
                    yerr = self.future['upper'] - self.future['lower'], 
                    capthick=1.4, color = 'k',linewidth = 2,
                   ecolor='darkblue', capsize = 4, elinewidth = 1, label = 'Pred with Range')

        # Plot formatting
        plt.legend(loc = 2, prop={'size': 10});
        plt.xticks(rotation = '45')
        plt.ylabel('Predicted Stock Price (VNĐ)');
        plt.xlabel('Date'); plt.title('Predictions for %s' % self.symbol);
        plt.show()
    
    def plot_history_and_prediction(self):
        self.reset_plot()
        plt.title('Predicted Price on Close Price and Moving Averages on {}'.format(self.symbol))
        for (key, predicted) in self.predictions.items():
            history_range = predicted[predicted['ds'] < max(self.stock['ds'])]
            plt.plot(history_range['ds'], history_range['yhat'], label=key)
            plt.plot(self.futures[key]['Date'], self.futures[key]['estimate'], label='predicted {}'.format(key))
        plt.legend(loc='best')
        plt.show()
        

    def plot_stock(self, columns=['Close'], show_volume=False, moving_avg_lags=[30]):
        self.reset_plot()
        self.get_moving_average(lags = moving_avg_lags)
        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        plt.style.use('seaborn')

        for i, column in enumerate(columns):
            plt.plot(self.stock['ds'], self.stock[column], color=colors[i])

        if show_volume:
            plt.bar(self.stock['ds'].values, self.stock['Volume'])
        
        for i, (key, moving_avg) in enumerate(self.moving_averages.items()):
            plt.plot(self.stock['ds'], moving_avg, ls='--', label = key)

        plt.plot(self.stock['ds'], moving_avg, color='powderblue')
        plt.title('{} on {}'.format(self.symbol, ' '.join(columns)))
        plt.legend(loc='best')
        plt.show()

    def get_moving_average(self, lags = [30], columns='Close'):
        close_price_log = np.log(self.stock[columns])
        close_price = self.stock['Close']
        self.moving_averages= pd.DataFrame()
        self.lags = lags
        for i, lag in enumerate(lags):
            self.moving_averages['ma_{}'.format(lag)] = close_price.rolling(lag).mean()
        self.reset_plot()
        return self.moving_averages
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