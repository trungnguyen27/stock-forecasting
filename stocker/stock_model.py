import pandas as pd
import numpy as np
from fbprophet import Prophet
import pytrends
from pytrends.request import TrendReq
# matplotlib pyplot for plotting
import matplotlib.pyplot as plt
import matplotlib
from financial_data import FinancialData as fd
from sklearn.preprocessing import MinMaxScaler

class SModel():
    def __init__(self, ticker):
        df = fd.load_data('vic')
        self.symbol = ticker.capitalize()
        df['ds']=df.index
        df['y']=df['Close']
        self.stock = df.copy()
          
        # Minimum and maximum date in range
        self.min_date = min(df['Date'])
        self.max_date = max(df['Date'])

        self.intialize_model_parameters()

    def intialize_model_parameters(self, 
                            seasonalities=['monthly', 'quarterly', 'yearly'],
                            changepoints = None,
                            training_years=10):
        self.reset_model_paramaters()

        for seasonality in seasonalities:
            if seasonality == 'daily':
                self.daily_seasonality = True
            elif seasonality == 'weekly':
                self.weekly_seasonality = True
            elif seasonality == 'monthly':
                self.monthly_seasonality = True
            elif seasonality == 'yearly':
                self.yearly_seasonality = True
            elif seasonality == 'quarterly':
                self.quarterly_seasonality = True
    
        self.changepoints = changepoints
        self.training_years = training_years
            
        print('Seasonalities: Daily[{}] Weeky[{}] Monthly[{}] Yearly[{}] Quarterly[{}]'
                .format(self.daily_seasonality, 
                    self.weekly_seasonality,
                    self.monthly_seasonality,
                    self.yearly_seasonality,
                    self.quarterly_seasonality))
        print('Changepoints: {}'.format(' '.join(changepoints if changepoints else ['None'])))
        print('Number of training years: {}'.format(training_years))

    def reset_model_paramaters(self):
        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05 
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = False
        self.yearly_seasonality = False
        self.quarterly_seasonality = False
        self.changepoints = None
        self.training_years = 0
       
    def build_model(self):
        model = Prophet(interval_width=0.2, 
                        daily_seasonality=self.daily_seasonality, 
                        weekly_seasonality=self.weekly_seasonality, 
                        yearly_seasonality=self.yearly_seasonality, 
                        changepoint_prior_scale=self.changepoint_prior_scale, 
                        changepoints=self.changepoints)
        if self.monthly_seasonality:
            # Add monthly seasonality
            model.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5)
        if self.quarterly_seasonality:
            model.add_seasonality(name='quarterly', period=90, fourier_order=5)
        if self.yearly_seasonality:
            model.add_seasonality(name='yearly', period=365, fourier_order=5)
        
        return model

    def predict(self, use_moving_avg = False, days = 30):
        # Use past self.training_years years for training
        futures = dict()
        if use_moving_avg:
            lags = self.moving_averages['lags']
            futures['lags']= lags
            for i, (key, value) in enumerate(self.moving_averages.items()):
                if key == 'lags':
                    continue
                predicted = pd.DataFrame()
                for lag in lags:
                    d = {'ds': value['ds'], 'y': value['ma_%d' %lag]}
                    train_set = pd.DataFrame(data=d)
                    result = self.predict_single_dataset(train=train_set, days=days)

                    # Rename the columns for presentation
                      # Rename the columns for presentation
                    renamed = result['predicted'].rename(columns={'yhat': 'estimate_%d' %lag, 'diff': 'change_%d' %lag, 
                                        'yhat_upper': 'upper_%d' %lag, 'yhat_lower': 'lower_%d' %lag})
                    if predicted.empty:
                        predicted = renamed
                    else:
                        predicted = pd.merge(predicted, renamed, on = 'ds', how = 'inner')
                    # value['increases_%d' %lag] = result[result['predicted']['direction'] ==1]
                    # value['decreases_%d' %lag] = result[result['predicted']['direction'] ==0]
                futures[key] = predicted
        else:
            self.stock['y'] = self.stock['Close']
            training_sets['ma_0']= self.stock
        self.futures = futures
        # self.training_sets = training_sets
        
        # #train = self.stock[self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=training_years)).date()]
        # plt.style.use('seaborn')
       
        # self.predictions = predictions_dict
        # self.future_increases= future_increases
        # self.future_decreases = future_decreases
        self.plot_history_and_prediction()
        # for i, (key, training_set) in enumerate(training_sets.items()):
        #     result = self.predict_single_dataset()


            # # Print out the dates
            # print('\nPredicted Increase: \n')
            # print(self.future_increase[['Date', 'estimate', 'change', 'upper', 'lower']])
            
            # print('\nPredicted Decrease: \n')
            # print(self.future_decrease[['Date', 'estimate', 'change', 'upper', 'lower']])
            
        
    def predict_single_dataset(self, train, days):
        model = self.build_model()
        model.fit(train)
    
        # Future dataframe with specified number of days to predict
        predicted = model.make_future_dataframe(periods=days, freq='D')
        predicted = model.predict(predicted)
        # Only concerned with future dates
        future = predicted[predicted['ds'] >= max(self.stock['Date']).date()]
    
        # Remove the weekends
        #future = self.remove_weekends(future)
        
        # Calculate whether increase or not
        predicted['diff'] = predicted['yhat'].diff()
        future['diff'] = future['yhat'].diff()
    
        future = future.dropna()

        # Find the prediction direction and create separate dataframes
        future['direction'] = (future['diff'] > 0) * 1
        predicted['direction'] = (predicted['diff']> 0) *1
        result = dict()
        result['predicted'] = predicted 
        result['future'] = future
        return result

    def plot_history_and_prediction(self):
        self.reset_plot()
        plt.title('Predicted Price on Close Price and Moving Averages on {}'.format(self.symbol))
        lags = self.moving_averages['lags']
        for (key, future) in self.futures.items():
            if key == 'lags':
                continue

            for lag in lags:
                history_range = future[future['ds'] < max(self.stock['ds'])]
                future_range = future[future['ds'] > max(self.stock['ds'])]
                plt.plot(history_range['ds'], history_range['estimate_%d' %lag], label=key)
                plt.plot(future_range['ds'], future_range['estimate_%d' %lag], label='predicted %d' %lag)
                
                # Plot the uncertainty interval
                plt.fill_between(future['ds'].dt.to_pydatetime(), future['upper_%d' %lag],
                                    future['lower_%d' %lag],
                                    alpha = 0.3, edgecolor = 'k', linewidth = 0.6)

        plt.legend(loc='best')
        plt.show()
        
      # Graph the effects of altering the changepoint prior scale (cps)
    
    def changepoint_prior_analysis(self, changepoint_priors=[0.001, 0.05, 0.1, 0.2], colors=['b', 'r', 'grey', 'gold']):
        # Training and plotting with specified years of data
        #train = self.stock[(self.stock['Date'] > (max(self.stock['Date']) - pd.DateOffset(years=self.training_years)).date())]
        
        # Iterate through all the changepoints and make models
        for index, prior in enumerate(changepoint_priors):
            # Select the changepoint
            self.changepoint_prior_scale = prior
            
            # Create and train a model with the specified cps
            for i, (key, train) in enumerate(self.training_sets.items()):
                model = self.build_model()
                model.fit(train)
                future = model.make_future_dataframe(periods=180, freq='D')
                    
                future = model.predict(future)

                # Make a dataframe to hold predictions
                if index == 0:
                    predictions = future.copy()
                
                # Fill in prediction dataframe
                predictions['%.3f_yhat_upper_%s' %(prior, key)] = future['yhat_upper']
                predictions['%.3f_yhat_lower_%s' %(prior, key)] = future['yhat_lower']
                predictions['%.3f_yhat_%s' %(prior, key)] = future['yhat']
         
                # Remove the weekends
                #predictions = self.remove_weekends(predictions)
                
        # Plot set-up
        self.reset_plot()
        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots(1, 1)
        
        # Actual observations
        ax.plot(train['ds'], train['y'], 'ko', ms = 4, label = 'Observations')
        color_dict = {prior: color for prior, color in zip(changepoint_priors, colors)}

        # Plot each of the changepoint predictions
        for prior in changepoint_priors:
            for i, (key, train) in enumerate(self.training_sets.items()):
                # Plot the predictions themselves
                ax.plot(predictions['ds'], predictions['%.3f_yhat_%s' %(prior, key)], linewidth = 1.2,
                        color = color_dict[prior], label = '%.3f prior scale_%s' %(prior, key))
                
                # Plot the uncertainty interval
                ax.fill_between(predictions['ds'].dt.to_pydatetime(), predictions['%.3f_yhat_upper_%s' %(prior, key)],
                                predictions['%.3f_yhat_lower_%s' %(prior, key)], facecolor = color_dict[prior],
                                alpha = 0.3, edgecolor = 'k', linewidth = 0.6)
                            
        # Plot labels
        plt.legend(loc = 2, prop={'size': 10})
        plt.xlabel('Date')
        plt.ylabel('Stock Price (VNĐ)')
        plt.title('Effect of Changepoint Prior Scale')
        plt.show()

    def evaluate_prediction(self, start_date=None, end_date=None, nshares = None):
        
        # Default start date is one year before end of data
        # Default end date is end date of data
        if start_date is None:
            start_date = self.max_date - pd.DateOffset(months=5)
        if end_date is None:
            end_date = self.max_date
        # Training data starts self.training_years years before start date and goes up to start date
        pivot = self.training_sets['ma_30']
        train = self.stock[(pivot['ds'] < start_date.date()) & 
                           (pivot['ds'] > (start_date - pd.DateOffset(years=self.training_years)).date())]
        
        # Testing data is specified in the range
        test = self.stock[(pivot['ds'] >= start_date.date()) & (pivot['ds'] <= end_date.date())]

        model = self.build_model()
        model.fit(train)

        # Make a future dataframe and predictions
        future = model.make_future_dataframe(periods = 365, freq='D')
        future = model.predict(future)

        # Merge predictions with the known values
        test = pd.merge(test, future, on = 'ds', how = 'inner')

        train = pd.merge(train, future, on = 'ds', how = 'inner')

        # Calculate the differences between consecutive measurements
        test['pred_diff'] = test['yhat'].diff()
        test['real_diff'] = test['y'].diff()
        
        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1

        # Correct is when we predicted the correct direction
        test['correct'] = (np.sign(test['pred_diff']) == np.sign(test['real_diff'])) * 1
        
        # Accuracy when we predict increase and decrease
        increase_accuracy = 100 * np.mean(test[test['pred_diff'] > 0]['correct'])
        decrease_accuracy = 100 * np.mean(test[test['pred_diff'] < 0]['correct'])

        # Calculate mean absolute error
        test_errors = abs(test['y'] - test['yhat'])
        test_mean_error = np.mean(test_errors)

        train_errors = abs(train['y'] - train['yhat'])
        train_mean_error = np.mean(train_errors)

        # Calculate percentage of time actual value within prediction range
        test['in_range'] = False

        for i in test.index:
            if (test.ix[i, 'y'] < test.ix[i, 'yhat_upper']) & (test.ix[i, 'y'] > test.ix[i, 'yhat_lower']):
                test.ix[i, 'in_range'] = True

        in_range_accuracy = 100 * np.mean(test['in_range'])

        if not nshares:

            # Date range of predictions
            print('\nPrediction Range: {} to {}.'.format(start_date.date(),
                end_date.date()))

            # Final prediction vs actual value
            print('\nPredicted price on {} = VNĐ{:.2f}.'.format(max(future['ds']).date(), future.ix[len(future) - 1, 'yhat']))
            print('Actual price on    {} = VNĐ{:.2f}.\n'.format(max(test['ds']).date(), test.ix[len(test) - 1, 'y']))

            print('Average Absolute Error on Training Data = VNĐ{:.2f}.'.format(train_mean_error))
            print('Average Absolute Error on Testing  Data = VNĐ{:.2f}.\n'.format(test_mean_error))

            # Direction accuracy
            print('When the model predicted an increase, the price increased {:.2f}% of the time.'.format(increase_accuracy))
            print('When the model predicted a  decrease, the price decreased  {:.2f}% of the time.\n'.format(decrease_accuracy))

            print('The actual value was within the {:d}% confidence interval {:.2f}% of the time.'.format(int(100 * model.interval_width), in_range_accuracy))


             # Reset the plot
            self.reset_plot()
            
            # Set up the plot
            fig, ax = plt.subplots(1, 1)

            # Plot the actual values
            ax.plot(train['ds'], train['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            ax.plot(test['ds'], test['y'], 'ko-', linewidth = 1.4, alpha = 0.8, ms = 1.8, label = 'Observations')
            
            # Plot the predicted values
            ax.plot(future['ds'], future['yhat'], 'navy', linewidth = 2.4, label = 'Predicted');

            # Plot the uncertainty interval as ribbon
            ax.fill_between(future['ds'].dt.to_pydatetime(), future['yhat_upper'], future['yhat_lower'], alpha = 0.6, 
                           facecolor = 'gold', edgecolor = 'k', linewidth = 1.4, label = 'Confidence Interval')

            # Put a vertical line at the start of predictions
            plt.vlines(x=min(test['ds']).date(), ymin=min(future['yhat_lower']), ymax=max(future['yhat_upper']), colors = 'r',
                       linestyles='dashed', label = 'Prediction Start')

            # Plot formatting
            plt.legend(loc = 2, prop={'size': 8}); plt.xlabel('Date'); plt.ylabel('Price VNĐ');
            plt.grid(linewidth=0.6, alpha = 0.6)
                       
            plt.title('{} Model Evaluation from {} to {}.'.format(self.symbol,
                start_date.date(), end_date.date()));
            plt.show();
    
    def plot_stock(self, columns=['Close'], show_data=True, show_volume=False, show_moving_avg = False):
        self.reset_plot()
        colors = ['r', 'b', 'g', 'y', 'c', 'm']
        plt.style.use('seaborn')
        
        if show_data:
            for i, column in enumerate(columns):
                plt.plot(self.stock['ds'], self.stock[column], color=colors[i])

        if show_volume:
            monthly_resampled_volume=self.stock[['ds', 'Volume']].resample('M', on='ds').sum()
            min_max_scaler = MinMaxScaler()
            scaled_volume = min_max_scaler.fit_transform(np.array(monthly_resampled_volume).reshape(-1,1))
            scaled_volume *= max(self.stock['Close'])/2
            plt.bar(monthly_resampled_volume.index, scaled_volume.flatten(), width=10)
        
        if show_moving_avg: 
            for i, (key, moving_avg) in enumerate(self.moving_averages.items()):
                if key == 'lags':
                    continue
                for lag in self.moving_averages['lags']:
                    plt.plot(moving_avg['ds'], moving_avg['ma_%d' %lag], ls='--', label='[%s] MA %d' %(key, lag))
                
        #plt.plot(self.stock['ds'], moving_avg, color='powderblue')
        plt.title('{} on {}'.format(self.symbol, ' '.join(columns)))
        plt.legend(loc='best')
        plt.show()

    def set_moving_averages(self, lags = [30], columns=['Close']):
        # ing_averages= pd.DataFrame()
        moving_averages = dict()
        moving_averages['lags'] = lags

        # with each columns, we calculate the moving average with lags
        for column in columns:
            try:
                data = self.stock[['ds', column]]
                for i, lag in enumerate(lags):
                    data['ma_{}'.format(lag)] = data[column].rolling(lag).mean()
                moving_averages[column] = data
            except Exception as e:
                print('An error occured:')
                print(e)

        self.lags = lags
        
        self.moving_averages = moving_averages

        lags_str =""
        for lag in lags:
            lags_str += str(lag)
        print('Moving averages generated: [{}] on columns [{}]'.format(lags_str, '-'.join(columns)))
        return moving_averages
        
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