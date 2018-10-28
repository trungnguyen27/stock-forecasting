import pandas as pd
import numpy as np
import warnings 
from financial_data import FinancialData as fd 
from stocker_vn import Stocker
warnings.filterwarnings("ignore", category=DeprecationWarning)

stock = Stocker('vic')
# stock.plot_stock()

model, model_data = stock.create_prophet_model(days=90)
# stock.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
stock.changepoint_prior_validation(start_date='2017-01-18', end_date='2018-01-18', changepoint_priors=[0.5, 0.7, 0.9, 1])
# stock.changepoint_prior_scale = 0.5
# stock.evaluate_prediction()
# stock.changepoint_prior_analysis()
# stock.changepoint_prior_scale = 0.7
# stock.evaluate_prediction(nshares=10000)
# stock.predict_future(days=10)
# stock.changepoint_date_analysis(search='vincom')