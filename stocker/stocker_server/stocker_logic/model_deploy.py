import pickle
from stock_model import SModel
from financial_data import FinancialData

stock = FinancialData(ticker="VIC")
stock.describe_stock()

lags = [1]
days = 7

mas = stock.get_moving_averages(lags = lags, columns=['Close'])

smodel = SModel(stock=stock)

for (key, ma) in mas.items():
    model = smodel.get_trained_model(ma)
    #serializing our model to a file called model.pkl
    pickle.dump(model, open("prophet_model_%s.pkl" %key,"wb"))

#loading a model from a file called model.pkl
model = pickle.load(open("prophet_model_MA_1_Close.pkl","b"))