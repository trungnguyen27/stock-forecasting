from flask import Flask, jsonify
from flask import request
import pickle
from global_configs import configs
from stocker_server.stocker_logic.stock_model import SModel
from stocker_server.stock_database.financial_data import FinancialData


#code which helps initialize our server
app =  Flask(__name__)
model_path = configs['exported_model_path']

model = pickle.load(open("%s/prophet_model_MA_1_Close.pkl" %model_path, "rb"))
stock = FinancialData(ticker="VIC")

@app.route('/describe', methods=["GET"])
def describe():
    result = stock.describe_stock()
    return jsonify(result)

@app.route('/price/<string:ticker>', methods = ["GET"])
def get_stock_data(ticker):
    return stock.get_data().to_json(orient="records")
#defining a /hello route for only post requests
@app.route('/future', methods=['GET'])
def index():
     # Future dataframe with specified number of days to predict
    predicted = model.make_future_dataframe(periods=30, freq='D')
    predicted = model.predict(predicted)
    # Only concerned with future dates
    future = predicted[predicted['ds'] >= stock.max_date.date()]

    # Remove the weekends
    #future = self.remove_weekends(future)

    future = future.dropna()
    
    # Calculate whether increase or not
    predicted['diff'] = predicted['yhat'].diff()
    future['diff'] = future['yhat'].diff()

    predicted = predicted.dropna()

    # Find the prediction direction and create separate dataframes
    future['direction'] = (future['diff'] > 0) * 1
    predicted['direction'] = (predicted['diff']> 0) *1
    predicted['y'] = predicted['yhat']
    result = dict()
    result['predicted'] = predicted.to_json()
    result['future'] = future.to_json()
    
    return predicted[["y", "ds"]].head().to_json()

# if __name__ == '__main__':
#     app.run(debug=True)