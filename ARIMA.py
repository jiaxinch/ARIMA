import winsound
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
 return datetime.strptime(x, '%Y %m %d %H')


def evaluate_arima_model(X, arima_order):
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
	    model = ARIMA(history, order=arima_order)
	    model_fit = model.fit(disp=0)
	    output = model_fit.forecast()
	    yhat = output[0]
	    predictions.append(yhat)
	    obs = test[t]
	    history.append(obs)
    error = mean_squared_error(test, predictions)
    return error


def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    best_order=(0,0,0)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                        best_order=order
                    #print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    #print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_order

def out_put(X, arima_order):
    model = ARIMA(X, order=arima_order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    return output

dataset = read_csv('final_data.csv', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')
dataSet =[]
bestOrders=[]
outputs =[]

num_of_column = len(values[0])
for i in range(0,num_of_column):
    temp=[]
    for x in values:
        temp.append(x[i])
    dataSet.append(temp)
# for x in values:
#     dataSet.append(x[49])
p_values = [0,1,2,4,6,8,10]#[0, 1, 2, 4, 6, 8, 10]
d_values = [0,1,2]#range(0, 3)
q_values = [0,1,2]#range(0, 3)

for x in dataSet:
    bestOrders.append(evaluate_models(x, p_values, d_values, q_values))
for i in range (0, len(dataSet)):
    outputs.append(dataSet[i],bestOrders[i])
print(outputs)

# order=(0,0,0)
# output =out_put(twitter_pro,order)
# print(output[0])
winsound.Beep(500,1000)