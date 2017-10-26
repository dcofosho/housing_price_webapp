from flask import Flask, render_template, request, redirect, jsonify, url_for, flash
import httplib2
import json
from flask import make_response
import requests
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

client_data = [[5,17,15],[4,32,22],[8,3,12]]

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

print("Boston housing dataset has {} datapoints with {} variables each".format(*data.shape))

min_price = np.min(prices)
print("Min:"+"\n"+str(min_price))

max_price = np.max(prices)
print("Max:"+"\n"+str(max_price))

median_price = np.median(prices)
print("Median:"+"\n"+str(median_price))

std_price = np.std(prices)
print("Standard Deviation:"+"\n"+str(std_price))

X_train, X_test, y_train, t_test = train_test_split(prices,
						features,
						test_size=0.2)

print("training data length: "+str(X_train.shape[0]))
print("testing data length: "+str(X_test.shape[0]))

#vs.ModelLearning(features, prices)

def performance_metric(y_true, y_predict):
	r2 = r2_score(y_true, y_predict)
	return r2
print("R2 score"+"\n"+str(performance_metric(y_train, y_train)))

def fit_model(X, y):
	regressor = DecisionTreeRegressor()
	parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
	scoring_function = make_scorer(performance_metric,
	 greater_is_better=True)
	cv = ShuffleSplit(n_splits = 10, test_size = 0.20, train_size=None, random_state = 0)
	reg = GridSearchCV(regressor, parameters, scoring=scoring_function, cv=cv)
	reg.fit(X,y)
	return reg.best_estimator_

print("Predicted sales prices for client input data:"+"\n"+str(fit_model(features, prices).predict(client_data)))

def getPrice(rm,lstat,ptr):
	return str(fit_model(features, prices).predict([[rm,lstat,ptr]])).replace("[","$").replace("]", " ").split(".")[0].replace(".", " ")

@app.route('/boston', methods=['GET','POST'])
def homePage():
	if request.method == 'POST':
		x=request.form['rm']
		y=request.form['lstat']
		z=request.form['ptr']
		return "Number of rooms: "+str(x)+"<br>"+"Percentage of neighborhood below poverty line: "+str(y)+"<br>"+"Pupil-Teacher Ratio: "+str(z)+"<br>"+"<br>"+"Given these parameters, the predicted price of a house in boston is "+getPrice(x, y, z)+"<br>"+"<button onClick='history.back()'>Go Back</button>"
	else:
		return render_template("home.html")

@app.route('/bostonapi/<int:rm>/<int:lstat>/<int:ptr>')
def bostonAPI(rm, lstat, ptr):
        return jsonify(rm=rm, lstat=lstat, ptr=ptr, price=getPrice(rm, lstat, ptr))

if __name__ == '__main__':
	app.secret_key='totally_secure_key'
	app.debug = True
	app.run()
