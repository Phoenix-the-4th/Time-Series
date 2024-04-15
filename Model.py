# pip install pandas_datareader requests
# pip install setuptools
# pip install pandas
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy
import sklearn.linear_model
import sklearn.metrics

# Reading Voltas shares from yahoo finance server    
# shares_df = pdr.DataReader('AAPL', 'yahoo', start='2023-01-01', end='2023-12-31')
# Look at the data read
# print(shares_df)


# Read the CSV file into a pandas DataFrame
data = pd.read_csv('trial.csv')

# Now you can work with the data using pandas functions
data['Date'] = (pd.to_datetime(data['Date']))
data['Avg'] = (data['Open'] + data['Close'])/2
print(data.head())  # Display the first few rows

# plt.plot(data['Date'], data['Open'])
# plt.plot(data['Date'], data['Close'])
# plt.plot(data['Date'], data['High'])
# plt.plot(data['Date'], data['Low'])
# plt.plot(data['Date'], data['Avg'])
# plt.xlabel('Date')
# plt.ylabel('INR')
# plt.legend(['Open','Close','High','Low', 'Avg'])
# plt.show()


print("modelling")
k = 3
# creating feature variables 
kset = data['Avg'][0:k]
print('Here', str(len(data['Avg'])))
X = []
for i in range(0, len(data['Avg'])-k):
    X.append(data['Avg'][i:i+k])
y = data['Avg'].shift(-k).dropna()

# creating train and test sets 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

# creating a regression model 
model = sklearn.linear_model.LinearRegression() 
# fitting the model 
model.fit(X_train,y_train)
# making predictions 
predictions = model.predict(X_test) 
# model evaluation 
print('mean_squared_error : ', sklearn.metrics.mean_squared_error(y_test, predictions))
tkeys = y_test.keys()
xax = [i for i in range(len(tkeys))]
yax = [y_test[tkeys[i]] - predictions[i] for i in xax]
plt.plot(xax, yax)
plt.show()