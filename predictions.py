#%%
import pandas as pd
import datetime
import pandas_datareader as web
from pandas import Series, DataFrame

start = datetime.datetime(2017, 10, 30)
end = datetime.datetime(2019, 9, 6)

df = web.DataReader("CGC", 'yahoo', start, end)
df.tail()

#%% Rolling Mean and Plotting
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

import matplotlib.pyplot as plt
from matplotlib import style

import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

style.use('ggplot')

close_px.plot(label='GCG')
mavg.plot(label='mavg')
plt.legend()

#%% Return Deviation
rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')


#%% Competitor Analysis
dfcomp = web.DataReader(["CGC", "TLRY"], 'yahoo', start=start, end=end) ['Adj Close']
dfcomp.tail()

#%% Correlation Analysis
retscomp = dfcomp.pct_change()
corr = retscomp.corr()
corr

#%% ScatterPlot to view return distribution
plt.scatter(retscomp.CGC, retscomp.TLRY)
plt.xlabel('Returns CGC')
plt.ylabel('Returns TLRY')


#%% Heat map to visualize correlation ranges between Stocks
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)


#%%
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
    )

#%% High-Low Pct and PCT change
dfreg = df.loc[:, ['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0


#%% Pre-processing and Cross Validation
import math
import numpy as np
from sklearn import preprocessing

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

#%% Model Generation
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


X_train = X
X_test = X
y_train = y
y_test = y

# Linear Regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

#Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

#%% Evaluation
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)

#%%
print('The linear regression confidence is:', confidencereg)
print('The quadratic regression 2 confidence is:', confidencepoly2)
print('The quadratic regression 3 confidence is:',  confidencepoly3)
print('The knn regression confidence is:', confidenceknn)

#%%
forecast_set_knn = clfknn.predict(X_lately)
dfreg['Forecast'] = np.nan


#%%
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set_knn:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)] +[i]

dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#%%
