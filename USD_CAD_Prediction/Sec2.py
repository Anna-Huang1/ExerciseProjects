#!/usr/bin/env python
# coding: utf-8

# # Quant Analyst Intern Assessment
# 
# ## Section #2: Multivariate Time Series Analysis
# 
# ### In this section, you will build a Multivariate Time Series Model. Please use Jupyter Notebook for this exercise. You may answer all questions in MARKDOWN inside the Jupyter Notebook.

# 
# ### a. Propose up to five factors that may affect the USD/CAD trading range and explain your reasons for making these assumptions. Also explain the type of data frequency you will use (for example, intraday 5 minutes/daily/weekly).
# 
# *I choose the following four factors:
# WTI crude oil price, xauusd, U.S. retail sales year-on-year, U.S. crude oil exports to Canada.*
# 
# *I think there are not many factors that affect USD/CAD. It is mainly considered from the oil price data and U.S. economic data.*
# 
# *First of all, the Canadian dollar is a commodity currency. Canada’s economic orientation is also dominated by exports. It is one of the important oil exporting countries. Therefore, rising oil prices may put a lot of pressure on the Canadian dollar. At this time, its purchasing power is limited. It will be greatly reduced, and it will depreciate against major currencies. It should be noted that although Canada exports a lot of oil, it also imports a lot of oil from the United States. A considerable part of this is that Canada first exported crude oil to the United States, then import it back from the United States. So I chose the WTI crude oil price and the amount of crude oil exported from the U.S. to Canada.*
# 
# *Secondly, US consumption data, QE, Taper, and fiscal policies will all have an impact on usd/cad. Gold is closely related to the US dollar, so I chose xauusd. Since consumption accounts for a large proportion of the US economy, so from the perspective of consumption data I chose the retail sales year-on-year, and similarly, I can also choose the consumer confidence index instead. Here is a detail, the poor employment data of the United States will drive the Canadian dollar to rise, because the United States is full of employment, and the production of shale oil also increases, the oil prices fall.*
# 
# *The specific choice actually depends on the strategy in different periods. For example, the current international oil prices are rising, but Canadian export oil is selling well, causing usdcad to fall a lot. If the U.S. dollar taper, international oil prices are weak, OPEC increases production capacity, and the U.S. dollar appreciates, the Canadian dollar will weaken relatively, and the logic will become dependent on US consumption data and manufacturing data.*
# 
# *In the actual trading strategy, the higher the data frequency, the better. The FX leverage ratio is very high, so it is trading back and forth in the min range, and no one can bear the fluctuation of the next day.*
# 
# *And I think the best construction of this model should be based on the long-term trend of the U.S. dollar index and the long-term trend of international oil prices, to build a large-scale model of usdcad, and build it with mathematical data from a macro perspective. There is a dual relationship between the surface and the inside called the trend structure in technical theory, that is, any trend in a large level must fall in a certain direction. The long-term trend is the general direction, and the time series modeling can be better in the changing market to capture the high and low points of our trading, so it will be more useful in higher frequency transactions.*
# 
# *However, due to the limited frequency and range of data I can obtain, and in order to simplify the model, the data selected below are all daily frequencies.*

# ### b.	Causality & Statistical Signifiance 
# 
#    i. Utilize the Granger’s Causality Test to test foe causation in the variables that you have selected. Construct a matrix to show the p-values of the variables against one another (i.e., you also need to show that there are possible relationships between one another; not just USD/CAD).   
# 
#    ii. [Test for Statistical Significance Between Time Series] Use the Johassen’s test of cointegration to check statistical significance in your variables. 
# 
# ### c.	[Check for Stationarity] Implement an ADF test to check for stationarity. If it is not stationary, conduct a first-order differencing and re-check for stationarity. Show the ADF p-values for all the selected variables. If first-order differencing is not stationary, conduct a second-order differencing and re-run the ADF test.
# 
# ### d.	[Model Selection] Using fit comparison estimates of AIC, BIC, FPE, and HQIC. Using the Fit comparison estimates to derive the most optimal number of lags for the model.
# 

# ### Data Preprocessing
# 

# In[1]:


# Data Preprocessing

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def Interpolation(data_process):
    
    """
    Interpolation

    Perform arithmetic interpolation on the missing values of a column of data
    """
    
    j = 0
    df_seq = []
    df = data_process.values
    
    # arithmetic interpolation, the value of 0 is also considered here.
    for i in range(len(df)):
        if df[i] != 0:
            diff = (df[i]-df[j])/(i-j)
            data_arange = np.arange(df[j], df[i], diff)
            j = i
            for num in range(len(data_arange)):
                df_seq.append(data_arange[num])
                
        #if i == len(df)-1 and df[i] != 0:
        #    df_seq.append(df[i])
        
        if i == len(df)-1 and df[i] == 0:
            end = df[j] + diff * (i-j)
            data_arange = np.arange(df[j], end, diff)
            for num in range(len(data_arange)):
                df_seq.append(data_arange[num])
            if len(df_seq) != len(df):
                df_seq.append(end)
                      
    return df_seq


data = pd.read_csv('./HistoryData.csv')  # Read data.

data['Date'] = pd.to_datetime(data['Date'])
data.index = data['Date']  # Convert date format

# data.info()

data['WTI'] = data['WTI'].bfill() # There are few missing values in wti, so only mean interpolation is used.

data = data.fillna(0)

# These two are monthly data, and its trend needs to be retained, so use arithmetic interpolation
data['Sales_Rate'] = Interpolation(data['Sales_Rate'])
data['CrudeOilExports_UStoCanada'] = Interpolation(data['CrudeOilExports_UStoCanada'])
data = data.drop(data[(data['Sales_Rate']==0) |(data['CrudeOilExports_UStoCanada']==0)].index)

data.to_csv('./train_data.csv', index = False, header = True)
data.describe()


# In[4]:


# Data Visualization
DailyWTI = plt.figure(figsize = (10, 4))
ax = DailyWTI.add_subplot(111)
ax.set(title = 'WTI_DailyData',
        ylabel = 'Price', xlabel='Date')

plt.plot(data['Date'], data['WTI'])
plt.show()    

DailyXAU_USD = plt.figure(figsize = (10, 4))
ax = DailyXAU_USD.add_subplot(111)
ax.set(title = 'XAU_USD_DailyData',
        ylabel = 'Price', xlabel='Date')

plt.plot(data['Date'], data['XAU_USD'])
plt.show() 

DailySales_Rate = plt.figure(figsize = (10, 4))
ax = DailySales_Rate.add_subplot(111)
ax.set(title = 'Sales_Rate_DailyData',
        ylabel = 'Price', xlabel='Date')

plt.plot(data['Date'], data['Sales_Rate'])
plt.show()    

DailyCrudeOilExports_UStoCanada = plt.figure(figsize = (10, 4))
ax = DailyCrudeOilExports_UStoCanada.add_subplot(111)
ax.set(title = 'CrudeOilExports_UStoCanada_DailyData',
        ylabel = 'Price', xlabel='Date')

plt.plot(data['Date'], data['CrudeOilExports_UStoCanada'])
plt.show() 


# ### Granger Causality Tests
# 
# ##### if p is less than 0.05, Granger causality is considered

# In[5]:


def Grangercausalitytests(data, maxlag_num):
    
    """
    Granger Causality Tests

    if p is less than 0.05, Granger causality is considered
    """
    
    from statsmodels.tsa.stattools import grangercausalitytests
    grangercausalitytests(data, maxlag=maxlag_num)

    
# All variables need to be tested
print('Grangercausalitytests Result between WTI and USD_CAD')
Grangercausalitytests(data[['WTI', 'USD_CAD']], 3)
      
print('\nGrangercausalitytests Result between XAU_USD and USD_CAD')
Grangercausalitytests(data[['XAU_USD', 'USD_CAD']], 3)

print('\nGrangercausalitytests Result between Sales_Rate and USD_CAD')
Grangercausalitytests(data[['Sales_Rate', 'USD_CAD']], 3)
      
print('\nGrangercausalitytests Result between CrudeOilExports_UStoCanada and USD_CAD')
Grangercausalitytests(data[['CrudeOilExports_UStoCanada', 'USD_CAD']], 3)


print('\nGrangercausalitytests Result between WTI and CrudeOilExports_UStoCanada')
Grangercausalitytests(data[['WTI', 'CrudeOilExports_UStoCanada']], 3)
      
print('\nGrangercausalitytests Result between Sales_Rate and CrudeOilExports_UStoCanada')
Grangercausalitytests(data[['Sales_Rate', 'CrudeOilExports_UStoCanada']], 3)
      
print('\nGrangercausalitytests Result between XAU_USD and CrudeOilExports_UStoCanada')
Grangercausalitytests(data[['XAU_USD', 'CrudeOilExports_UStoCanada']], 3)


print('\nGrangercausalitytests Result between WTI and Sales_Rate')
Grangercausalitytests(data[['WTI', 'Sales_Rate']], 3)
      
print('\nGrangercausalitytests Result between XAU_USD and Sales_Rate')
Grangercausalitytests(data[['XAU_USD', 'Sales_Rate']], 3)


print('\nGrangercausalitytests Result between WTI and XAU_USD')
Grangercausalitytests(data[['WTI', 'XAU_USD']], 3)


# ### ADF: Augmented Dickey-Fuller Unit Root  Tests
# 
# ##### Check if it is a stationary series

# In[6]:


def ADF_diff(timeseries, name):
    
    """
    ADF: Augmented Dickey-Fuller unit root test.
    
    Regression: Constant and trend order to include {“c”,”ct”,”ctt”,”nc”}
    1. “c” : constant only (default).
    2. “ct” : constant and trend.
    3. “ctt” : constant, and linear and quadratic trend.
    4. “nc” : no constant, no trend.
    """

    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller as ADF
    
    # Sequence after generating the differencing.
    timeseries_diff1 = timeseries.diff(1)
    timeseries_diff2 = timeseries_diff1.diff(1)

    timeseries_diff1 = timeseries_diff1.fillna(0)
    timeseries_diff2 = timeseries_diff2.fillna(0)

    # ADF unit root test -- ct
    print('Result of ADF--ct Test ')
    timeseries_adf = ADF(timeseries[name].tolist(), regression='ct')
    timeseries_diff1_adf = ADF(timeseries_diff1[name].tolist(), regression='ct')
    timeseries_diff2_adf = ADF(timeseries_diff2[name].tolist(), regression='ct')

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()
    
    # ADF unit root test -- c
    print('Result of ADF--c Test ')
    timeseries_adf = ADF(timeseries[name].tolist(), regression='c')
    timeseries_diff1_adf = ADF(timeseries_diff1[name].tolist(), regression='c')
    timeseries_diff2_adf = ADF(timeseries_diff2[name].tolist(), regression='c')

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()
    
    # ADF unit root test -- nc
    print('Result of ADF--nc Test ')
    timeseries_adf = ADF(timeseries[name].tolist(), regression='nc')
    timeseries_diff1_adf = ADF(timeseries_diff1[name].tolist(), regression='nc')
    timeseries_diff2_adf = ADF(timeseries_diff2[name].tolist(), regression='nc')

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()

# Parse data with date for training   
dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
train_data = pd.read_csv('./train_data.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse)

# Extract the data of each variable for future use
data_WTI = pd.DataFrame(train_data['WTI'])
data_XAU_USD = pd.DataFrame(train_data['XAU_USD'])
data_Sales_Rate = pd.DataFrame(train_data['Sales_Rate'])
data_CrudeOilExports_UStoCanada = pd.DataFrame(train_data['CrudeOilExports_UStoCanada'])
data_USD_CAD = pd.DataFrame(train_data['USD_CAD'])

# ADF Tests
print("Resulty of ADF -- WTI")
ADF_diff(data_WTI, 'WTI')
print("Resulty of ADF -- XAU_USD")
ADF_diff(data_XAU_USD, 'XAU_USD')
print("Resulty of ADF -- Sales_Rate")
ADF_diff(data_Sales_Rate, 'Sales_Rate')
print("Resulty of ADF -- CrudeOilExports_UStoCanada")
ADF_diff(data_CrudeOilExports_UStoCanada, 'CrudeOilExports_UStoCanada')
print("Resulty of ADF -- USD_CAD")
ADF_diff(data_USD_CAD, 'USD_CAD')


# ### Fit Comparison Estimates of AIC, BIC, FPE, and HQIC.
# 
# ##### Using fit comparison estimates of AIC, BIC, FPE, and HQIC. 
# ##### Using the Fit comparison estimates to derive the most optimal number of lags for the model.

# In[7]:


def order_selection(timeseries):
    
    """
    Select the optimal lag order.
    
    Use Vector Auto Regression (VAR) Model to select order.
    The var model can pass a maximum number of lags and the order criterion to use for lag order selection.
    """
    
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    
    # Vector Auto Regression (VAR) Model
    var_model = VAR(timeseries)

    #Lag order selection
    order = var_model.select_order(10)
    print(order.summary())
    
    #var_results = var_model.fit(maxlags=5, ic='aic')
    #var_results.summary()


# Use original data to select order
order_selection(train_data)

# data after differencing
train_data_diff = train_data

data_WTI_diff1 = data_WTI.diff(1)
data_WTI_diff1 = data_WTI_diff1.fillna(0)
train_data_diff['WTI'] = data_WTI_diff1

data_XAU_USD_diff1 = data_XAU_USD.diff(1)
data_XAU_USD_diff1 = data_XAU_USD_diff1.fillna(0)
train_data_diff['XAU_USD'] = data_XAU_USD_diff1

data_Sales_Rate_diff1 = data_Sales_Rate.diff(1)
data_Sales_Rate_diff2 = data_Sales_Rate_diff1.diff(1)
data_Sales_Rate_diff2 = data_Sales_Rate_diff2.fillna(0)
train_data_diff['Sales_Rate'] = data_Sales_Rate_diff2

data_CrudeOilExports_UStoCanada_diff1 = data_CrudeOilExports_UStoCanada.diff(1)
data_CrudeOilExports_UStoCanada_diff2 = data_CrudeOilExports_UStoCanada_diff1.diff(1)
data_CrudeOilExports_UStoCanada_diff2 = data_CrudeOilExports_UStoCanada_diff2.fillna(0)
train_data_diff['CrudeOilExports_UStoCanada'] = data_CrudeOilExports_UStoCanada_diff2

data_USD_CAD_diff1 = data_USD_CAD.diff(1)
data_USD_CAD_diff1 = data_USD_CAD_diff1.fillna(0)
train_data_diff['USD_CAD'] = data_USD_CAD_diff1

# Use data after differencing to select order
order_selection(train_data_diff)


# ### Johassen’s test of cointegration

# In[8]:


def coint_johansen(timeseries):
    
    """
    Johassen’s test of cointegration
    
    Use the Johassen’s test of cointegration to check statistical significance in your variables. 
    """
    
    from statsmodels.tsa.vector_ar import vecm
    
    # Johansen cointegration test, situation setting = continuous features
    jres1 = vecm.coint_johansen(timeseries, det_order=0, k_ar_diff=1)

    # View johansen cointegration test results
    johansen_result = vecm.select_coint_rank(timeseries,det_order=0, k_ar_diff=1, signif=0.1)
    print(johansen_result.summary())

    # Choice of rank
    print(johansen_result.rank)

    j_name=np.array([['Order'], ['trace statistics'], ['CV 90%'],  ['CV 95%'], ['CV 99%'] ] )
    print(j_name)
    
    j_order = jres1.ind 
    print(j_order) # Order of eigenvalues 
    
    j_lr1 = jres1.lr1 
    print(j_lr1) # trace statistics
    
    j_cvt = jres1.cvt 
    print(j_cvt) # Critical values (90%, 95%, 99%) of trace statistic

    # j_result = np.vstack(( j_order,  j_lr1, j_cvt))
    # print(j_result)

    pd.DataFrame(jres1.r0t).plot(kind='bar')  # Residuals for Δ
    
    
coint_johansen(train_data_diff)


# ### e.	[Forecast] Utilizing your model, forecast the daily close for the next five days, include a 75% confidence interval. Derive the USD/CAD closing price range for the next 5 days (i.e., Min & Max).
# 

# ### VECM Model

# In[346]:


# Use VECM Model to Forecast
mod = vecm.VECM(train_data_diff, k_ar_diff=1, coint_rank=5, freq='B', deterministic="ci")
res = mod.fit()
print(res.summary())


# In[347]:


"""
Notes:

forecast - ndarray (steps x neqs) or three ndarrays

In case of a point forecast: each row of the returned ndarray represents the forecast of the neqs variables for a specific period. 
The first row (index [0]) is the forecast for the next period, the last row (index [steps-1]) is the steps-periods-ahead- forecast.
"""

# An image of data that predicts one step forward (showing confidence interval, confidence interval 75%)
res.plot_forecast(steps=5,plot_conf_int=True,alpha=0.75)

# Predict the value one step forward, with 75% confidence
res.predict(steps=5, alpha=0.75)


# In[ ]:





# In[ ]:




