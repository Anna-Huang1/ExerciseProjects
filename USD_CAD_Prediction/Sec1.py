#!/usr/bin/env python
# coding: utf-8

# # Quant Analyst Intern Assessment
# 
# ## Section #1: Univariate Time Series Analysis 
# 
# ### 1. In this section, you will build an ARIMA Model for USD/CAD utilizing daily data. Please use Jupyter Notebook for this exercise. You may answer all questions in MARKDOWN inside the Jupyter Notebook.
# 
# #### a.	[ Variable d ] What is the optimal order of differencing in the ARIMA model? Explain how you derived the variable d by using first using ACF and cross-checking with ADF/KPSS/ PP tests. You may plot the relevant charts and tables in the Jupyter cell. 

# ### Data Processing

# In[513]:


# Data Processing
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

FXdata_usd_cad = pd.read_csv('./USD_CAD_2year.csv') # Read USD/CAD data.

FXdata_usd_cad['Date'] = pd.to_datetime(FXdata_usd_cad['Date'])
FXdata_usd_cad.index = FXdata_usd_cad['Date']  # Convert date format

DailyFX = plt.figure(figsize = (10, 4))
ax = DailyFX.add_subplot(111)
ax.set(title = 'FX_DailyData_USD/CAD',
        ylabel = 'Price', xlabel='Date')

plt.plot(FXdata_usd_cad['Date'], FXdata_usd_cad['Price'])
plt.show()    # Trend view of USD/CAD price


# In[514]:


def generate_price():
    
    """
    Method: model training and test data
    Data: USD/CAD 1/1/2020-10/8/2021
    Source: investing.com
    """
    
    import pandas as pd
    
    # Read the required data
    FX_data = pd.read_csv('./USD_CAD_2year.csv')
    data = FX_data.iloc[:, :2]
    
    # Split the training and test data, ratio 7:3
    num_row = int(len(data)*0.7)
    data_train = data.iloc[:num_row, :]
    data_test = data.iloc[num_row:, :]
    
    # Store the data for modeling
    data_train.to_csv('./data_train.csv', index = False, header = True)
    data_test.to_csv('./data_test.csv', index = False, header = True)

    
generate_price()


# ### ADF: Augmented Dickey-Fuller Unit Root  Tests
# 
# ##### Check if it is a stationary series

# In[516]:


def ADF_diff(timeseries):
    
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
    timeseries_adf = ADF(timeseries['Price'].tolist(), regression='ct')
    timeseries_diff1_adf = ADF(timeseries_diff1['Price'].tolist(), regression='ct')
    timeseries_diff2_adf = ADF(timeseries_diff2['Price'].tolist(), regression='ct')

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
    timeseries_adf = ADF(timeseries['Price'].tolist(), regression='c')
    timeseries_diff1_adf = ADF(timeseries_diff1['Price'].tolist(), regression='c')
    timeseries_diff2_adf = ADF(timeseries_diff2['Price'].tolist(), regression='c')

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
    timeseries_adf = ADF(timeseries['Price'].tolist(), regression='nc')
    timeseries_diff1_adf = ADF(timeseries_diff1['Price'].tolist(), regression='nc')
    timeseries_diff2_adf = ADF(timeseries_diff2['Price'].tolist(), regression='nc')

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse)
ADF_diff(price_train)


# In[101]:


"""
def KPSS_test(timeseries):
    
    # KPSS: Kwiatkowski–Phillips–Schmidt–Shin unit root test
    
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import kpss
    
    # Sequence after generating the differencing.
    timeseries_diff1 = timeseries.diff(1)
    timeseries_diff2 = timeseries_diff1.diff(1)

    timeseries_diff1 = timeseries_diff1.fillna(0)
    timeseries_diff2 = timeseries_diff2.fillna(0)
    
    # KPSS unit root test
    print('Result of KPSS Test ')
    timeseries_kpss = kpss(timeseries)
    timeseries_diff1_kpss = kpss(timeseries_diff1)
    timeseries_diff2_kpss = kpss(timeseries_diff2)

    print('timeseries_kpss : ', timeseries_kpss)
    print('timeseries_diff1_kpss : ', timeseries_diff1_kpss)
    print('timeseries_diff2_kpss : ', timeseries_diff2_kpss)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()
    
    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse)
KPSS_test(price_train)

"""


# #### The optimal order of differencing in the ARIMA model is 1.

# ### ACF: Auto-Correlation Function
# 
# ##### View the auto-correlation and partial auto-correlation of data

# In[139]:


def autocorrelation(timeseries, lags):
    
    """
    View the auto-correlation and partial auto-correlation of data
    
    ACF: auto-correlation function which gives us values of auto-correlation of any series with its lagged values.
    PACF: partial auto-correlation function which finds correlation of the residuals with the next lag value.
    """
    
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # ACF of data
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphicsics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    
    # DCF of data
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse)

price_train_diff = price_train.diff(1)
price_train_diff = price_train_diff.fillna(0)  # optimal order of differencing is one.

autocorrelation(price_train_diff, 20)


# ### IC: Information Criterion
# 
# ##### determine p and q values

# In[110]:


def IC(timeseries):
    
    """
    IC(Information Criterion): to determine p and q values
    
    AIC: Akaike Information Criterion
    BIC: Bayesian Information Criterion
    """
    
    import statsmodels.api as sm
    
    # calculate AIC and BIC.
    IC_evaluate = sm.tsa.arma_order_select_ic(timeseries, ic=['aic', 'bic'], trend='nc', max_ar=6,
                                            max_ma=6)
    print('AIC', IC_evaluate.aic_min_order)
    print('BIC', IC_evaluate.bic_min_order)

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

price_train_diff = price_train.diff(1)
price_train_diff = price_train_diff.fillna(0)  # optimal order of differencing is one.

IC(price_train_diff)


# ##### From acf and pacf, p and q should be 1. But this result is more subjective.
# 
# ##### The AIC/BIC results are more objective, the first is 3 and 2, the second is 0 and 1.
# 
# ##### When using the ARIMA model, it need to build a model to try and determine the optimal parameters

# ### ARIMA_Model
# 

# In[129]:


def ARIMA_Model(timeseries, order):
    
    """
    ARIMA Model
    
    Notes:
    1. Time series data should choose the stationary series after differencing.
    2. Order includes p, d, q. 
        p is the order of AR. 
        d is the order of differencing. 
        q is the order of MA.
    3. Pandas will automatically determine the frequency of the data.
    """
    
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA

    # Fit ARIMA Model
    model = ARIMA(timeseries, order=order)
    model_fit = model.fit()
    print(model_fit.summary())  # Summary of ARIMA

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

price_train_diff = price_train.diff(1)
price_train_diff = price_train_diff.fillna(0)  # optimal order of differencing is one.

# Model training
ARIMA_Model(price_train_diff, (0,1,1))
ARIMA_Model(price_train_diff, (1,1,1))
ARIMA_Model(price_train_diff, (3,1,2))


# ##### Order includes p, d, q. 
#     p is the order of AR. 
#     d is the order of differencing. 
#     q is the order of MA.
# 
# ##### The smaller the AIC/BIC, the better.
# 
# ##### The coef is the coefficient of our model. 
# 
# ##### If the p value is less than 0.05, it is valid. If it is greater than 0.05, we need to change the differencing order.

# ### Residual Analysis

# In[172]:


def residual_errors(timeseries,order):
    
    """
    Residual sequence of model fitting
    
    The residual sequence is the sequence obtained by subtracting the fitting sequence on the training data from the original sequence of the training data. 
    The more the sequence conforms to the random error distribution (normal distribution with a mean value of 0), the better the model fits.
    Otherwise, it means that there are still some factors that the model fails to consider.
    """
    
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA

    # ARIMA Model
    model = ARIMA(timeseries, order=order)
    model_fit = model.fit()
    
    # Actual vs Fitted
    train_predict = model_fit.predict()
    plt.plot(train_predict, color='red', label='fit_seq')
    plt.plot(timeseries, color='blue', label='price_seq_train')
    plt.legend(loc='best')
    plt.show()

    # Plot charts of Residual Sequence
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2,figsize = (20, 5))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()
    

dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

price_train_diff = price_train.diff(1)
price_train_diff = price_train_diff.fillna(0)  # optimal order of differencing is one.

# Residual Errors of Model 
residual_errors(price_train_diff, (0,1,1))


# ##### The density curve of the residuals is very close to the nearly normal distribution, indicating that the model is effective.
# ##### However, it can be seen from the fitting curve that we only fit the mean trend, but the fluctuation trend is not captured.
# 
# #### Next, we consider decomposing the trend and residual series to model.

# ### Decomposing
# 
# #### Decompose the trend and residual series to model

# In[262]:


def decomposing(timeseries):
    
    """
    Decomposing 
    
    Use seasonal_decompose to decompose the trend and residual series to model.
    
    """

    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose

    # decompose the trend and residual series to model.
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # plot relevant charts.
    plt.figure(figsize=(16, 12))
    plt.subplot(411)
    plt.plot(timeseries, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonarity')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

decomposing(price_train)


# ### ADF: Augmented Dickey-Fuller Unit Root  Tests
# 
# ##### Check if it is a stationary series

# In[273]:


def ADF_diff_trend(timeseries):
    
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
    timeseries_adf = ADF(timeseries['trend'].tolist(), regression='ct')
    timeseries_diff1_adf = ADF(timeseries_diff1['trend'].tolist(), regression='ct')
    timeseries_diff2_adf = ADF(timeseries_diff2['trend'].tolist(), regression='ct')

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
    timeseries_adf = ADF(timeseries['trend'].tolist(), regression='c')
    timeseries_diff1_adf = ADF(timeseries_diff1['trend'].tolist(), regression='c')
    timeseries_diff2_adf = ADF(timeseries_diff2['trend'].tolist(), regression='c')

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
    timeseries_adf = ADF(timeseries['trend'].tolist(), regression='nc')
    timeseries_diff1_adf = ADF(timeseries_diff1['trend'].tolist(), regression='nc')
    timeseries_diff2_adf = ADF(timeseries_diff2['trend'].tolist(), regression='nc')

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse)

decomposition = seasonal_decompose(price_train)
trend = decomposition.trend
seasonal = decomposition.seasonal

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)

trend = pd.DataFrame(trend)
trend.drop(trend.head(2).index,inplace=True) 
trend.drop(trend.tail(2).index,inplace=True) 
ADF_diff_trend(trend)


# In[274]:


def ADF_diff_residual(timeseries):
    
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
    timeseries_adf = ADF(timeseries['resid'].tolist(), regression='ct')
    timeseries_diff1_adf = ADF(timeseries_diff1['resid'].tolist(), regression='ct')
    timeseries_diff2_adf = ADF(timeseries_diff2['resid'].tolist(), regression='ct')

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
    timeseries_adf = ADF(timeseries['resid'].tolist(), regression='c')
    timeseries_diff1_adf = ADF(timeseries_diff1['resid'].tolist(), regression='c')
    timeseries_diff2_adf = ADF(timeseries_diff2['resid'].tolist(), regression='c')

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
    timeseries_adf = ADF(timeseries['resid'].tolist(), regression='nc')
    timeseries_diff1_adf = ADF(timeseries_diff1['resid'].tolist(), regression='nc')
    timeseries_diff2_adf = ADF(timeseries_diff2['resid'].tolist(), regression='nc')

    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)

    plt.figure(figsize=(12, 8))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse)

decomposition = seasonal_decompose(price_train)
seasonal = decomposition.seasonal
residual = decomposition.resid

seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

residual = pd.DataFrame(residual)
residual.drop(residual.head(2).index,inplace=True) 
residual.drop(residual.tail(2).index,inplace=True) 

ADF_diff_residual(residual)


# ### ACF: Auto-Correlation Function
# 
# ##### View the auto-correlation and partial auto-correlation of data

# In[277]:


def autocorrelation(timeseries, lags):
    
    """
    View the auto-correlation and partial auto-correlation of data
    
    ACF: auto-correlation function which gives us values of auto-correlation of any series with its lagged values.
    PACF: partial auto-correlation function which finds correlation of the residuals with the next lag value.
    """
    
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # ACF of data
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    
    # DCF of data
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse)

decomposition = seasonal_decompose(price_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

trend = pd.DataFrame(trend)
trend.drop(trend.head(2).index,inplace=True) 
trend.drop(trend.tail(2).index,inplace=True)

residual = pd.DataFrame(residual)
residual.drop(residual.head(2).index,inplace=True) 
residual.drop(residual.tail(2).index,inplace=True) 

autocorrelation(trend, 40)

autocorrelation(residual, 20)


# ### IC: Information Criterion
# 
# ##### determine p and q values

# In[278]:


def IC(timeseries):
    
    """
    IC(Information Criterion): to determine p and q values
    
    AIC: Akaike Information Criterion
    BIC: Bayesian Information Criterion
    """
    
    import statsmodels.api as sm
    
    # calculate AIC and BIC.
    IC_evaluate = sm.tsa.arma_order_select_ic(timeseries, ic=['aic', 'bic'], trend='nc', max_ar=6,
                                            max_ma=6)
    print('AIC', IC_evaluate.aic_min_order)
    print('BIC', IC_evaluate.bic_min_order)

    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

decomposition = seasonal_decompose(price_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

trend = pd.DataFrame(trend)
trend.drop(trend.head(2).index,inplace=True) 
trend.drop(trend.tail(2).index,inplace=True) 

residual = pd.DataFrame(residual)
residual.drop(residual.head(2).index,inplace=True) 
residual.drop(residual.tail(2).index,inplace=True) 

IC(trend)
IC(residual)


# ### ARIMA_Model

# In[319]:


def ARIMA_Model(timeseries, order):
    
    """
    ARIMA Model
    
    Notes:
    1. Time series data should choose the stationary series after differencing.
    2. Order includes p, d, q. 
        p is the order of AR. 
        d is the order of differencing. 
        q is the order of MA.
    3. Pandas will automatically determine the frequency of the data.
    """
    
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA

    # Fit ARIMA Model
    model = ARIMA(timeseries, order=order)
    model_fit = model.fit()
    print(model_fit.summary())  # Summary of ARIMA

    return model_fit
    
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

decomposition = seasonal_decompose(price_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

trend = pd.DataFrame(trend)
trend.drop(trend.head(2).index,inplace=True) 
trend.drop(trend.tail(2).index,inplace=True) 

residual = pd.DataFrame(residual)
residual.drop(residual.head(2).index,inplace=True) 
residual.drop(residual.tail(2).index,inplace=True) 

# Model training -- trend
trend_model = ARIMA_Model(trend, (1,0,4))
# trend_model = ARIMA_Model(trend, (6,0,2))

# Model training -- residual
residual_model = ARIMA_Model(residual, (0,0,2))
# residual_model = ARIMA_Model(residual, (1,0,1))


# ### Residual Analysis

# In[280]:


def residual_errors(timeseries,order):
    
    """
    Residual sequence of model fitting
    
    The residual sequence is the sequence obtained by subtracting the fitting sequence on the training data from the original sequence of the training data. 
    The more the sequence conforms to the random error distribution (normal distribution with a mean value of 0), the better the model fits.
    Otherwise, it means that there are still some factors that the model fails to consider.
    """
    
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA

    # ARIMA Model
    model = ARIMA(timeseries, order=order)
    model_fit = model.fit()
    
    # Actual vs Fitted
    train_predict = model_fit.predict()
    plt.plot(train_predict, color='red', label='fit_seq')
    plt.plot(timeseries, color='blue', label='price_seq_train')
    plt.legend(loc='best')
    plt.show()

    # Plot charts of Residual Sequence
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2,figsize = (20, 5))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()
    

dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

decomposition = seasonal_decompose(price_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

trend = pd.DataFrame(trend)
trend.drop(trend.head(2).index,inplace=True) 
trend.drop(trend.tail(2).index,inplace=True)

residual = pd.DataFrame(residual)
residual.drop(residual.head(2).index,inplace=True)
residual.drop(residual.tail(2).index,inplace=True) 

# Residual Errors of Model 
residual_errors(trend, (1,0,4))
residual_errors(residual, (0,0,2))


# #### It can be seen that our model fits very well. Next, we will make predictions.

# ### Prediction

# In[379]:


# trend prediction
pred_trend = trend_model.get_prediction(start='2021-10-11', end='2021-10-15', dynamic=False)
pred_trend_ci = pred_trend.conf_int(alpha=0.25)

# residual prediction
pred_residual = residual_model.get_prediction(start='2021-10-11', end='2021-10-15', dynamic=False)
pred_residual_ci = pred_residual.conf_int(alpha=0.25)

# seasonal prediction
pred_seasonal = pd.DataFrame(seasonal['2020-01-07':'2020-01-13'])
pred_seasonal.index = pd.Series(['2021-10-11', '2021-10-12', '2021-10-13', '2021-10-14', '2021-10-15']).apply(lambda dates: datetime.strptime(dates, '%Y-%m-%d'))
pred_seasonal['seasonal_copy'] = pred_seasonal['seasonal']

# price prediction
pred_residual_ci.columns = ['lower price','upper price']
pred_trend_ci.columns = ['lower price','upper price']
pred_seasonal.columns = ['lower price','upper price']
predict_seq = pred_residual_ci + pred_trend_ci + pred_seasonal
print(predict_seq)


# #### Alternate method: auto_arima

# In[ ]:


"""

Partial parameter analysis of auto_arima:
            1.start_p: the starting value of p, the order of the autoregressive ("AR") model (or the number of lag times), must be a positive integer
            2. start_q: the initial value of q, the order of the moving average (MA) model. Must be a positive integer.
            3. max_p: the maximum value of p, which must be a positive integer greater than or equal to start_p.
            4.max_q: the maximum value of q, which must be a positive integer greater than start_q
            5.seasonal: Is it suitable for seasonal ARIMA. The default is correct. Note that if season is true and m == 1, then season will be set to False.
            6.stationary: Whether the time series is stationary and whether d is zero.
            6. information_criterion: Information criterion is used to select the best ARIMA model. One of (‘aic’, ‘bic’, ‘hqic’, ‘oob’)
            7.alpha: The test significance of the test level, the default is 0.05
            8.test: If stationary is false and d is None, the type of unit root test used to detect stationarity. The default is ‘kpss’; it can be set to adf
            9.n_jobs: The number of models fitted in parallel in the grid search (stepwise = False). The default value is 1, but -1 can be used to mean "as much as possible".
            10.suppress_warnings: Many warnings may be thrown in statsmodel. If suppress_warnings is true, then all warnings from ARIMA will be suppressed
            11.error_action: If ARIMA cannot be matched for some reason, you can control the error handling behavior. (warn, raise, ignore, trace)
            12.max_d: the maximum value of d, that is, the maximum number of non-seasonal differences. Must be a positive integer greater than or equal to d.
            13.trace: Whether to print suitable status. If the value is False, no debugging information will be printed. If the value is true, some will be printed


import pmdarima as pm
# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(price_train, start_p=0, start_q=2,
                         information_criterion='aic',
                         test='adf',
                         max_p=6, max_q=6, m=5,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
smodel.summary()


Notes：
The stepwise parameter here, the default value is True, 
which means that the stepwise algorithm is used to select the best parameter combination.
It will be much faster than calculating all parameter combinations, and it will hardly overfit. 
Of course, it is possible to ignore the optimal parameter combination.
So if you want the model to automatically calculate all parameter combinations, 
and then choose the best, you can set stepwise to False.
"""


# ### Rolling forecast

# In[501]:


# Train the model, same as above

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_train = pd.read_csv('./data_train.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

decomposition = seasonal_decompose(price_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

trend = pd.DataFrame(trend)
trend.drop(trend.head(2).index,inplace=True)
trend.drop(trend.tail(2).index,inplace=True) 

residual = pd.DataFrame(residual)
residual.drop(residual.head(2).index,inplace=True)
residual.drop(residual.tail(2).index,inplace=True) 

# Model training -- trend
trend_model = ARIMA(trend, order=(1,0,4))
trend_model_fit = trend_model.fit()

# Model training -- residual
residual_model = ARIMA(residual, order=(0,0,2))
residual_model_fit = residual_model.fit()


# In[517]:


# Process the test set data
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d')
price_test = pd.read_csv('./data_test.csv', parse_dates=['Date'],
                                 index_col='Date', date_parser=dateparse) 

decomposition_test = seasonal_decompose(price_test)
trend_test = decomposition_test.trend
seasonal_test = decomposition_test.seasonal
residual_test = decomposition_test.resid

trend_test = trend_test.fillna(0)
seasonal_test = seasonal_test.fillna(0)
residual_test = residual_test.fillna(0)

trend_test = pd.DataFrame(trend_test)
trend_test.drop(trend_test.head(2).index,inplace=True) 
trend_test.drop(trend_test.tail(2).index,inplace=True) 

residual_test = pd.DataFrame(residual_test)
residual_test.drop(residual_test.head(2).index,inplace=True) 
residual_test.drop(residual_test.tail(2).index,inplace=True) 


# In[518]:


# seasonal prediction
pred_seasonal = pd.DataFrame(seasonal['2020-01-02':'2020-01-08'])
pred_seasonal['seasonal_copy'] = pred_seasonal['seasonal']
pred_seasonal.columns = ['lower price','upper price']

trend_forecast_data = trend
residual_forecast_data = residual

# use for-loop to predict trend and residual data in the future by adding new daily data.
for i in range(len(trend_test)):
    if (i+1)%5 == 0:
            trend_NewModel = ARIMA(trend_forecast_data, order=(1,0,4))
            trend_NewModel_fit = trend_NewModel.fit()
            pred_trend_NewModel = trend_NewModel_fit.get_forecast(5)
            pred_trend_NewModel_ci = pred_trend_NewModel.conf_int(alpha=0.25)
        
            residual_NewModel = ARIMA(residual_forecast_data, order=(0,0,2))
            residual_NewModel_fit = residual_NewModel.fit()
            pred_residual_NewModel = residual_NewModel_fit.get_forecast(5)
            pred_residual_NewModel_ci = pred_residual_NewModel.conf_int(alpha=0.25)
        
            pred_seasonal.index = pred_trend_NewModel_ci.index
            pred_residual_NewModel_ci.columns = ['lower price','upper price']
            pred_trend_NewModel_ci.columns = ['lower price','upper price']
        
            predict_new_seq = pred_trend_NewModel_ci + pred_residual_NewModel_ci + pred_seasonal
            print(predict_new_seq)
                
    cur_trend = trend_test.iloc[i:i+1, :1]
    trend_forecast_data = trend_forecast_data.append(cur_trend, ignore_index=False)
    
    cur_residual = residual_test.iloc[i:i+1, :1]
    residual_forecast_data = residual_forecast_data.append(cur_residual, ignore_index=False)
 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




