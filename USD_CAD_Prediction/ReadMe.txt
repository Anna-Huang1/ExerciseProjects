This is a project about USD/CAD Price Prediction by Time Series.




Here are the questions:

Section #1: Univariate Time Series Analysis 单变量时间序列分析
1.	In this section, you will build an ARIMA Model for USD/CAD utilizing daily data. Please use Jupyter Notebook for this exercise. 
You may answer all questions in MARKDOWN inside the Jupyter Notebook.
在本节中，您将利用每日数据为美元/加元构建 ARIMA 模型。请使用 Jupyter Notebook 进行此练习。您可以在 Jupyter Notebook 内的 MARKDOWN 中回答所有问题。

a.	[Variable d] What is the optimal order of differencing in the ARIMA model? Explain how you derived the variable d by using first using ACF and cross-checking with ADF/KPSS/ PP tests. 
You may plot the relevant charts and tables in the Jupyter cell. 
[变量d] ARIMA 模型中的最优差分顺序是什么？解释您是如何通过首先使用 ACF 并与 ADF/KPSS/PP 测试进行交叉检查来导出变量 d 的。您可以在 Jupyter 单元格中绘制相关图表和表格。

b.	[Variable p] What is the optimal order of AR terms by ACF.
[变量 p] ACF 的 AR 项的最优顺序是什么。

c.	[Variable q] What is the optimal order of MA terms by utilizing ACF. For question a – c, you may cross validate your analysis using AIC / BIC. 
[变量 q] 利用 ACF 的 MA 项的最佳顺序是什么。对于问题 a - c，您可以使用 AIC / BIC 交叉验证您的分析。

d.	[Modelling Fitting] Now that you have derived the terms of p, d, and q. 
[建模拟合] 现在您已经导出了 p、d 和 q 的项。
i.	[Training] Split your dataset into two parts. 70% would be utilized for the training / fitting of the model. 
Provide the statistical summary for the ARIMA model and interpret the summary.
[训练] 将您的数据集分成两部分。 70% 将用于模型的训练/拟合。提供 ARIMA 模型的统计摘要并解释该摘要。

ii.	[Residual Analysis] Conduct a residual analysis and interpret the model’s residual. Is the model valid for forecasting thus far?
[残差分析] 进行残差分析并解释模型的残差。到目前为止，该模型对预测有效吗？

iii.	[Forecasting] Utilizing your model, forecast the daily close next five days, include a 75% confidence interval. 
Derive the USD/CAD closing price range for the next 5days (i.e., Min & Max)
[预测] 利用您的模型，预测未来五天的每日收盘价，包括 75% 的置信区间。推导出未来 5 天的美元/加元收盘价范围（即最小和最大）

iv.	[Bonus Question] Conduct a static out-of-sample forecast. Forecast the USD/CAD closing price range for the next 5 days 
while KEEPING the model’s coefficient constant and REFRESHING the ACTUAL datapoints every 5 days. 
In other words, at time t, using the actual data at t-1 you will forecast t+1, t+2…t+5; at time t+6, instead of using the forecasted data at t+5, 
you should use the real daily close data to forecast the next 5 values. Using a loop, do this for the entirety of your “Test” section of your dataset.
[奖金问题] 进行静态的样本外预测。预测未来 5 天的美元/加元收盘价范围，同时保持模型的系数不变并每 5 天刷新一次实际数据点。换句话说，在时间 t，使用 t-1 的实际数据，您将预测 t+1、t+2…t+5；在 t+6 时刻，
您应该使用实际每日收盘数据来预测接下来的 5 个值，而不是使用 t+5 时刻的预测数据。使用循环，对数据集的整个“测试”部分执行此操作。
 
Section #2: Multivariate Time Series Analysis     多元时间序列分析
2.	In this section, you will build a Multivariate Time Series Model. Please use Jupyter Notebook for this exercise. 
You may answer all questions in MARKDOWN inside the Jupyter Notebook.
在本节中，您将构建一个多元时间序列模型。请使用 Jupyter Notebook 进行此练习。您可以在 Jupyter Notebook 内的 MARKDOWN 中回答所有问题。

a.	Propose up to five factors that may affect the trading range of USD/CAD and explain why you have made these assumptions. 
Also explain the type of data frequency that you will be using (e.g., Intraday 5-mins / Daily / Weekly).
提出最多五个可能影响美元/加元交易区间的因素，并解释您做出这些假设的原因。还要解释您将使用的数据频率类型（例如，盘中 5 分钟/每日/每周）。

b.	Causality & Statistical Signifiance 因果关系和统计意义

i.	Utilize the Granger’s Causality Test to test foe causation in the variables that you have selected. Construct a matrix to show the p-values of the variables against one another
(i.e., you also need to show that there are possible relationships between one another; not just USD/CAD).   
利用格兰杰因果检验来测试您选择的变量中的因果关系。构建一个矩阵来显示变量之间的 p 值（即，您还需要表明彼此之间可能存在关系；而不仅仅是美元/加元）。

ii.	[Test for Statistical Significance Between Time Series] Use the Johassen’s test of cointegration to check statistical significance in your variables. 
[时间序列之间的统计显着性检验] 使用 Johassen 协整检验来检查变量的统计显着性。

c.	[Check for Stationarity] Implement an ADF test to check for stationarity. 
If it is not stationary, conduct a first-order differencing and re-check for stationarity. 
Show the ADF p-values for all the selected variables. If first-order differencing is not stationary, conduct a second-order differencing and re-run the ADF test.
[检查平稳性] 实施 ADF 测试以检查平稳性。如果它不是平稳的，则进行一阶差分并重新检查平稳性。显示所有选定变量的 ADF p 值。如果一阶差分不是平稳的，则进行二阶差分并重新运行 ADF 测试。

d.	[Model Selection] Using fit comparison estimates of AIC, BIC, FPE, and HQIC. Using the Fit comparison estimates to derive the most optimal number of lags for the model.
[模型选择] 使用 AIC、BIC、FPE 和 HQIC 的拟合比较估计。使用拟合比较估计得出模型的最佳滞后数。

e.	[Forecast] Utilizing your model, forecast the daily close for the next five days, include a 75% confidence interval. Derive the USD/CAD closing price range for the next 5 days (i.e., Min & Max).
[预测] 利用您的模型，预测未来五天的每日收盘价，包括 75% 的置信区间。推导出未来 5 天的美元/加元收盘价范围（即最小值和最大值）。

