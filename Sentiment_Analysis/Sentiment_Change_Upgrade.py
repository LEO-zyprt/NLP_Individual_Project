import pandas as pd
import numpy as py
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels
import matplotlib as mpl
import matplotlib.pyplot as plt

df_raw = pd.read_csv(r'D:\gitrepo\NLP_Individual_Project\Sentiment_Analysis\output.csv')
df_AAPL = pd.read_csv(r'D:\gitrepo\NLP_Individual_Project\FinancialData\AAPL.csv')

# Sum up daily polarity value to further analyze
df_daily_polarity = df_raw.groupby(['Date'])['TextBlob_Polarity'].sum().reset_index()
#print(df_daily_polarity.head())

# after check the head and the tail of the dataframe, we find index 0,1 are not what we need
df_daily_polarity = df_daily_polarity.iloc[2:]

# to add the daily 1+return column
daily_return = df_AAPL['Adj Close']/df_AAPL['Adj Close'].shift(1)  # without minus 1
df_AAPL['daily_return'] = daily_return
# print(df_AAPL.head(10))

# transform sentiment to sentiment change 
Sentiment_Change = df_daily_polarity['TextBlob_Polarity'] - df_daily_polarity['TextBlob_Polarity'].shift(1) 
df_daily_polarity['Sentiment_Change'] = Sentiment_Change

# analyze monthly data
Month = []
for time in df_daily_polarity['Date']:
    month = time[:7]
    Month.append(month)
    df_daily_polarity['Month'] = pd.Series(Month) 
df_monthly_SentiChange = df_daily_polarity.groupby(['Month'])['Sentiment_Change'].sum().reset_index()
#print(df_monthly_SentiChange.head())

Month = []
for time in df_AAPL['Date']:
    month = time[:7]
    Month.append(month)
    df_AAPL['Month'] = pd.Series(Month) 
df_monthly_CumReturn = df_AAPL.groupby(['Month'])['daily_return'].prod().reset_index()
df_monthly_CumReturn['monthly_return'] = df_monthly_CumReturn['daily_return'] - 1
#print(df_monthly_CumReturn.head())

# Merge the sentiment_change data and monthly_return data
df_combine = pd.merge(df_monthly_CumReturn, df_monthly_SentiChange, on = 'Month')[['Month','monthly_return','Sentiment_Change']]
#print(df_combine.head(20))

# Make OLS regression between monthly return with sentiment change
Result = smf.ols('monthly_return ~ Sentiment_Change', data=df_combine).fit().summary()
print(Result)

'''
# plot the sentiment_change and monthly return together
fig, ax1 = plt.subplots()
#Plot bars
ax1.bar(df_combine['Month'], df_combine['Sentiment_Change'], color = 'blue')
ax1.set_xlabel('Month')
# Make the y-axis label 
ax1.set_ylabel('Sentiment Change')
#Set up ax2 to be the second y axis with x shared
ax2 = ax1.twinx()
#Plot a line
ax2.plot(df_combine['Month'], df_combine['monthly_return'], color = 'black')
# Make the y-axis label 
ax2.set_ylabel('Monthly Return')

plt.show()
'''

# lag the sentiment change to see the relationship between monthly return and last month's sentiment change
df_combine['Sentiment_Change'] = df_combine['Sentiment_Change'].shift(1)
df_sentiment_lag = df_combine.iloc[1:]
print(df_sentiment_lag.head())

# Make OLS regression between monthly return with Lagged sentiment change 
Result_lag = smf.ols('monthly_return ~ Sentiment_Change', data=df_sentiment_lag).fit().summary()
print(Result_lag)