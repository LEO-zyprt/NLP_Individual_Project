import pandas as pd
import numpy as py
import matplotlib as mpl
import matplotlib.pyplot as plt

df_raw = pd.read_csv(r'D:\gitrepo\NLP_Individual_Project\Sentiment_Analysis\output.csv')
df_AAPL = pd.read_csv(r'D:\gitrepo\NLP_Individual_Project\FinancialData\AAPL.csv')

# Sum up daily polarity value to further analyze
df_daily_polarity = df_raw.groupby(['Date'])['TextBlob_Polarity'].sum().reset_index()
#print(df_daily_polarity.head())

# after check the head and the tail of the dataframe, we find index 0,1 are not what we need
df_daily_polarity = df_daily_polarity.iloc[2:]

# to add the daily return column
daily_return = df_AAPL['Adj Close']/df_AAPL['Adj Close'].shift(1) - 1
df_AAPL['daily_return'] = daily_return
# print(df_AAPL.head())

# transform sentiment to sentiment change 
Sentiment_Change = df_daily_polarity['TextBlob_Polarity'] - df_daily_polarity['TextBlob_Polarity'].shift(1) 
df_daily_polarity['Sentiment_Change'] = Sentiment_Change

# Merge the sentiment data and finanical data, ignore the day without sentiment value
df_combine = pd.merge(df_AAPL, df_daily_polarity, on = 'Date')[['Date','daily_return','Sentiment_Change']].iloc[1:,:]
print(df_combine.head(20))

# plot the sentiment and return respectively
df_combine.plot(x='Date',y='Sentiment_Change', kind = 'bar', color ='blue') #sentiment_change
# plt.show()
df_combine.plot(x='Date',y='daily_return', kind = 'bar', color ='Grey') #Return
# plt.show()

# plot the sentiment and return together
fig, ax1 = plt.subplots()
#Plot bars
ax1.bar(df_combine['Date'], df_combine['Sentiment_Change'], color = 'blue')
ax1.set_xlabel('Date')
# Make the y-axis label 
ax1.set_ylabel('Sentiment Change')
#Set up ax2 to be the second y axis with x shared
ax2 = ax1.twinx()
#Plot a line
ax2.plot(df_combine['Date'], df_combine['daily_return'], color = 'black')
# Make the y-axis label 
ax2.set_ylabel('Daily Return')

plt.show()