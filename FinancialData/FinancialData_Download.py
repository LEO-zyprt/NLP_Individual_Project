import pandas_datareader as pdr
import pandas as pd
from datetime import datetime

# download close price within the same period[2020-2-26 to 2021-2-27]
AAPL = pdr.get_data_yahoo(symbols='AAPL', start=datetime(2020, 2, 26), end=datetime(2021, 2, 27))
# print(AAPL['Adj Close'])
AAPL.to_csv(r'D:\gitrepo\NLP_Individual_Project\FinancialData\AAPL.csv')
