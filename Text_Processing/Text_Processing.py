import pandas as pd
from textblob import TextBlob

df = pd.read_csv(r'D:\gitrepo\NLP_Individual_Project\Text_Processing\AAPL_2020-02-26_2021-02-27.csv')

# trim the date
Date =[]
for time in df['Timestamp']:
    date = str(time)[0:10]
    Date.append(date)
    df['Date'] = pd.Series(Date)

# define a function recognize the number of hashtags 
def hashtag_count(text):
    counts = 0
    # splitting the text into words 
    for word in text.split(): 
        # checking the first charcter of every word 
        if word[0] == '#': 
            # counts will add one
            counts = counts + 1
    return counts   

# conduct sentiment analysis of each text
TextBlob_Polarity = []
for line in df['Text']:
    # count those tweets's sentiment with more than 5 tags as 0 bacause abnormal users may exit
    if hashtag_count(line) <= 5:
        polarity = TextBlob(str(line)).sentiment.polarity 
        TextBlob_Polarity.append(polarity)
    else:
        polarity = 0
        TextBlob_Polarity.append(polarity)
# add another column of sentiment polarity to the dataframe    
df['TextBlob_Polarity'] = pd.Series(TextBlob_Polarity)

df.to_csv(r'D:\gitrepo\NLP_Individual_Project\Text_Processing\output.csv')






