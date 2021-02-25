from scweet import scrap
 
data = scrap(
    # Keywords for query
    words = 'Amazon', 
    # Tweets language, here we just care about english tweets
    lang='en', 
    # Start date for search query
    start_date="2020-01-01",
    # End date for search query
    max_date="2021-01-01", 
    # Days between two query, 1 means search everyday without jump
    interval=1, 
    # True means show the operation on Chrome, False means hide operation background
    headless=True, 
    # Proxy settings for selenium webdriver, not needed if you are nott in mainland China
    proxy="socks5://127.0.0.1:7891", 
    # Resume the last scraping work, all parameters should be the same to last execution
    resume=False)
