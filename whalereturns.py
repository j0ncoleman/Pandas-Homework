import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sn
# Reading whale returns
whale_file = '../Pandas-Homework/Resources/whale_returns.csv'
whale_df = pd.read_csv(whale_file, index_col="Date",
                       infer_datetime_format=True, parse_dates=True)
# Count nulls
whale_df.isnull().sum()
# Drop nulls
whale_df.dropna(inplace=True)
# Checking again for nulls
whale_df.isnull().sum()

# Reading algortithmic returns
algo_file = '../Pandas-Homework/Resources/algo_returns.csv'
algo_df = pd.read_csv(algo_file, index_col="Date",
                      infer_datetime_format=True, parse_dates=True)
# Count nulls
algo_df.isnull().sum()
# Drop nulls
algo_df.dropna(inplace=True)
# Checking again for nulls
algo_df.isnull().sum()

# Reading S&P TSX 60 Closing Prices
spfile = '../Pandas-Homework/Resources/sp_tsx_history.csv'
sp_df = pd.read_csv(spfile, index_col='Date',
                    infer_datetime_format=True, parse_dates=True)
# Check Data Types
sp_df.dtypes

# Fix Data Types
sp_df['Close'] = sp_df['Close'].str.replace('$', '')
sp_df['Close'] = sp_df['Close'].str.replace(',', '')
sp_df['Close'] = sp_df['Close'].astype('float')

# Count nulls
sp_df.isnull().sum()
# Drop nulls
sp_df.dropna(inplace=True)
# Checking again for nulls
sp_df.isnull().sum()

# Count nulls
sp_df.isnull().sum()
# Drop nulls
sp_df.dropna(inplace=True)
# Checking again for nulls
sp_df.isnull().sum()

# Calculate Daily Returns
daily_returns = sp_df.pct_change()

sp_df['Close'] = sp_df['Close'].pct_change()
sp_df

# Drop nulls
sp_df.dropna(inplace=True)

# Rename `Close` Column to be specific to this portfolio.
sp_df.rename(columns={'Close': 'S&P TSX'}, inplace=True)
sp_df.head()


# Join Whale Returns, Algorithmic Returns, and the S&P TSX 60 Returns into a single DataFrame with columns for each portfolio's returns.

combined_df = pd.concat([whale_df, algo_df, sp_df],
                        axis="columns", join="inner")
combined_df

# Plot daily returns of all portfolios
combined_df.plot(title="Daily Returns of All Portfolios")
# Calculate cumulative returns of all portfolios
all_cumulative_returns = (1 + combined_df).cumprod()

# Plot cumulative returns
all_cumulative_returns.plot(title="All Cumulative Returns")

# Risk Analysis
# Box plot to visually show risk

sp_df.plot.box(title="TSX S&P 60 Risk")
algo_df.plot.box(title="Algo Portfolios Risk")
whale_df.plot.box(title="Whale Portfolios Risk")

# Calculate the daily returns of all portfolios
# Calculate the daily standard deviations of all portfolios
sp_daily_std = sp_df.std()
algo_daily_std = algo_df.std()
whale_daily_std = whale_df.std()
# Determine which portfolios are riskier than the S&P TSX 60

whale_volatility = whale_df.std() * np.sqrt(252)
whale_volatility = whale_volatility.sort_values()

sp_df.head()
sp_volatility = sp_daily_std * np.sqrt(252)
sp_volatility = sp_volatility.sort_values()

algo_volatility = algo_df.std() * np.sqrt(252)
algo_volatility = algo_volatility.sort_values()

sp_volatility.mean()
whale_volatility.mean()  # Whale portfolio is riskier than S&P TSX 60
algo_volatility.mean()  # Algo portfolio is riskier than S&P TSX 60

sp_annualized_std = sp_daily_std * np.sqrt(252)
sp_annualized_std.head()

algo_annualized_std = algo_daily_std * np.sqrt(252)
algo_annualized_std.head()

whale_annualized_std = whale_daily_std * np.sqrt(252)
whale_annualized_std.head()

# Calculate the rolling standard deviation for all portfolios using a 21-day window
# Plot the rolling standard deviation
rolling_std_all_portfolios = combined_df.rolling(window=21).std()
rolling_std_all_portfolios.plot(
    title="21 Day Rolling Standard Deviation of All Portfolios")
# Calculate the correlation
correlation = combined_df.corr()
combined_df.head()
# Calculate covariance of a single portfolio
covariance = combined_df['SOROS FUND MANAGEMENT LLC'].cov(
    combined_df['S&P TSX'])
covariance
# Calculate variance of S&P TSX
variance = combined_df['S&P TSX'].var()
variance
# Computing beta
soros_beta = covariance / variance
# Plot beta trend
rolling_covariance = combined_df['SOROS FUND MANAGEMENT LLC'].rolling(
    window=60).cov(combined_df['S&P TSX'])
rolling_variance = combined_df['S&P TSX'].rolling(window=60).var()

rolling_beta = rolling_covariance / rolling_variance
rolling_beta.plot(figsize=(15, 10),
                  title='Rolling 60-Day Beta of SOROS FUND MANAGEMENT LLC')

# Use `ewm` to calculate the rolling window


# Annualized Sharpe Ratios
sharpe_ratios = []
sp_sharpe_ratio = (
    sp_df.mean() * 252 / (sp_df.std() * np.sqrt(252))
)
sp_sharpe_df = pd.DataFrame(sp_sharpe_ratio)
algo_sharpe_ratio = (
    algo_df.mean() * 252 / (algo_df.std() * np.sqrt(252))
)
algo_sharpe_df = pd.DataFrame(algo_sharpe_ratio)

whale_sharpe_ratio = (
    whale_df.mean() * 252 / (whale_df.std() * np.sqrt(252))
)
whale_sharpe_df = pd.DataFrame(whale_sharpe_ratio)

sharpe_ratios = pd.concat(
    [sp_sharpe_df, algo_sharpe_df, whale_sharpe_df], axis='rows', join='outer')

# Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot.bar()
# Determine whether the algorithmic strategies outperform both the market (S&P TSX 60) and the whales portfolios.
print("The Algorithms outperform both the market and the whales portfolios")

# Create Custom Portfolio
# Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
aapl_file = '../Pandas-Homework/Resources/aapl_historical.csv'
amzn_file = '../Pandas-Homework/Resources/amzn_historical.csv'
dis_file = '../Pandas-Homework/Resources/dis_historical.csv'
meta_file = '../Pandas-Homework/Resources/meta_historical.csv'

# Reading data from 1st stock
aapl_df = pd.read_csv(aapl_file, index_col="Date",
                      infer_datetime_format=True, parse_dates=True)
# Reading data from 2nd stock
amzn_df = pd.read_csv(amzn_file, index_col="Date",
                      infer_datetime_format=True, parse_dates=True)
# Reading data from 3rd stock
dis_df = pd.read_csv(dis_file, index_col="Date",
                     infer_datetime_format=True, parse_dates=True)
# Reading data from 4th stock
meta_df = pd.read_csv(meta_file, index_col="Date",
                      infer_datetime_format=True, parse_dates=True)
# Combine all stocks in a single DataFrame
combined_stocks_df = pd.concat(
    [aapl_df, amzn_df, dis_df, meta_df], axis="columns", join="inner")

# Reset Date index
combined_stocks_df.reset_index()
# Reorganize portfolio data by having a column per symbol
combined_stocks_df.columns = ['AAPL', 'AMZN', 'DIS', 'META']
combined_stocks_df
# Calculate daily returns
combined_daily_returns = combined_stocks_df.pct_change()
# Drop NAs
combined_daily_returns.dropna(inplace=True)
# Check for null values
combined_daily_returns.isnull().sum()
# Display sample data
combined_daily_returns.head()

# Set weights
weights = [1/4, 1/4, 1/4, 1/4]

# Calculate portfolio return
portfolio_returns = combined_daily_returns.dot(weights)
# Display sample data
portfolio_returns.head()
new_returns_df = pd.concat(
    [combined_df, combined_daily_returns], axis="columns", join="outer")
new_returns_df.dropna(inplace=True)
new_returns_df.head()

# Calculate the annualized `std`
my_portfolio_daily_std = combined_daily_returns.std()
my_portfolio_annualized_std = my_portfolio_daily_std * np.sqrt(252)
my_portfolio_annualized_std
# Calculate and plot rolling std with 21-day window
rolling_std_my_portfolio = combined_daily_returns.rolling(window=21).std()
rolling_std_my_portfolio.plot(
    title="21 Day Rolling Standard Deviation of My Portfolio")
# Calculate and plot the correlation
my_correlation = combined_daily_returns.corr()
ax = plt.axes()
sn.heatmap(my_correlation, annot=True, ax=ax)
ax.set_title("Correlation")
plt.show()

# Calculate and Plot the 60-day Rolling Beta for Your Portfolio compared to the S&P 60 TSX
rolling_covariance_aapl = new_returns_df["AAPL"].rolling(
    window=60).cov(new_returns_df['S&P TSX'])
rolling_covariance_amzn = new_returns_df["AMZN"].rolling(
    window=60).cov(new_returns_df['S&P TSX'])
rolling_covariance_dis = new_returns_df["DIS"].rolling(
    window=60).cov(new_returns_df['S&P TSX'])
rolling_covariance_meta = new_returns_df["META"].rolling(
    window=60).cov(new_returns_df['S&P TSX'])
rolling_variance_sp = new_returns_df['S&P TSX'].rolling(window=60).var()

rolling_beta_aapl = rolling_covariance_aapl / rolling_variance_sp
rolling_beta_amzn = rolling_covariance_amzn / rolling_variance_sp
rolling_beta_dis = rolling_covariance_dis / rolling_variance_sp
rolling_beta_meta = rolling_covariance_meta / rolling_variance_sp
rolling_beta_aapl.plot(figsize=(15, 10),
                       title='Rolling 60-Day Beta of Apple')
rolling_beta_amzn.plot(figsize=(15, 10),
                       title='Rolling 60-Day Beta of Amazon')
rolling_beta_dis.plot(figsize=(15, 10),
                      title='Rolling 60-Day Beta of Disney')
rolling_beta_meta.plot(figsize=(15, 10),
                       title='Rolling 60-Day Beta of Meta')
# Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot
my_sharpe_ratio = combined_daily_returns.mean(
) * 252 / (combined_daily_returns.std() * np.sqrt(252))
sharpe_df = pd.DataFrame(my_sharpe_ratio)
sharpe_df.plot.bar(title="My Portfolio's Sharpe Ratios")
# Combine all sharpe ratios and plot them on a bar graph
combined_sharpe_ratios = pd.concat(
    [sharpe_ratios, my_sharpe_ratio], axis="rows", join="inner")
combined_sharpe_ratios.plot.bar(title="All Portfolio Sharpe Ratios")
# How does your portfolio do?
print("My portfolio does quite well compared to the market. Looking at the bar graph we can wee that our portfolio is only outperformed by Algorithm 1. The only stock in my portfolio which is outperformed by the market and 2/5 of the whale portfolios is Meta. In conclusion, my portfolio outperforms all other portfolios listed.")
