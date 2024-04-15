# pip install pandas_datareader requests
# pip install setuptools
# pip install pandas
import pandas_datareader as pdr
import pandas as pd

# Reading Voltas shares from yahoo finance server    
# shares_df = pdr.DataReader('AAPL', 'yahoo', start='2023-01-01', end='2023-12-31')
# Look at the data read
# print(shares_df)


# Read the CSV file into a pandas DataFrame
data = pd.read_csv('trial.csv')

# Now you can work with the data using pandas functions
print(data.head())  # Display the first few rows
