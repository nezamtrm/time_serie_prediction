import pandas as pd
dat = pd.read_csv('euro_to_usd_last.csv')
dat = dat.iloc[700000:, :]
dat.to_csv('./data1.csv')