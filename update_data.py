import data

# Available offline datasets
stocks = data.get_from_files()

# Get the latest day. Will replace this day
start = data.pandas.to_datetime(stocks.index[-1])

print('Checking for data since {} ...'.format(start))

# If today hasn't concluded, there is no need to update
if start == data.pandas.to_datetime('today'):
    print('Check if the stock market has closed')
    # To be continued ...
    quit()
elif start > data.pandas.to_datetime('today'):
    print('Data is already at latest.')
    quit()

# Retrieve the newer set using the same symbol set
symbols = stocks.columns.levels[1].tolist()
stocks_new = data.get_from_net(symbols, start = start)

a = stocks.columns.levels[0].tolist()
b = stocks_new.columns.levels[0].tolist()
if not a == b:
    print('Column names not in the same order!')
    stocks_new = stocks_new[a]
    quit()

if stocks_new.shape[0] is 1:
    if all(stocks_new.iloc[0, :].values == stocks.iloc[-1, :].values):
        print('No change in data')
        quit()

# Concatenate the datasets and discard the last day of the offline data
big = data.pandas.concat([stocks.iloc[:-1, :], stocks_new])

# Save
#data.save_to_folder(big)
