import data

# Available offline datasets
stocks = data.get_from_files()

# Get the latest day. Will replace this day
start = stocks.iloc[:, :, 0].index[-1]

print('Checking for data since {} ...'.format(start))

# If today hasn't concluded, there is no need to update
if start == data.pandas.to_datetime('today'):
    print('Check if the stock market has closed')
    # To be continued ...
    quit()
elif start > data.pandas.to_datetime('today'):
    print('Data is already at latest.')
    quit()

# Retrieve the newer
stocks_new = data.get_from_net(data.SYMBOLS, start = start)

# Sort the data symbols and discard the last day of offline data
stocks = stocks[:, :, data.SYMBOLS].iloc[:, :-1, :]
stocks_new = stocks_new[:, :, data.SYMBOLS]

# Concatenate the datasets and save them
big = data.pandas.concat([stocks, stocks_new], axis = 1)
data.save_to_folder(big)
