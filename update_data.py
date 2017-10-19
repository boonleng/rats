import data

# Available offline datasets
stocks = data.get_old_data()
stocks = stocks[:, :, data.SYMBOLS]

# Get the latest day and offset by 1
start = stocks.iloc[:, :, 0].index[-1] + data.pandas.tseries.offsets.DateOffset(days = 1)

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
stocks_new = stocks_new[:, :, data.SYMBOLS]

# Concatenate the datasets and save them
big = data.pandas.concat([stocks, stocks_new], axis = 1)
data.save_to_folder(big)
