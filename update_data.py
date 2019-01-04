import data

# Available offline datasets
old = data.file()

# Get the latest day. Will replace this day
start = data.pandas.to_datetime(old.index[-1])

print('Checking for data since {} ...'.format(start.strftime('%Y-%m-%d')))

# If today hasn't concluded, there is no need to update
if start >= data.yesterday():
	print('No new data is expected')
	quit()

# Retrieve the newer set using the same symbol set
symbols = old.columns.levels[1].tolist()
new = data.net(symbols, start=start)

# Verify if new row is different
u = old.iloc[-1]
v = new.iloc[0]
if new.shape[0] is 1 and all(u.sub(v).values < 1.0e-3):
    print('No change in data, skip saving data')
    quit()

# Concatenate the datasets and discard the last day of the offline data
print('Saving new data ...')
big = data.pandas.concat([old.iloc[:-1], new])
data.save(big)
