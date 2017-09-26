Various Analyses on Time Series
===

Example analyses using various regression techniques, filtering, preditions and eventually some deep learning type fitting


## Requirements

```shell
pip3 install pandas_datareader requests_cache joblib
```

### Stock Market Time Series

What other time series is more interesting than the stock market, right? So, that's what we will use.

The data can be retrieved live using the Yahoo or Google API through [pandas-datareader]. While the Google API is yet to be stable, a cache copy can be stored locally for repetitive experimentation.

### Chart

A convenient function to generate chart is included.

![chart](images/AAPL.png)

[pandas-datareader]:https://pandas-datareader.readthedocs.io/en/latest/
