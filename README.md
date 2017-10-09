Various Analyses on Time Series
===

Example analyses using various regression techniques, filtering, preditions and eventually some deep learning type fitting and prediction.


## Requirements

```shell
pip3 install pandas_datareader requests_cache joblib
```

### Stock Market Time Series

The stock market is arguably one of the most interesting time-series data, right? So, that's what we will start with.

The data can be retrieved live using the Yahoo or Google API through [pandas-datareader]. While the Google API is yet to be stable, a cache copy can be stored locally for repetitive experimentation.

### Chart

A convenient function to generate chart is included.

![chart](images/AAPL.png)

[pandas-datareader]:https://pandas-datareader.readthedocs.io/en/latest/

### Example Scripts

A few trivial example scripts as I was learning how to setup NN through tensorflow.

- `eg1.py` - Simple linear prediction
- `eg2.py` - Simple 2-layer FC NN with redundant data for up-down inference
 
