import numpy as np

X = [['France', 44.0, 72000.0],
     ['Spain', 27.0, 48000.0],
     ['Germany', 30.0, 54000.0],
     ['Spain', 38.0, 61000.0],
     ['Germany', 40.0, np.nan],
     ['France', 35.0, 58000.0],
     ['Spain', np.nan, 52000.0],
     ['France', 48.0, 79000.0],
     ['Germany', 50.0, 83000.0],
     ['France', 37.0, 67000.0]]
X = X.astype(object)

# Replace NaN with the column mean

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
