import numpy as np

Xr = np.array([['France', 44.0, 72000.0],
              ['Spain', 27.0, 48000.0],
              ['Germany', 30.0, 54000.0],
              ['Spain', 38.0, 61000.0],
              ['Germany', 40.0, np.nan],
              ['France', 35.0, 58000.0],
              ['Spain', np.nan, 52000.0],
              ['France', 48.0, 79000.0],
              ['Germany', 50.0, 83000.0],
              ['France', 37.0, 67000.0]], dtype=object)
yr = np.array(['no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'yes'])

# Replace NaN with the column mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
age_income = imputer.fit(Xr[:, 1:3]).transform(Xr[:, 1:3])
age_income = age_income.astype(np.float32)

# Encode categorical data as numerical label
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
c1 = labelencoder.fit_transform(Xr[:, 0])

# Encode categorical data as one hot vector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
onehotencoder = ColumnTransformer([("dummy", OneHotEncoder(categories='auto'), [0])])
c2 = onehotencoder.fit_transform(Xr)

X = np.hstack((c2[:, 1:], age_income))

y = labelencoder.fit_transform(yr)

# Splitting a dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
