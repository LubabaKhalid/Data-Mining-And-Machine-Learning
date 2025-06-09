import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

laptops = pd.read_csv("/kaggle/input/laptop-prices-dataset/laptopPrice.csv")

laptops.head()

laptops.info()

laptops.describe()

numerical_columns = ['Price', 'Number of Ratings', 'Number of Reviews']

laptops = laptops[(np.abs(stats.zscore(laptops[numerical_columns])) < 3).all(axis=1)]

print("After removing outliers, our dataset has {} rows.".format(laptops.shape[0]))

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress only that specific warning
warnings.filterwarnings("ignore", message=".*figure layout has changed to tight*")

# Clean data (optional)
laptops.replace([np.inf, -np.inf], np.nan, inplace=True)
laptops.dropna(inplace=True)

# Seaborn styling
sns.set_context('notebook')
sns.set_style('whitegrid')
sns.set_palette("Dark2")
sns.despine()

# Plot
sns.pairplot(laptops)
plt.show()


sns.lmplot(x='Number of Ratings',y='Number of Reviews',data=laptops);

laptops[numerical_columns].hist(figsize=(12, 6));

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.brand)
axes[0].set_title("Count of laptops by Brands")

sns.boxplot(ax=axes[1], x=laptops.brand, y=laptops.Price)
axes[1].set_title("Boxplot of Prices by Brands");

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.processor_brand)
axes[0].set_title("Count of laptops by Processor Brands")

sns.boxplot(ax=axes[1], x=laptops.processor_brand, y=laptops.Price)
axes[1].set_title("Boxplot of Prices by Processor Brands");

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.processor_name)
axes[0].set_title("Count of laptops by Processor Name")
axes[0].tick_params(axis='x', rotation=90)

sns.boxplot(ax=axes[1], x=laptops.processor_name, y=laptops.Price)
axes[1].set_title("Boxplot of Prices by Processor Name")
axes[1].tick_params(axis='x', rotation=90);

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.processor_gnrtn)
axes[0].set_title("Count of laptops by Generation")

sns.boxplot(ax=axes[1], x=laptops.processor_gnrtn, y=laptops.Price)
axes[1].set_title("Boxplot of Prices by Generation");

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.ram_gb)
axes[0].set_title("Count of laptops by Ram Gb")

sns.boxplot(ax=axes[1], x=laptops.ram_gb, y=laptops.Price)
axes[1].set_title("Boxplot of Prices by Ram Gb");

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.ssd)
axes[0].set_title("Count of laptops by SSD")
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(ax=axes[1], x=laptops.ssd, y=laptops.Price)
axes[1].set_title("Boxplot of Prices by SSD")
axes[1].tick_params(axis='x', rotation=45);

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.os, hue=laptops.os_bit)
axes[0].set_title("Count of laptops by OS")

sns.boxplot(ax=axes[1], x=laptops.os, y=laptops.Price, hue=laptops.os_bit)
axes[1].set_title("Boxplot of Prices by OS");

fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(14,6))

sns.countplot(ax=axes[0], x=laptops.Touchscreen)
axes[0].set_title("Count of laptops by Touchscreen")

sns.boxplot(ax=axes[1], x=laptops.Touchscreen, y=laptops.Price)
axes[1].set_title("Boxplot of Prices by Touchscreen");

laptops[laptops['ssd'] == '128 GB']

laptops['processor_gnrtn'].value_counts(normalize=True)

laptops[laptops['processor_gnrtn'] == "Not Available"]['processor_brand'].value_counts()

laptops['processor_brand'].value_counts()

categorical_variables = laptops.columns[laptops.dtypes == 'object']

laptops = pd.get_dummies(laptops, columns=categorical_variables, drop_first=True)

laptops.head()

print("After feature encoding, our dataset has {} columns.".format(laptops.shape[1]))

X = laptops.loc[:, laptops.columns != "Price"]

y = laptops['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X=X_train, y=y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions,s=15)
plt.xlabel('Y Test(True Values)')
plt.ylabel('Predicted Values')
plt.plot(y_test, y_test, color='red', lw=1)

plt.show()

print("R^2 on training  data ",lm.score(X_train, y_train))
print("R^2 on testing data ",lm.score(X_test,y_test))

sns.histplot(x=(y_test-predictions), kde=True, bins=50);