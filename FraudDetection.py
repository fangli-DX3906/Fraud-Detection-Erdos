import pandas as pd
import numpy as np
import pickle

from collections import Counter, defaultdict

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Data summary
try:
    df = pd.read_csv('/Users/kangdi/Desktop/NN/creditcard.csv')
except:
    df = pd.read_csv('/Users/fangli/Python/kaggle/credit card/creditcard.csv')

df.head()

df.describe()
df.isnull().sum().sum()

# imbalanced dataset
summary = df['Class'].value_counts()
num_observations = len(df['Class'])
fraud_percentage = (summary[1] / num_observations) * 100
nonfraud_percentage = (summary[0] / num_observations) * 100
# print
print(f"Fraud takes up {fraud_percentage} % of the dataset.")
print(f"Non-fraud takes up {nonfraud_percentage} % of the dataset.")

### visualization ###
# Class
sns.countplot(df, x='Class')
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()

# Time & Amount
fig, ax = plt.subplots(1, 2, figsize=(18, 4))

sns.histplot(df['Amount'], ax=ax[0], kde=True, stat="density")
ax[0].set_title('Distribution of Transaction Amount')
ax[0].set_ylim(0, 0.004)
ax[0].set_xlim([min(df['Amount']), max(df['Amount'])])

sns.histplot(df['Time'], ax=ax[1], kde=True, stat="density")
ax[1].set_title('Distribution of Transaction Time')
ax[1].set_xlim([min(df['Time']), max(df['Time'])])
plt.show()

# V1-28
num_cols = 4  # Number of columns for subplots
num_rows = (len(df.columns) - 1) // num_cols + 1  # Number of rows needed

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))

axes = axes.flatten()

for i, col in enumerate(df.columns[1:29]):  # Exclude the first column which is assumed to be 'Class'
    sns.histplot(df[col], ax=axes[i], kde=True, stat="density")
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

### Preprocessing ###
Time_transformer = RobustScaler().fit(df[['Time']])
Amount_transformer = RobustScaler().fit(df[['Amount']])

Scaled_time = Time_transformer.transform(df[['Time']])
Scaled_amount = Amount_transformer.transform(df[['Amount']])

df.insert(0, 'Scaled_time', Scaled_time)
df.insert(1, 'Scaled_amount', Scaled_amount)

### original ###
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

### Downsampling ###
fraud_count = summary[1]

df = df.sample(frac=1)

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:fraud_count]

concat_df = pd.concat([fraud_df, non_fraud_df])

undersampled_df = concat_df.sample(frac=1, random_state=42)

print(undersampled_df['Class'].value_counts())

X_under = undersampled_df.drop('Class', axis=1)
y_under = undersampled_df['Class']
X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under, y_under, test_size=0.2,
                                                                            random_state=42)
X_under_train = X_under_train.to_numpy()
X_under_test = X_under_test.to_numpy()
y_under_train = y_under_train.to_numpy()
y_under_test = y_under_test.to_numpy()

### Oversampling ###
over = SMOTE(sampling_strategy='minority')
under = RandomUnderSampler(sampling_strategy=1)

steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_over, y_over = pipeline.fit_resample(X, y)
counter = Counter(y)
print(counter)
counter_over = Counter(y_over)
print(counter_over)

X_over_train, X_over_test, y_over_train, y_over_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)
X_over_train = X_over_train.to_numpy()
X_over_test = X_over_test.to_numpy()
y_over_train = y_over_train.to_numpy()
y_over_test = y_over_test.to_numpy()

###
data_dict = defaultdict(dict)
data_dict = {'under_sample': [X_under_train, X_under_test, y_under_train, y_under_test],
             'over_sample': [X_over_train, X_over_test, y_over_train, y_over_test],
             'original_sample': [X_train, X_test, y_train, y_test]}

with open('data.pkl', 'wb') as file:
    pickle.dump(data_dict, file)
