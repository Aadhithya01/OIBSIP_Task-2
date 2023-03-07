from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('Unemployment.csv')

# Printing the column names
print(df.columns)

# Printing the first 5 values in the data
print(df.head())

# Printing the shape of the data
print(df.shape)

# Printing the basic info of the data
print(df.info())

print(df.isnull().sum())

df = df.dropna()
X = df[[' Estimated Unemployment Rate (%)', ' Estimated Employed', ' Estimated Labour Participation Rate (%)']]

y = df[' Estimated Employed']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lm = LinearRegression()
# fit the model inside it
lm.fit(X_train, y_train)

LinearRegression()

# Predict the model
predictions = lm.predict(X_test)

# plotting the prediction agains the target variable
plt.scatter(y_test, predictions)
plt.show()
# evaluating model
coeff_data = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])

# Representing the data in the form of a heat map
plt.figure(figsize=(10,9))
sns.heatmap(df.corr())
plt.show()

# Representing the data as a Histogram
df.columns= ["Region","Date","Frequency",
               "Estimated Unemployment Rate (%)","Estimated Employed",
               "Estimated Labour Participation Rate (%)","Area"]
plt.title("Indian Unemployment")
sns.histplot(x='Estimated Employed', hue="Region", data=df)
plt.show()

df.columns= ["Region","Date","Frequency",
               "Estimated Unemployment Rate (%)","Estimated Employed",
               "Estimated Labour Participation Rate (%)","Area"]
sns.histplot(x='Estimated Unemployment Rate (%)', hue="Region", data=df)
plt.ylabel('Count')
plt.title("Region_data")
plt.show()

# Plotting a pairplot for analysis
sns.pairplot(df,hue="Area")
plt.show()