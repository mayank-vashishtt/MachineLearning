#one hot encoding 

import pandas as pd

# Data
data = {'Fruit': ['Apple', 'Banana', 'Cherry', 'Apple']}
df = pd.DataFrame(data)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Fruit'])
print(df_encoded)


# Label encoding

from sklearn.preprocessing import LabelEncoder

# Data
data = {'Fruit': ['Apple', 'Banana', 'Cherry', 'Apple']}
df = pd.DataFrame(data)

# Label encoding
encoder = LabelEncoder()
df['Fruit_Encoded'] = encoder.fit_transform(df['Fruit'])
print(df)

# Target encoding

import pandas as pd

# Data
data = {'Department': ['Sales', 'HR', 'Sales'], 'Churn': [1, 0, 1]}
df = pd.DataFrame(data)

# Target Encoding
target_mean = df.groupby('Department')['Churn'].mean()
df['Dept_Encoded'] = df['Department'].map(target_mean)
print(df)


#Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'Age': [22, 25, 47, 52, 46],
    'Salary': [20000, 24000, 50000, 60000, 58000],
    'Churn': [1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Splitting data
X = df[['Age', 'Salary']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))




