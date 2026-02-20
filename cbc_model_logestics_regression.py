import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("cbc_dataset.csv")

# print(df.head())       # first 5 rows
# print(df.tail())       # last 5 rows
# print(df.shape)        # rows, columns
# print(df.columns)      # column names
# print(df.info())       # data types
# print(df.describe())   # statistics


# print(df.isnull().sum())

# Histogram *************************************************************************************************************************

# print(df['Diagnosis'].value_counts())
# sns.countplot(x='Diagnosis', data=df)
# plt.show()

# df.hist(figsize=(15,10))
# plt.tight_layout()
# plt.show()

# Boxplot *************************************************************************************************************************

# plt.figure(figsize=(15,8))
# sns.boxplot(data=df.drop('Diagnosis', axis=1))
# plt.xticks(rotation=90)
# plt.show()


# Heat MAPPING *************************************************************************************************************************


le = LabelEncoder()
df['Diagnosis_encoded'] = le.fit_transform(df['Diagnosis'])
# # mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# # print(mapping)


# plt.figure(figsize=(12,8))
# sns.heatmap(df.select_dtypes(include=['number']).corr(),
#             annot=False,
#             cmap='coolwarm')
# plt.show()

# Handling Duplicates *************************************************************************************************************************

print("Duplicates before:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates after:", df.duplicated().sum())

df.columns = df.columns.str.lower()
print(df.columns)

numeric_cols = df.select_dtypes(include=['number']).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.clip(df[col], lower, upper)


for col in numeric_cols:
    print(col, (df[col] < 0).sum())

    df = df[df[col] >= 0]

print(df.describe())

#imbalance***************************************

# we will handle imblance when doing logistic regression by class weights 


# Feature Scaling *************************************************************************************************************************


# Separate features and target
X = df.drop(['diagnosis', 'diagnosis_encoded'], axis=1)  # all numeric features
y = df['diagnosis_encoded']  # numeric target

# Initialize scaler
scaler = StandardScaler()

# Fit and transform features
X_scaled = scaler.fit_transform(X)

# Optional: convert back to DataFrame for readability
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled.head())

print(X_scaled.mean().round(2))
print(X_scaled.std().round(2))


# train-test split *************************************************************************************************************************


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train distribution:", Counter(y_train))
print("y_test distribution:", Counter(y_test))

# Logistic Regression with class weights *************************************************************************************************************************

# Initialize logistic regression
model = LogisticRegression(  
    solver='lbfgs',             
    max_iter=1000,
    class_weight='balanced'     
)

# Fit model
model.fit(X_train, y_train)



# predict *****************************************************************************

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#model.predict_proba
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))