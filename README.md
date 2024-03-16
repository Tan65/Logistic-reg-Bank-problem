# Logistic-reg-Bank-problem
Output variable -> y y -> Whether the client has subscribed a term deposit or not  Binomial ("yes" or "no")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from CSV file
data = pd.read_csv("bank-full.csv", delimiter=";")

# Perform label encoding for categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Visualize univariate analysis
def visualize_univariate(data):
    plt.figure(figsize=(20, 12))
    for i, col in enumerate(data.columns):
        plt.subplot(5, 5, i+1)
        sns.histplot(data[col], kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.show()

# Visualize bivariate analysis
def visualize_bivariate(data):
    plt.figure(figsize=(15, 10))
    sns.pairplot(data.sample(frac=0.01), diag_kind='kde', markers='o')
    plt.suptitle("Pair Plot of Features", y=1.02)
    plt.show()

# Visualize correlation matrix
def visualize_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# Split the data into features (X) and the target variable (y)
X = data.drop('y', axis=1)
y = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize univariate analysis
visualize_univariate(X)

# Visualize bivariate analysis
visualize_bivariate(X)

# Visualize correlation matrix
visualize_correlation_matrix(X)
