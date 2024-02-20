import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Gather Data
file_path = 'housing_price_dataset.csv'
df = pd.read_csv(file_path)

# Explore the Dataset
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Convert Categorical Variables
df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

# Split Data into Features (X) and Target Variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split Data into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose and Train a Model

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model Performance

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
