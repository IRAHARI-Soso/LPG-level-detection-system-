# 1. Import Libraries and Load Data (Already Done)
#
# Imported necessary libraries (pandas, matplotlib, seaborn, jax, optax, sqlalchemy, etc.).
# Loaded the lpg_data table from your MySQL database into a pandas DataFrame.
# Converted relevant columns to numeric format and handled the timestamp column.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import jax.numpy as jnp
from jax import random, nn  # Corrected import
import optax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the Dataset
# Connect to the MySQL database
engine = create_engine("mysql+pymysql://root:@127.0.0.1/lpg_monitoring")

# Load the `lpg_data` table into a Pandas DataFrame
query = "SELECT * FROM lpg_data"
df = pd.read_sql(query, engine)

# Ensure columns are numeric
df['lpg_remaining'] = pd.to_numeric(df['lpg_remaining'], errors='coerce')
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
df['gas_mass'] = pd.to_numeric(df['gas_mass'], errors='coerce')

# Convert the 'timestamp' column to a numerical format
df['timestamp_numeric'] = df['timestamp'].apply(lambda x: x.timestamp()).astype(int)




# 2. Preprocess the Data
# You need to handle any missing data, scale features, and split the data into training and validation sets.



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ensure only numeric columns are used for filling NaN values
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Select features and target variable
features = ['lpg_used', 'distance', 'gas_mass', 'timestamp_numeric']
target = 'lpg_remaining'  # Assuming 'lpg_level' is the target column

X = df[features].values
y = df[target].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# 3. Data visualization


# 1. Display the First Few Rows of the DataFrame

# Display the first few rows of the DataFrame
print(df.head())


# 2. Visualize the Data to Understand Its Distribution
# 2.1 Histograms to Show the Distribution of Features


# Set up the plotting environment
plt.figure(figsize=(14, 10))

# Plot histograms for the selected features
features_to_plot = ['lpg_remaining', 'distance', 'gas_mass', 'timestamp_numeric']

for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature], kde=True, bins=50, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# 3. Data Analysis with Graphs Using Seaborn
# 3.1 Pair Plot to Analyze Relationships Between Features


# Pair plot to visualize relationships between features
sns.pairplot(df[features_to_plot])
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()


# 3.2 Correlation Heatmap


# Compute the correlation matrix
corr_matrix = df[features_to_plot].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Selected Features')
plt.show()


# 1. Use Bar Plots to Show the gas_mass of lpg_data


# Bar plot of gas_mass
plt.figure(figsize=(12, 6))
sns.barplot(x=df.index, y='gas_mass', data=df, color='lightblue')
plt.title('Bar Plot of Gas Mass')
plt.xlabel('Index')
plt.ylabel('Gas Mass')
plt.show()


# 2. Use Box Plots to Show the gas_mass of lpg_data


# Box plot of gas_mass
plt.figure(figsize=(6, 8))
sns.boxplot(y='gas_mass', data=df, color='lightgreen')
plt.title('Box Plot of Gas Mass')
plt.ylabel('Gas Mass')
plt.show()


# 3. Identify Any Missing or Incorrect Values
# 3.1 Check for Missing Values in Each Column


# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)


# 3.2 Check for Any Outliers or Incorrect Values



# Detect outliers using the IQR method
Q1 = df[features_to_plot].quantile(0.25)
Q3 = df[features_to_plot].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers
outliers = ((df[features_to_plot] < (Q1 - 1.5 * IQR)) | (df[features_to_plot] > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"Number of outliers detected: {outliers.sum()}")


# 4. Perform Necessary Corrections to the Data
# 4.1 Handle Missing Values


# Fill missing values in numeric columns with the mean
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Alternatively, you could drop rows with missing values
# df = df.dropna()

# Ensure all missing values are handled
print("Missing values after handling:")
print(df.isnull().sum())



# 4.2 Correct Any Outliers or Incorrect Values

# Calculate IQR
Q1 = df[features_to_plot].quantile(0.25)
Q3 = df[features_to_plot].quantile(0.75)
IQR = Q3 - Q1

# Option 1: Cap the outliers (Winsorization)
# Clip the values to the specified bounds
df[features_to_plot] = df[features_to_plot].clip(lower=(Q1 - 1.5 * IQR), upper=(Q3 + 1.5 * IQR), axis=1)


# 5. Create Plots to Analyze Relationships Between Variables
# 5.1 Plot lpg_used vs. distance


plt.figure(figsize=(8, 6))
sns.scatterplot(x='distance', y='lpg_used', data=df, color='orange')
plt.title('LPG Used vs. Distance')
plt.xlabel('Distance')
plt.ylabel('LPG Used')
plt.show()


# 5.2 Plot lpg_remaining vs. distance


plt.figure(figsize=(8, 6))
sns.scatterplot(x='distance', y='lpg_remaining', data=df, color='blue')
plt.title('LPG Remaining vs. Distance')
plt.xlabel('Distance')
plt.ylabel('LPG Remaining')
plt.show()


# 5.3 Plot gas_mass vs. distance


plt.figure(figsize=(8, 6))
sns.scatterplot(x='distance', y='gas_mass', data=df, color='green')
plt.title('Gas Mass vs. Distance')
plt.xlabel('Distance')
plt.ylabel('Gas Mass')
plt.show()

# 5.4 Plot Combinations of gas_mass, lpg_remaining, and lpg_used


# Pair plot for gas_mass, lpg_remaining, and lpg_used
sns.pairplot(df[['gas_mass', 'lpg_remaining', 'lpg_used']])
plt.suptitle('Pair Plot of Gas Mass, LPG Remaining, and LPG Used', y=1.02)
plt.show()


# Define the Neural Network Model Using JAX


import jax.numpy as jnp
from jax import random
import optax

# Define a simple neural network model using JAX
def init_params(layer_sizes, key):
    """Initialize the parameters of the neural network."""
    keys = random.split(key, len(layer_sizes))
    params = [
        (random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), jnp.zeros(n))
        for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])
    ]
    return params

def relu(x):
    """ReLU activation function."""
    return jnp.maximum(0, x)

def neural_network(params, x):
    """Forward pass of the neural network."""
    for w, b in params[:-1]:
        x = relu(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jnp.dot(x, w) + b

# Example usage:
layer_sizes = [4, 64, 64, 1]  # 4 input features, two hidden layers with 64 units, 1 output
key = random.PRNGKey(42)
params = init_params(layer_sizes, key)

# Test the model with dummy input data
x_test = jnp.array([[0.1, 0.2, 0.3, 0.4]])  # Example input with 4 features
output = neural_network(params, x_test)
print("Neural network output:", output)


# Initialize Parameters and Optimizer


import jax.numpy as jnp
from jax import random, grad, jit
import optax

# Define the layer sizes (e.g., 4 input features, two hidden layers with 64 units each, 1 output)
layer_sizes = [4, 64, 64, 1]

# Initialize random seed key
key = random.PRNGKey(42)

# Initialize the parameters of the neural network
params = init_params(layer_sizes, key)

# Define the optimizer (e.g., Adam optimizer with a learning rate of 0.001)
learning_rate = 0.001
optimizer = optax.adam(learning_rate)

# Initialize the optimizer state
opt_state = optimizer.init(params)

# Example loss function (Mean Squared Error)
def loss_fn(params, x, y):
    predictions = neural_network(params, x)
    return jnp.mean((predictions - y) ** 2)

# Compute gradients
grads = grad(loss_fn)(params, x_test, jnp.array([1.0]))  # Example true value `1.0`

# Apply gradients (Perform a single optimization step)
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)

# Output the loss after the first step
loss_value = loss_fn(params, x_test, jnp.array([1.0]))
print("Initial loss after one optimization step:", loss_value)


# ====================================================================================================================
# Training the Model
# ====================================================================================================================


import jax.numpy as jnp
from jax import grad, jit, random
import optax

# Initialize layer sizes, random seed, parameters, and optimizer
layer_sizes = [4, 64, 64, 1]  # Example: 4 input features, 2 hidden layers with 64 units each, 1 output
key = random.PRNGKey(42)
params = init_params(layer_sizes, key)
learning_rate = 0.001
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Define the loss function (Mean Squared Error)
def loss_fn(params, x, y):
    predictions = neural_network(params, x)
    return jnp.mean((predictions - y) ** 2)

# JIT compile the gradient function for better performance
@jit
def update(params, opt_state, x, y):
    """Compute gradients and update model parameters."""
    grads = grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Training loop
num_epochs = 100  # Number of epochs to train
batch_size = 32   # Batch size for training
num_batches = X_train.shape[0] // batch_size  # Number of batches

for epoch in range(num_epochs):
    # Shuffle data
    perm = random.permutation(key, X_train.shape[0])
    X_train_shuffled = X_train[perm]
    y_train_shuffled = y_train[perm]

    # Loop over batches
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        # Update parameters with the current batch
        params, opt_state = update(params, opt_state, X_batch, y_batch)

    # Compute and print the loss after each epoch
    train_loss = loss_fn(params, X_train, y_train)
    val_loss = loss_fn(params, X_val, y_val)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Final model evaluation
final_train_loss = loss_fn(params, X_train, y_train)
final_val_loss = loss_fn(params, X_val, y_val)
print(f"Final Train Loss: {final_train_loss:.4f}, Final Validation Loss: {final_val_loss:.4f}")



# =================================================================================================================
# Evaluate the Model
# After training, evaluate the model’s performance on the validation set.
# =================================================================================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Make predictions on the validation set
y_val_pred = neural_network(params, X_val)

# Compute evaluation metrics
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)

print(f"Validation MSE: {mse:.4f}")
print(f"Validation MAE: {mae:.4f}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, color='blue', alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()


# ==========================================================================================
#
# ==========================================================================================

import numpy as np
import matplotlib.pyplot as plt

# Assuming y_val and y_val_pred are defined and are lists or numpy arrays
y_val = np.array(y_val)
y_val_pred = np.array(y_val_pred)

# Ensure they are of the same length
if len(y_val) != len(y_val_pred):
    raise ValueError("y_val and y_val_pred must be the same length")

# Calculate residuals
residuals = y_val - y_val_pred

# Print lengths for debugging
print("Length of y_val_pred:", len(y_val_pred))
print("Length of residuals:", len(residuals))

import numpy as np
import matplotlib.pyplot as plt

# Convert to numpy arrays if they aren't already
y_val = np.array(y_val)
y_val_pred = np.array(y_val_pred)

# Ensure lengths match
if len(y_val) != len(y_val_pred):
    raise ValueError("y_val and y_val_pred must be the same length")

# Calculate residuals
residuals = y_val - y_val_pred

# Check sample values
print("Sample y_val_pred values:\n", y_val_pred[:5])
print("Sample residuals values:\n", residuals[:5])



print(f"Length of y_val_pred: {len(y_val_pred)}")
print(f"Length of residuals: {len(residuals)}")

if len(y_val) != len(y_val_pred):
    raise ValueError("y_val and y_val_pred must have the same length")
residuals = y_val - y_val_pred
print("Any NaNs in y_val:", np.isnan(y_val).any())
print("Any NaNs in y_val_pred:", np.isnan(y_val_pred).any())
print("Any NaNs in residuals:", np.isnan(residuals).any())



# =================================================================
# ACCURACY
# =================================================================

import numpy as np
import jax.numpy as jnp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming y_val and y_val_pred are numpy arrays with actual and predicted values

# Convert to numpy arrays if they aren't already
y_val = np.array(y_val)
y_val_pred = np.array(y_val_pred)

# Compute evaluation metrics
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

# Print metrics
print(f"Validation MSE: {mse:.4f}")
print(f"Validation MAE: {mae:.4f}")
print(f"Validation R² Score: {r2:.4f}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, color='blue', alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()



# ======================================================
# Data sent nto database AND CALCULATE HOURS_REMAINING
# ======================================================






from sqlalchemy import create_engine, text
from datetime import datetime
import numpy as np

# Create the SQLAlchemy engine
engine = create_engine("mysql+pymysql://root:@127.0.0.1/lpg_monitoring")

# Define the table name
table_name = 'prediction'

# Ensure that actual, pred, and residual are all scalar values
data_to_insert = [
    (float(actual.flatten()[0]),  # Convert 2D array to 1D and then get the scalar value
     float(pred.flatten()[0]),    # Same here
     float(residual.flatten()[0]), # Same here
     datetime.now())
    for actual, pred, residual in zip(y_val, y_val_pred, residuals)
]

# SQL insert statement using SQLAlchemy's text construct
insert_query = text(f"""
    INSERT INTO {table_name} (actual_value, predicted_value, residual, timestamp)
    VALUES (:actual_value, :predicted_value, :residual, :timestamp)
""")

# Connect to the database and insert the data
with engine.connect() as connection:
    with connection.begin() as transaction:
        try:
            # Execute the insert statement for each set of values
            for data in data_to_insert:
                connection.execute(insert_query, {
                    'actual_value': data[0],
                    'predicted_value': data[1],
                    'residual': data[2],
                    'timestamp': data[3]
                })
            transaction.commit()
            print("Data successfully inserted into the database.")
        except Exception as e:
            transaction.rollback()
            print(f"An error occurred: {e}")
