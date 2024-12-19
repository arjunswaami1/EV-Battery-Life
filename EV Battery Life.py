import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Generate Sample Data
np.random.seed(42)
time_steps = 1000
data = {
    "Charge_Cycles": np.arange(time_steps),
    "Temperature": 25 + np.sin(np.arange(time_steps) * 0.01) * 10 + np.random.normal(0, 1, time_steps),
    "Capacity_Degradation": 100 - (np.arange(time_steps) * 0.05) + np.random.normal(0, 0.5, time_steps)
}
df = pd.DataFrame(data)

# Visualize the Sample Data
plt.figure(figsize=(10, 6))
plt.plot(df["Charge_Cycles"], df["Capacity_Degradation"], label="Capacity Degradation")
plt.xlabel("Charge Cycles")
plt.ylabel("Capacity (%)")
plt.title("Battery Capacity Degradation over Charge Cycles")
plt.legend()
plt.show()

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["Temperature", "Capacity_Degradation"]])

# Create time-series sequences for LSTM
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :-1])
        y.append(data[i + time_steps, -1])
    return np.array(X), np.array(y)

time_steps = 30
X, y = create_sequences(scaled_data, time_steps)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Plot Predictions vs True Values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="True Values", color='blue')
plt.plot(y_pred, label="Predicted Values", color='red', alpha=0.7)
plt.xlabel("Test Samples")
plt.ylabel("Scaled Capacity")
plt.title("True vs Predicted Battery Capacity")
plt.legend()
plt.show()
