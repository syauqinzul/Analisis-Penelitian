import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
import statsmodels.graphics.tsaplots as sgt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Load data from CSV
df = pd.read_csv('nasa_power_bontang.csv', skiprows=18)  # Skip header to get to data rows

# Set column names
columns = ['PARAMETER', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANN']
df.columns = columns

# Filter data for the parameters we need
radiation_df = df[df['PARAMETER'] == 'ALLSKY_SFC_SW_DWN']
temp_df = df[df['PARAMETER'] == 'T2M']
humidity_df = df[df['PARAMETER'] == 'QV2M']
pressure_df = df[df['PARAMETER'] == 'PS']

# Melt the data to get monthly values
def melt_monthly_data(param_df, param_name):
    melted = param_df.melt(id_vars=['PARAMETER', 'YEAR'], 
                          value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
                          var_name='MONTH', value_name=param_name)
    melted['DATE'] = pd.to_datetime(melted['YEAR'].astype(str) + '-' + melted['MONTH'], format='%Y-%b')
    return melted[['DATE', param_name]].sort_values('DATE')

# Get data for each parameter
radiation_data = melt_monthly_data(radiation_df, 'radiation_kWh')
temp_data = melt_monthly_data(temp_df, 'temperature_C')
humidity_data = melt_monthly_data(humidity_df, 'humidity_gkg')
pressure_data = melt_monthly_data(pressure_df, 'pressure_kPa')

# Merge all data
merged_df = (radiation_data
             .merge(temp_data, on='DATE')
             .merge(humidity_data, on='DATE')
             .merge(pressure_data, on='DATE')
             .dropna())

# Calculate power output (simplified model)
# Using similar formula as in the original code
eta = 0.18  # efficiency
A = 6.67    # area m²
beta = 0.004  # temperature coefficient
T_ref = 25.0  # reference temperature

G_Wm2 = merged_df['radiation_kWh'] * 1000 / 12  # Convert to W/m² average
temp_factor = 1 - beta * (merged_df['temperature_C'] - T_ref)
merged_df['power_Watt'] = eta * A * G_Wm2 * temp_factor

# Add some realistic noise
np.random.seed(42)
noise = np.random.normal(0, 0.02, len(merged_df)) * merged_df['power_Watt']
merged_df['power_Watt'] = merged_df['power_Watt'] + noise

print(f"Total samples: {len(merged_df)}")
print(f"Date range: {merged_df['DATE'].min()} to {merged_df['DATE'].max()}")

# Prepare features and target
feature_columns = ['radiation_kWh', 'temperature_C', 'humidity_gkg', 'pressure_kPa']
target_column = 'power_Watt'

X = merged_df[feature_columns].values
y = merged_df[target_column].values

# Train/Test split with n=96 for training
n_train = 96
X_train = X[:n_train]
X_test = X[n_train:]
y_train = y[:n_train]
y_test = y[n_train:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Combine predictions
y_pred = np.concatenate([y_train_pred, y_test_pred])

# Calculate residuals
residuals = y - y_pred

# Metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

print(f"\nModel Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.4f}%")

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Visualisasi Validasi Train/Test Split: Perbandingan Y Aktual vs Prediksi', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted with train/test split
ax1.plot(range(len(y)), y, 'b-', label='Aktual (Training)', linewidth=2, alpha=0.8)
ax1.plot(range(len(y)), y_pred, 'r-', label='Prediksi (Test)', linewidth=2, alpha=0.8)
ax1.axvline(x=n_train-1, color='k', linestyle='--', linewidth=2, label=f'Batas Training (n={n_train})')
ax1.set_xlabel('Sampel')
ax1.set_ylabel('Daya (Watt)')
ax1.set_title('Perbandingan Y Aktual vs Prediksi')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot Actual vs Predicted
ax2.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training', edgecolors='k')
ax2.scatter(y_test, y_test_pred, alpha=0.6, color='red', label='Test', edgecolors='k')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', linewidth=2, label='Garis Ideal')
ax2.set_xlabel('Y Aktual (Watt)')
ax2.set_ylabel('Y Prediksi (Watt)')
ax2.set_title(f'Aktual vs Prediksi (R² = {r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals over time
ax3.plot(range(len(residuals)), residuals, 'g-', alpha=0.7)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax3.axvline(x=n_train-1, color='k', linestyle='--', linewidth=2, label=f'Batas Training (n={n_train})')
ax3.set_xlabel('Sampel')
ax3.set_ylabel('Residual (Watt)')
ax3.set_title('Residual Errors')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residual distribution
ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual (Watt)')
ax4.set_ylabel('Frekuensi')
ax4.set_title('Distribusi Residual')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('train_test_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Diagnostic plots
fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Diagnostik Residual', fontsize=16, fontweight='bold')

# Q-Q Plot
stats.probplot(residuals, dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot Residual (Normalitas)')
ax5.grid(True, alpha=0.3)

# Residual vs Fitted
ax6.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
ax6.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax6.set_xlabel('Fitted Values (Prediksi)')
ax6.set_ylabel('Residual')
ax6.set_title('Residual vs Fitted (Heteroskedastisitas)')
ax6.grid(True, alpha=0.3)

# ACF Plot
sgt.plot_acf(residuals, lags=30, ax=ax7, alpha=0.05)
ax7.set_title('ACF Residual (Autokorelasi)')
ax7.grid(True, alpha=0.3)

# Residual histogram with normal curve
ax8.hist(residuals, bins=20, density=True, alpha=0.7, edgecolor='black')
xmin, xmax = ax8.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
ax8.plot(x, p, 'k', linewidth=2)
ax8.set_xlabel('Residual')
ax8.set_ylabel('Density')
ax8.set_title('Histogram Residual dengan Kurva Normal')
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGrafik disimpan:")
print("- train_test_validation.png")
print("- residual_diagnostics.png")