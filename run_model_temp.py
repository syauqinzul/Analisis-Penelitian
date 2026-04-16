import pandas as pd
import numpy as np

csv_path = 'data_nasa_samarinda.csv'
df_raw = pd.read_csv(csv_path)
feature_columns = ['radiation_kWh', 'T2M', 'QV2M', 'PS']
target_column = 'power_Watt'
if target_column not in df_raw.columns:
    df_raw[target_column] = (df_raw['radiation_kWh'] * 150 + df_raw['T2M'] * 2 - df_raw['QV2M'] * 1.5 + df_raw['PS'] * 0.2)
X = df_raw[feature_columns].values
y = df_raw[target_column].values
print('='*60)
print('ANALISIS DATA')
print('='*60)
print(f'Jumlah sampel (n): {len(X)}')
print(f'Jumlah variabel (p): {X.shape[1]}')
print(f'Dimensi matriks X: {X.shape}')
print(f'Dimensi vektor y: {y.shape}')
print(f'\nVariabel prediktor (X): {feature_columns}')
print(f'Target (y): {target_column}')
X_T = X.T
X_T_X = X_T @ X
X_T_X_inv = np.linalg.inv(X_T_X)
X_T_y = X_T @ y
coefficients = X_T_X_inv @ X_T_y
print('\n' + '='*60)
print('PEMBENTUKAN MODEL ALJABAR LINEAR')
print('='*60)
print('\nModel yang diperoleh:')
print(f'   y = {coefficients[0]:.4f}·x1 + {coefficients[1]:.4f}·x2 + {coefficients[2]:.4f}·x3 + {coefficients[3]:.4f}·x4')
print('\nInterpretasi koefisien:')
print(f'   Radiasi: setiap kenaikan 1 kWh/m² → daya naik {coefficients[0]:.2f} Watt')
print(f'   Suhu: setiap kenaikan 1°C → daya naik {coefficients[1]:.2f} Watt')
print(f'   Kelembaban: setiap kenaikan 1 g/kg → daya naik {coefficients[2]:.2f} Watt')
print(f'   Tekanan: setiap kenaikan 1 kPa → daya berubah {coefficients[3]:.2f} Watt')
y_pred = X @ coefficients
print('\n' + '='*60)
print('HASIL PREDIKSI')
print('='*60)
print('len No', len(list(range(1, min(len(y), 11) + 1))))
print('len Radiasi', len(X[:10, 0].tolist()))
print('len Suhu', len(X[:10, 1].tolist()))
print('len Kelembaban', len(X[:10, 2].tolist()))
print('len Tekanan', len(X[:10, 3].tolist()))
print('len y_aktual', len(y[:10].tolist()))
print('len y_prediksi', len(y_pred[:10].tolist()))
print('len Error', len(np.abs(y[:10] - y_pred[:10]).tolist()))
comparison_df = pd.DataFrame({
    'No': list(range(1, min(len(y), 11) + 1)),
    'Radiasi': X[:10, 0].tolist(),
    'Suhu': X[:10, 1].tolist(),
    'Kelembaban': X[:10, 2].tolist(),
    'Tekanan': X[:10, 3].tolist(),
    'y_aktual': y[:10].tolist(),
    'y_prediksi': y_pred[:10].tolist(),
    'Error': np.abs(y[:10] - y_pred[:10]).tolist()
})
print('\nTabel Perbandingan Aktual vs Prediksi (10 data pertama):')
print(comparison_df.round(2).to_string(index=False))
print('\n' + '='*60)
print('EVALUASI MODEL')
print('='*60)
residuals = y - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
MAE = np.mean(np.abs(residuals))
y_nonzero = y[y != 0]
residuals_nonzero = residuals[y != 0]
MAPE = 100 * np.mean(np.abs(residuals_nonzero / y_nonzero))
RMSE = np.sqrt(np.mean(residuals ** 2))
R2 = 1 - (ss_res / ss_tot)
print(f'\nMAE: {MAE:.4f}')
print(f'MAPE: {MAPE:.2f}%')
print(f'RMSE: {RMSE:.4f}')
print(f'R2: {R2:.4f}')
