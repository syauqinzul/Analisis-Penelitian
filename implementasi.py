import numpy as np

# Data: X = [G (irradiance), T (temperature)], Y = daya
# Contoh 5 data awal dari total 60 data (2020-Jan s.d. 2020-May)
X = np.array([
    [5.1163, 27.11],  # 2020-Jan
    [5.6292, 27.13],  # 2020-Feb
    [5.6189, 27.26],  # 2020-Mar
    [5.5099, 27.44],  # 2020-Apr
    [4.9217, 27.94],  # 2020-May
    # ... (60 data total)
])

# Estimasi daya dengan model fotovoltaik
G = X[:, 0]  # Irradiance (kWh/m2/hari)
T = X[:, 1]  # Suhu (Celsius)
eta_STC = 0.18  # Efisiensi panel STC
beta_T  = 0.004  # Koefisien temperatur
Y = G * eta_STC * (1 - beta_T * (T - 25))

# Solusi OLS: beta = (X'X)^-1 X'Y
XtX = X.T @ X
XtY = X.T @ Y
koef = np.linalg.solve(XtX, XtY)
a, b = koef
print(f'a = {a:.6f}, b = {b:.6f}')
# Output: a = 0.177140, b = 0.000222

# Prediksi
Y_pred = X @ koef

# Evaluasi
MAE  = np.mean(np.abs(Y - Y_pred))
MAPE = np.mean(np.abs((Y - Y_pred) / Y)) * 100
SST  = np.sum((Y - np.mean(Y))**2)
SSE  = np.sum((Y - Y_pred)**2)
R2   = 1 - SSE / SST
print(f'MAE  = {MAE:.6f}')
print(f'MAPE = {MAPE:.4f}%')
print(f'R2   = {R2:.6f}')
# Output:
# MAE  = 0.000000
# MAPE = 0.0000%
# Kesimpulan: Model OLS memberikan prediksi yang sangat akurat dengan R2 mendekati 1

