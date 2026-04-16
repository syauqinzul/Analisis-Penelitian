"""
MODEL PREDIKSI DAYA ENERGI SURYA
================================
Pendekatan Aljabar Linear dengan 4 variabel
Metode: Ordinary Least Squares (Normal Equation) [citation:3][citation:7]

Model: y = a·x₁ + b·x₂ + c·x₃ + d·x₄

Dimana:
x₁ = Intensitas radiasi matahari (kWh/m²/day)
x₂ = Suhu lingkungan (°C)
x₃ = Kelembaban spesifik (g/kg)
x₄ = Tekanan permukaan (kPa)
y  = Daya listrik (Watt)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PERSIAPAN DATA
# ============================================================

# Load data (gunakan df_raw dari hasil unduhan atau data sampel)
df = df_raw.copy()

# Pilih fitur (X) dan target (y)
feature_columns = ['radiation_kWh', 'temperature_C', 'humidity_gkg', 'pressure_kPa']
target_column = 'power_Watt'

X = df[feature_columns].values
y = df[target_column].values

print("="*60)
print("📊 ANALISIS DATA")
print("="*60)
print(f"Jumlah sampel (n): {len(X)}")
print(f"Jumlah variabel (p): {X.shape[1]}")
print(f"Dimensi matriks X: {X.shape}")
print(f"Dimensi vektor y: {y.shape}")
print(f"\nVariabel prediktor (X): {feature_columns}")
print(f"Target (y): {target_column}")

# ============================================================
# PEMBENTUKAN MODEL DENGAN NORMAL EQUATION
# ============================================================
# Metode: θ = (Xᵀ X)⁻¹ Xᵀ y [citation:3][citation:7]
# Ini adalah solusi closed-form untuk Ordinary Least Squares

print("\n" + "="*60)
print("🧮 PEMBENTUKAN MODEL ALJABAR LINEAR")
print("="*60)

# Tambahkan kolom bias (intercept) - optional
# Dalam model y = a·x₁ + b·x₂ + c·x₃ + d·x₄, kita tidak menambahkan intercept
# karena secara fisik, jika radiasi dan suhu = 0, daya juga 0

# Solusi dengan Normal Equation
# θ = (Xᵀ X)⁻¹ Xᵀ y
X_T = X.T
X_T_X = X_T @ X
X_T_X_inv = np.linalg.inv(X_T_X)  # Matriks invers
X_T_y = X_T @ y

coefficients = X_T_X_inv @ X_T_y

# Tampilkan model yang diperoleh
print("\n📐 Model yang diperoleh:")
print(f"   y = {coefficients[0]:.4f}·x₁ + {coefficients[1]:.4f}·x₂ + {coefficients[2]:.4f}·x₃ + {coefficients[3]:.4f}·x₄")

print("\nInterpretasi koefisien:")
print(f"   • Radiasi: setiap kenaikan 1 kWh/m² → daya naik {coefficients[0]:.2f} Watt")
print(f"   • Suhu: setiap kenaikan 1°C → daya naik {coefficients[1]:.2f} Watt")
print(f"   • Kelembaban: setiap kenaikan 1 g/kg → daya naik {coefficients[2]:.2f} Watt")
print(f"   • Tekanan: setiap kenaikan 1 kPa → daya berubah {coefficients[3]:.2f} Watt")

# ============================================================
# PREDIKSI
# ============================================================
y_pred = X @ coefficients

print("\n" + "="*60)
print("📈 HASIL PREDIKSI")
print("="*60)

# Tabel perbandingan (10 data pertama)
comparison_df = pd.DataFrame({
    'No': range(1, min(len(y), 11) + 1),
    'Radiasi': X[:10, 0],
    'Suhu': X[:10, 1],
    'Kelembaban': X[:10, 2],
    'Tekanan': X[:10, 3],
    'y_aktual': y[:10],
    'y_prediksi': y_pred[:10],
    'Error': np.abs(y[:10] - y_pred[:10])
})
print("\n📋 Tabel Perbandingan Aktual vs Prediksi (10 data pertama):")
print(comparison_df.round(2).to_string(index=False))

# ============================================================
# EVALUASI MODEL
# ============================================================
# Metrik evaluasi: MAE, MAPE, RMSE, R² [citation:4][citation:8]
print("\n" + "="*60)
print("📊 EVALUASI MODEL")
print("="*60)

n = len(y)
residuals = y - y_pred
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)

# 1. Mean Absolute Error (MAE)
MAE = np.mean(np.abs(residuals))

# 2. Mean Absolute Percentage Error (MAPE)
# Hindari division by zero
y_nonzero = y[y != 0]
residuals_nonzero = residuals[y != 0]
MAPE = 100 * np.mean(np.abs(residuals_nonzero / y_nonzero))

# 3. Root Mean Square Error (RMSE) [citation:8]
RMSE = np.sqrt(np.mean(residuals ** 2))

# 4. R-squared (Koefisien Determinasi) [citation:4]
R2 = 1 - (ss_res / ss_tot)

print(f"\n{'Metrik':<15} {'Nilai':<15} {'Interpretasi':<30}")
print("-" * 60)
print(f"{'MAE':<15} {MAE:.4f} {'Watt':<30}")
print(f"{'MAPE':<15} {MAPE:.2f}% {'<10% = Sangat Baik':<30}")
print(f"{'RMSE':<15} {RMSE:.4f} {'Watt':<30}")
print(f"{'R²':<15} {R2:.4f} {'1 = prediksi sempurna':<30}")

# Interpretasi R²
print("\n🔍 Interpretasi R²:")
if R2 >= 0.9:
    print(f"   R² = {R2:.4f} → Model menjelaskan {(R2*100):.1f}% variasi data (sangat baik)")
elif R2 >= 0.7:
    print(f"   R² = {R2:.4f} → Model menjelaskan {(R2*100):.1f}% variasi data (baik)")
elif R2 >= 0.5:
    print(f"   R² = {R2:.4f} → Model menjelaskan {(R2*100):.1f}% variasi data (cukup)")
else:
    print(f"   R² = {R2:.4f} → Model kurang menjelaskan variasi data")

# ============================================================
# PERBANDINGAN DENGAN PENELITIAN TERDAHULU
# ============================================================
print("\n" + "="*60)
print("📚 PERBANDINGAN DENGAN PENELITIAN TERDAHULU")
print("="*60)

print("\nPenelitian Pabate (2025) - Regresi Linear Berganda PLTS:")
print("   • MAE: tidak dilaporkan")
print("   • MAPE: tidak dilaporkan")
print("   • Metode: Weka 3.8")

print("\nPenelitian Shirazi et al. (2024) - IEEE Journal of Photovoltaics:")
print("   • RMSE: 0.05 kW/m² untuk prediksi iradiasi")
print("   • Metode: Lasso Regression")

print("\nPenelitian ini (Samarinda, 2024):")
print(f"   • MAE: {MAE:.4f} Watt")
print(f"   • MAPE: {MAPE:.2f}%")
print(f"   • RMSE: {RMSE:.4f} Watt")
print(f"   • R²: {R2:.4f}")
print(f"   • Metode: Normal Equation (XᵀX)⁻¹Xᵀy")

# ============================================================
# ANALISIS DOMINANSI VARIABEL
# ============================================================
print("\n" + "="*60)
print("🔍 ANALISIS DOMINANSI VARIABEL")
print("="*60)

# Normalisasi koefisien untuk melihat kontribusi relatif
coef_abs = np.abs(coefficients)
coef_normalized = coef_abs / np.sum(coef_abs) * 100

print("\nKontribusi relatif masing-masing variabel:")
for i, col in enumerate(feature_columns):
    print(f"   • {col:<20}: {coef_normalized[i]:.1f}%")
    
# Tentukan variabel dominan
dominant_idx = np.argmax(coef_normalized)
print(f"\n⭐ Variabel DOMINAN: {feature_columns[dominant_idx]}")
print(f"   (kontribusi {coef_normalized[dominant_idx]:.1f}% terhadap prediksi)")

