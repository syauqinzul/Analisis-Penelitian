"""
VISUALISASI HASIL MODEL
=======================
Grafik perbandingan aktual vs prediksi, residual, dan korelasi
"""

import matplotlib.pyplot as plt
import seaborn as sns

from solar_energi_bontang import RMSE

# Set style untuk publikasi
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ============================================================
# GRAFIK 1: Scatter Plot Aktual vs Prediksi
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Analisis Model Prediksi Daya Energi Surya\nSamarinda, Kalimantan Timur', fontsize=14, fontweight='bold')

# Plot 1: Aktual vs Prediksi
ax1 = axes[0, 0]
ax1.scatter(y, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Garis Ideal (y=x)')
ax1.set_xlabel('Daya Aktual (Watt)')
ax1.set_ylabel('Daya Prediksi (Watt)')
ax1.set_title(f'Aktual vs Prediksi (R² = {R2:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Line Chart Perbandingan
ax2 = axes[0, 1]
indices = range(len(y))
ax2.plot(indices, y, 'o-', label='Aktual', markersize=4, linewidth=1.5)
ax2.plot(indices, y_pred, 's-', label='Prediksi', markersize=4, linewidth=1.5)
ax2.set_xlabel('Urutan Sampel')
ax2.set_ylabel('Daya (Watt)')
ax2.set_title('Perbandingan Daya Aktual vs Prediksi')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residual Error
ax3 = axes[1, 0]
residuals = y - y_pred
ax3.bar(indices, residuals, alpha=0.7, edgecolors='k', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
ax3.set_xlabel('Urutan Sampel')
ax3.set_ylabel('Residual Error (Watt)')
ax3.set_title(f'Residual Error (MAE = {MAE:.2f} Watt)')
ax3.grid(True, alpha=0.3)

# Plot 4: Heatmap Korelasi
ax4 = axes[1, 1]
df_corr = df[feature_columns + [target_column]].copy()
df_corr.columns = ['Radiasi', 'Suhu', 'Kelembaban', 'Tekanan', 'Daya']
corr_matrix = df_corr.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', square=True, ax=ax4, cbar_kws={'shrink': 0.8})
ax4.set_title('Heatmap Korelasi Antar Variabel')

plt.tight_layout()
plt.savefig('hasil_analisis_energi_surya.png', dpi=300, bbox_inches='tight')
print("\n💾 Grafik disimpan ke: hasil_analisis_energi_surya.png")
plt.show()

# ============================================================
# GRAFIK TAMBAHAN: Distribusi Error
# ============================================================
fig2, ax = plt.subplots(figsize=(10, 5))
ax.hist(residuals, bins=15, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Error = 0')
ax.set_xlabel('Residual Error (Watt)')
ax.set_ylabel('Frekuensi')
ax.set_title(f'Distribusi Residual Error (RMSE = {RMSE:.2f} Watt)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribusi_error.png', dpi=300, bbox_inches='tight')
print("💾 Grafik disimpan ke: distribusi_error.png")
plt.show()

