"""
=============================================================================
ANALISIS DAN EVALUASI MODEL PREDIKSI DAYA ENERGI SURYA
MENGGUNAKAN PENDEKATAN ALJABAR LINEAR BERBASIS DATA NASA
DAN IMPLEMENTASI PYTHON

Lokasi  : Samarinda, Kalimantan Timur
         Lat 0.1333°N | Lon 117.50°E
Periode : Januari 2020 – Desember 2024 (60 sampel bulanan)
Sumber  : NASA POWER – CERES SYN1deg & MERRA-2

Variabel:
  x1 = ALLSKY_SFC_SW_DWN  → Intensitas Radiasi (kWh/m²/hari)
  x2 = T2M                 → Suhu Lingkungan (°C)
  y  = Daya Listrik (Watt) — dihitung via rumus panel surya standar

Model  : Regresi Linear Berganda (OLS — Normal Equation)
         koef = (X^T X)^{-1} X^T Y
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA NASA POWER — SAMARINDA (Lat: 0.1333°N | Lon: 117.50°E)
# ─────────────────────────────────────────────────────────────────────────────

bulan_label = [
    "Jan-20","Feb-20","Mar-20","Apr-20","Mei-20","Jun-20",
    "Jul-20","Agu-20","Sep-20","Okt-20","Nov-20","Des-20",
    "Jan-21","Feb-21","Mar-21","Apr-21","Mei-21","Jun-21",
    "Jul-21","Agu-21","Sep-21","Okt-21","Nov-21","Des-21",
    "Jan-22","Feb-22","Mar-22","Apr-22","Mei-22","Jun-22",
    "Jul-22","Agu-22","Sep-22","Okt-22","Nov-22","Des-22",
    "Jan-23","Feb-23","Mar-23","Apr-23","Mei-23","Jun-23",
    "Jul-23","Agu-23","Sep-23","Okt-23","Nov-23","Des-23",
    "Jan-24","Feb-24","Mar-24","Apr-24","Mei-24","Jun-24",
    "Jul-24","Agu-24","Sep-24","Okt-24","Nov-24","Des-24",
]

# x1: Intensitas Radiasi Matahari (kWh/m²/hari) — ALLSKY_SFC_SW_DWN
x1 = np.array([
    # 2020
    5.1163, 5.6292, 5.6189, 5.5099, 4.9217, 4.4110,
    4.4004, 5.0062, 4.5758, 4.5946, 4.8233, 4.5149,
    # 2021
    4.1858, 4.9368, 4.9553, 4.9894, 4.6190, 4.6723,
    4.4412, 4.6627, 4.7107, 4.9865, 4.4244, 4.0805,
    # 2022
    4.6224, 4.7448, 4.9200, 4.8415, 4.7114, 4.8958,
    4.2365, 4.6968, 4.7544, 4.6066, 4.6265, 4.3896,
    # 2023
    4.1479, 4.4117, 4.8358, 4.9121, 4.6111, 4.4189,
    4.5787, 5.0652, 5.1734, 5.5207, 4.9087, 5.0638,
    # 2024
    4.8684, 5.4365, 5.4485, 5.7012, 4.7258, 4.1088,
    4.3018, 4.3037, 5.3138, 5.0810, 4.8744, 4.1318,
])

# x2: Suhu Lingkungan (°C) — T2M MERRA-2
x2 = np.array([
    # 2020
    27.11, 27.13, 27.26, 27.44, 27.94, 27.01,
    26.67, 26.94, 26.84, 27.05, 27.21, 27.02,
    # 2021
    26.73, 26.85, 26.91, 27.19, 27.52, 27.25,
    26.57, 26.82, 26.73, 27.25, 27.08, 26.98,
    # 2022
    26.52, 26.46, 26.99, 27.21, 27.49, 27.00,
    26.84, 26.63, 26.94, 27.00, 27.15, 26.98,
    # 2023
    26.77, 26.68, 26.88, 27.46, 27.84, 27.32,
    27.09, 27.27, 27.63, 28.32, 27.95, 27.62,
    # 2024
    27.32, 27.56, 27.58, 28.10, 28.22, 27.40,
    27.09, 27.12, 27.45, 27.50, 27.55, 27.40,
])

# ─────────────────────────────────────────────────────────────────────────────
# 2. HITUNG DAYA LISTRIK (Y) — Formula Panel Surya Standar
#
#    P = η × A × G_day × H × (1 - β × (T - T_ref))
#
#    Asumsi panel surya 1 kWp (tipikal instalasi residensial/penelitian):
#      η   = 0.18        → efisiensi modul 18%
#      A   = 6.67 m²     → luas panel (untuk 1 kWp @ STC 1000 W/m²)
#      H   = 5 jam/hari  → peak sun hours rata-rata
#      β   = 0.004 /°C   → koefisien temperatur daya (Si kristal)
#      T_ref = 25°C      → suhu referensi STC
#      Hari per bulan rata-rata = 30.44
#
#    Satuan: G_day (kWh/m²/hari) × 1000 = G (Wh/m²/hari)
#    Output Y dalam Watt (daya rata-rata harian)
# ─────────────────────────────────────────────────────────────────────────────

eta   = 0.18          # efisiensi panel
A     = 6.67          # luas panel (m²)
beta  = 0.004         # koefisien suhu (/°C)
T_ref = 25.0          # suhu referensi STC (°C)

# Konversi irradiasi ke W/m² rata-rata harian (÷ jam siang ≈ 12 jam)
G_Wm2 = x1 * 1000 / 12   # kWh/m²/hari → W/m² rata-rata

# Faktor koreksi suhu
temp_factor = 1 - beta * (x2 - T_ref)

# Daya output (Watt)
Y = eta * A * G_Wm2 * temp_factor

# Tambah sedikit noise realistis (± 2%) untuk simulasi pengukuran lapangan
np.random.seed(42)
noise = np.random.normal(0, 0.02, len(Y)) * Y
Y = Y + noise

print("=" * 65)
print("  ANALISIS MODEL PREDIKSI DAYA ENERGI SURYA")
print("  Lokasi: Samarinda, Kalimantan Timur")
print("  Sumber: NASA POWER | Periode: 2020–2024 (60 bulan)")
print("=" * 65)

print("\n[PREVIEW DATA — 5 Sampel Pertama]")
print(f"  {'Bulan':<10} {'Radiasi (x1)':>14} {'Suhu (x2)':>12} {'Daya Y':>12}")
print(f"  {'':10} {'kWh/m²/hari':>14} {'°C':>12} {'Watt':>12}")
print("  " + "-" * 52)
for i in range(5):
    print(f"  {bulan_label[i]:<10} {x1[i]:>14.4f} {x2[i]:>12.2f} {Y[i]:>12.2f}")
print(f"  {'...':10}")
print(f"\n  Total sampel: {len(Y)}")
print(f"  Radiasi  → min={x1.min():.4f}, max={x1.max():.4f}, rata-rata={x1.mean():.4f} kWh/m²/hari")
print(f"  Suhu     → min={x2.min():.2f}, max={x2.max():.2f}, rata-rata={x2.mean():.2f} °C")
print(f"  Daya (Y) → min={Y.min():.2f}, max={Y.max():.2f}, rata-rata={Y.mean():.2f} Watt")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MATRIKS DESAIN X  [1 | x1 | x2]  (n×3)
# ─────────────────────────────────────────────────────────────────────────────

n    = len(Y)
ones = np.ones((n, 1))
X    = np.hstack([ones, x1.reshape(-1, 1), x2.reshape(-1, 1)])

print("\n" + "─" * 65)
print("  MATRIKS DESAIN X  (shape: {} × {})".format(*X.shape))
print("─" * 65)
print("  Kolom: [Bias(1) | Radiasi(x1) | Suhu(x2)]")
print(f"  Contoh baris pertama: {X[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMAL EQUATION — koef = (X^T X)^{-1} X^T Y
# ─────────────────────────────────────────────────────────────────────────────

XtX     = X.T @ X
XtX_inv = np.linalg.inv(XtX)
XtY     = X.T @ Y
koef    = XtX_inv @ XtY

a0, a1, a2 = koef

print("\n" + "─" * 65)
print("  HASIL ESTIMASI KOEFISIEN (Normal Equation)")
print("─" * 65)
print(f"  a0 (intercept)          = {a0:+.6f}")
print(f"  a1 (koef. Radiasi x1)   = {a1:+.6f}  Watt per kWh/m²/hari")
print(f"  a2 (koef. Suhu    x2)   = {a2:+.6f}  Watt per °C")
print(f"\n  Model Persamaan:")
print(f"  Ŷ = ({a0:.4f}) + ({a1:.4f})·x1 + ({a2:.4f})·x2")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDIKSI & EVALUASI
# ─────────────────────────────────────────────────────────────────────────────

Y_pred  = X @ koef
resid   = Y - Y_pred

SS_res  = np.sum(resid**2)
SS_tot  = np.sum((Y - np.mean(Y))**2)
R2      = 1 - SS_res / SS_tot
R2_adj  = 1 - (1 - R2) * (n - 1) / (n - 3)   # k=2 fitur
MAE     = np.mean(np.abs(resid))
RMSE    = np.sqrt(np.mean(resid**2))
MAPE    = np.mean(np.abs(resid / Y)) * 100

print("\n" + "─" * 65)
print("  EVALUASI MODEL")
print("─" * 65)
print(f"  R²          (Koefisien Determinasi)  = {R2:.6f}  ({R2*100:.4f}%)")
print(f"  R² Adjusted                           = {R2_adj:.6f}  ({R2_adj*100:.4f}%)")
print(f"  MAE         (Mean Absolute Error)     = {MAE:.4f} Watt")
print(f"  RMSE        (Root Mean Squared Error) = {RMSE:.4f} Watt")
print(f"  MAPE        (Mean Abs. Pct. Error)    = {MAPE:.4f} %")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TABEL HASIL LENGKAP
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 65)
print("  TABEL HASIL PREDIKSI LENGKAP")
print("─" * 65)
print(f"  {'Bulan':<10} {'x1':>8} {'x2':>7} {'Y Aktual':>12} {'Y Pred':>12} {'Selisih':>10}")
print("  " + "-" * 63)
for i in range(n):
    print(f"  {bulan_label[i]:<10} {x1[i]:>8.4f} {x2[i]:>7.2f} "
          f"{Y[i]:>12.4f} {Y_pred[i]:>12.4f} {resid[i]:>+10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PREDIKSI DATA HIPOTETIS BARU
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 65)
print("  PREDIKSI SKENARIO BARU (Hipotetis)")
print("─" * 65)
skenario = np.array([
    [1, 4.0, 26.5],
    [1, 4.5, 27.0],
    [1, 5.0, 27.5],
    [1, 5.5, 28.0],
    [1, 6.0, 28.5],
    [1, 6.5, 29.0],
])
Y_new = skenario @ koef
print(f"  {'Radiasi (x1)':>14}  {'Suhu (x2)':>10}  {'Prediksi Daya':>15}")
print(f"  {'kWh/m²/hari':>14}  {'°C':>10}  {'Watt':>15}")
print("  " + "-" * 44)
for row, pred in zip(skenario, Y_new):
    print(f"  {row[1]:>14.1f}  {row[2]:>10.1f}  {pred:>15.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("dark_background")
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"
GOLD    = "#e3a02a"
CYAN    = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
PURPLE  = "#bc8cff"

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle(
    "Analisis Model Prediksi Daya Energi Surya — Samarinda, Kalimantan Timur\n"
    "Data NASA POWER | Jan 2020 – Des 2024 | Regresi Linear OLS",
    fontsize=13, color=TEXT, fontweight="bold", y=0.98
)

gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)
ax1 = fig.add_subplot(gs[0, :])    # Aktual vs Prediksi (full width)
ax2 = fig.add_subplot(gs[1, 0])    # Scatter radiasi vs Y
ax3 = fig.add_subplot(gs[1, 1])    # Scatter suhu vs Y
ax4 = fig.add_subplot(gs[1, 2])    # Residual bar
ax5 = fig.add_subplot(gs[2, 0])    # Histogram residual
ax6 = fig.add_subplot(gs[2, 1])    # Y aktual vs Y prediksi (diagonal)
ax7 = fig.add_subplot(gs[2, 2])    # Radiasi per tahun

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

idx   = np.arange(n)
years = [2020, 2021, 2022, 2023, 2024]

# ── Plot 1: Time Series Aktual vs Prediksi ──────────────────────────────────
ax1.plot(idx, Y,      color=CYAN,  lw=1.6, label="Y Aktual",   zorder=3)
ax1.plot(idx, Y_pred, color=GOLD,  lw=1.6, label="Ŷ Prediksi",
         ls="--", zorder=4)
ax1.fill_between(idx, Y, Y_pred, alpha=0.12, color=RED)
# Garis pemisah tahun
for yr_idx in [12, 24, 36, 48]:
    ax1.axvline(yr_idx, color=BORDER, lw=0.8, ls=":")
# Label tahun
for i, yr in enumerate(years):
    ax1.text(i*12 + 5.5, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else Y.min()-5,
             str(yr), fontsize=7.5, color=MUTED, ha="center")
ax1.set_title("Daya Aktual vs Prediksi — Time Series 60 Bulan")
ax1.set_xlabel("Indeks Bulan")
ax1.set_ylabel("Daya Output (Watt)")
xtick_pos = np.arange(0, 60, 6)
ax1.set_xticks(xtick_pos)
ax1.set_xticklabels([bulan_label[i] for i in xtick_pos], rotation=35, ha="right", fontsize=6.5)
ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, loc="upper right")
ax1.text(0.01, 0.96, f"R² = {R2:.4f}  |  RMSE = {RMSE:.2f} W  |  MAPE = {MAPE:.2f}%",
         transform=ax1.transAxes, ha="left", va="top",
         fontsize=8, color=GREEN,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, alpha=0.8))

# ── Plot 2: Radiasi vs Daya ──────────────────────────────────────────────────
ax2.scatter(x1, Y,      color=CYAN,   s=25, alpha=0.7, zorder=3, label="Aktual")
ax2.scatter(x1, Y_pred, color=GOLD,   s=15, alpha=0.7, marker="^", zorder=4, label="Prediksi")
sort_i = np.argsort(x1)
ax2.plot(x1[sort_i], Y_pred[sort_i], color=GOLD, lw=1.2, ls="--", alpha=0.5)
ax2.set_title("Radiasi vs Daya")
ax2.set_xlabel("Radiasi x1 (kWh/m²/hari)")
ax2.set_ylabel("Daya (Watt)")
ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

# ── Plot 3: Suhu vs Daya ─────────────────────────────────────────────────────
ax3.scatter(x2, Y,      color=PURPLE, s=25, alpha=0.7, zorder=3, label="Aktual")
ax3.scatter(x2, Y_pred, color=GOLD,   s=15, alpha=0.7, marker="^", zorder=4, label="Prediksi")
sort_j = np.argsort(x2)
ax3.plot(x2[sort_j], Y_pred[sort_j], color=GOLD, lw=1.2, ls="--", alpha=0.5)
ax3.set_title("Suhu vs Daya")
ax3.set_xlabel("Suhu x2 (°C)")
ax3.set_ylabel("Daya (Watt)")
ax3.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

# ── Plot 4: Residual per Sampel ──────────────────────────────────────────────
colors_bar = [GREEN if r >= 0 else RED for r in resid]
ax4.bar(idx, resid, color=colors_bar, width=0.7, alpha=0.85)
ax4.axhline(0, color=MUTED, lw=1, ls="--")
ax4.set_title("Residual (Y − Ŷ)")
ax4.set_xlabel("Indeks Bulan")
ax4.set_ylabel("Residual (Watt)")
ax4.set_xlim(-1, n)

# ── Plot 5: Distribusi Residual ──────────────────────────────────────────────
ax5.hist(resid, bins=12, color=CYAN, edgecolor=PANEL, alpha=0.85)
ax5.axvline(0,         color=RED,   lw=1.5, ls="--", label="Nol")
ax5.axvline(resid.mean(), color=GOLD, lw=1.5, ls="-.", label=f"Mean={resid.mean():.2f}")
ax5.set_title("Distribusi Residual")
ax5.set_xlabel("Residual (Watt)")
ax5.set_ylabel("Frekuensi")
ax5.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

# ── Plot 6: Y Aktual vs Y Prediksi (diagonal) ────────────────────────────────
min_val = min(Y.min(), Y_pred.min()) - 2
max_val = max(Y.max(), Y_pred.max()) + 2
ax6.scatter(Y, Y_pred, color=CYAN, s=20, alpha=0.75, zorder=3)
ax6.plot([min_val, max_val], [min_val, max_val], color=GOLD, lw=1.5, ls="--", label="Ideal (y=x)")
ax6.set_title("Y Aktual vs Ŷ Prediksi")
ax6.set_xlabel("Y Aktual (Watt)")
ax6.set_ylabel("Ŷ Prediksi (Watt)")
ax6.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax6.text(0.05, 0.93, f"R²={R2:.4f}", transform=ax6.transAxes,
         fontsize=8, color=GREEN)

# ── Plot 7: Rata-rata Radiasi per Tahun ──────────────────────────────────────
rad_yr  = x1.reshape(5, 12).mean(axis=1)
daya_yr = Y.reshape(5, 12).mean(axis=1)
x_yr    = np.arange(5)
ax7b    = ax7.twinx()
ax7.bar(x_yr - 0.2, rad_yr,  width=0.35, color=CYAN,   alpha=0.8, label="Radiasi avg")
ax7b.bar(x_yr + 0.2, daya_yr, width=0.35, color=GOLD,   alpha=0.8, label="Daya avg")
ax7.set_title("Rata-rata Tahunan")
ax7.set_xlabel("Tahun")
ax7.set_ylabel("Radiasi (kWh/m²/hari)", color=CYAN)
ax7b.set_ylabel("Daya (Watt)", color=GOLD)
ax7.set_xticks(x_yr)
ax7.set_xticklabels(years, fontsize=8)
ax7.tick_params(axis="y", colors=CYAN)
ax7b.tick_params(axis="y", colors=GOLD)
ax7b.set_facecolor(PANEL)
for sp in ax7b.spines.values():
    sp.set_color(BORDER)
lines1, labels1 = ax7.get_legend_handles_labels()
lines2, labels2 = ax7b.get_legend_handles_labels()
ax7.legend(lines1+lines2, labels1+labels2, fontsize=6.5,
           facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

plt.savefig("/mnt/user-data/outputs/solar_samarinda_plot.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("\n[INFO] Plot disimpan → solar_samarinda_plot.png")
print("\n" + "=" * 65)
print("  SELESAI — Implementasi berhasil dijalankan.")
print("=" * 65)
plt.show()