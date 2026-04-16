"""
=============================================================================
ANALISIS DAN EVALUASI MODEL PREDIKSI DAYA ENERGI SURYA
MENGGUNAKAN PENDEKATAN ALJABAR LINEAR BERBASIS DATA NASA
DAN IMPLEMENTASI PYTHON
=============================================================================
Metode: Ordinary Least Squares (OLS) via Normal Equation
         koef = (X^T X)^{-1} X^T Y
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA  (simulasi data berbasis NASA POWER / PVGIS)
#    Kolom X : [irradiasi (kWh/m²/hari), suhu (°C)]
#    Y       : daya output panel surya (Watt)
# ─────────────────────────────────────────────────────────────────────────────

# Data latih (training data)
X_raw = np.array([
    [4.5, 27.5],
    [5.5, 28.2],
    [5.8, 28.6],
    [5.7, 28.5],
    [5.6, 28.3],
    [6.0, 29.0],
    [6.2, 29.3],
    [5.0, 27.8],
    [4.8, 27.2],
    [6.5, 30.1],
    [4.2, 26.5],
    [5.3, 28.0],
])

Y = np.array([220, 270, 295, 285, 280, 310, 320, 245, 235, 340, 205, 265])

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING: TAMBAHKAN KOLOM BIAS (intercept)
#    X = [1, irradiasi, suhu]  ← bentuk matriks desain
# ─────────────────────────────────────────────────────────────────────────────

n = X_raw.shape[0]
ones = np.ones((n, 1))
X = np.hstack([ones, X_raw])          # shape: (n, 3)

print("=" * 60)
print("  ANALISIS MODEL PREDIKSI DAYA ENERGI SURYA")
print("  Pendekatan Aljabar Linear — Normal Equation")
print("=" * 60)
print(f"\n[INFO] Jumlah data latih : {n} sampel")
print(f"[INFO] Fitur             : Bias, Irradiasi (kWh/m²/hari), Suhu (°C)")
print(f"[INFO] Target            : Daya Output (Watt)\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ESTIMASI KOEFISIEN — NORMAL EQUATION
#    koef = (X^T X)^{-1} X^T Y
# ─────────────────────────────────────────────────────────────────────────────

XtX     = X.T @ X                       # (3×n)(n×3) = (3×3)
XtX_inv = np.linalg.inv(XtX)            # (3×3)^{-1}
XtY     = X.T @ Y                       # (3×n)(n,)  = (3,)
koef    = XtX_inv @ XtY                 # (3,)

a0, a1, a2 = koef                       # intercept, koef irradiasi, koef suhu

print("─" * 60)
print("  HASIL ESTIMASI KOEFISIEN (Normal Equation)")
print("─" * 60)
print(f"  a0 (intercept)    = {a0:+.4f}")
print(f"  a1 (irradiasi)    = {a1:+.4f}  Watt per kWh/m²/hari")
print(f"  a2 (suhu)         = {a2:+.4f}  Watt per °C")
print(f"\n  Model: Y = {a0:.2f} + {a1:.2f}·Irradiasi + ({a2:.2f})·Suhu")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDIKSI & EVALUASI
# ─────────────────────────────────────────────────────────────────────────────

Y_pred = X @ koef

# Metrik evaluasi
residuals   = Y - Y_pred
SS_res      = np.sum(residuals**2)
SS_tot      = np.sum((Y - np.mean(Y))**2)
R2          = 1 - SS_res / SS_tot
MAE         = np.mean(np.abs(residuals))
RMSE        = np.sqrt(np.mean(residuals**2))
MAPE        = np.mean(np.abs(residuals / Y)) * 100

print("\n" + "─" * 60)
print("  EVALUASI MODEL (Data Latih)")
print("─" * 60)
print(f"  R²   (koefisien determinasi) = {R2:.6f}  ({R2*100:.2f}%)")
print(f"  MAE  (Mean Absolute Error)   = {MAE:.4f} Watt")
print(f"  RMSE (Root Mean Sq. Error)   = {RMSE:.4f} Watt")
print(f"  MAPE (Mean Abs. Pct. Error)  = {MAPE:.4f} %")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDIKSI DATA BARU
# ─────────────────────────────────────────────────────────────────────────────

data_baru = np.array([
    [1, 5.0, 28.0],
    [1, 6.1, 29.5],
    [1, 4.3, 26.8],
    [1, 7.0, 31.0],
])

Y_baru = data_baru @ koef

print("\n" + "─" * 60)
print("  PREDIKSI DATA BARU")
print("─" * 60)
print(f"  {'Irradiasi':>12}  {'Suhu':>8}  {'Prediksi Daya':>15}")
print(f"  {'(kWh/m²/hari)':>12}  {'(°C)':>8}  {'(Watt)':>15}")
print("  " + "-" * 40)
for row, pred in zip(data_baru, Y_baru):
    print(f"  {row[1]:>12.1f}  {row[2]:>8.1f}  {pred:>15.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────

plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
fig.suptitle(
    "Prediksi Daya Energi Surya — Pendekatan Aljabar Linear (OLS)\n"
    "Berbasis Data NASA | Implementasi Python",
    fontsize=14, color="#e6edf3", y=0.98, fontweight="bold"
)

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])   # Aktual vs Prediksi
ax2 = fig.add_subplot(gs[0, 2])    # Residual Plot
ax3 = fig.add_subplot(gs[1, 0])    # Scatter: Irradiasi vs Daya
ax4 = fig.add_subplot(gs[1, 1])    # Scatter: Suhu vs Daya
ax5 = fig.add_subplot(gs[1, 2])    # Distribusi Residual

GOLD   = "#f0a500"
CYAN   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
PANEL  = "#161b22"

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")

idx = np.arange(n)

# ── Plot 1: Aktual vs Prediksi ──
ax1.plot(idx, Y,      "o-",  color=CYAN,  lw=1.8, ms=6, label="Aktual (Y)")
ax1.plot(idx, Y_pred, "s--", color=GOLD,  lw=1.8, ms=5, label="Prediksi (Ŷ)")
ax1.fill_between(idx, Y, Y_pred, alpha=0.15, color=RED)
ax1.set_title("Daya Aktual vs Prediksi")
ax1.set_xlabel("Indeks Data")
ax1.set_ylabel("Daya Output (Watt)")
ax1.legend(fontsize=8, facecolor=PANEL, edgecolor="#30363d", labelcolor="#e6edf3")
ax1.text(0.98, 0.05, f"R² = {R2:.4f}\nRMSE = {RMSE:.2f} W",
         transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=8, color=GREEN,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117", alpha=0.7))

# ── Plot 2: Residual ──
ax2.bar(idx, residuals, color=[GREEN if r >= 0 else RED for r in residuals], width=0.6)
ax2.axhline(0, color="#8b949e", lw=1, ls="--")
ax2.set_title("Residual (Y − Ŷ)")
ax2.set_xlabel("Indeks Data")
ax2.set_ylabel("Residual (Watt)")

# ── Plot 3: Irradiasi vs Daya ──
ax3.scatter(X_raw[:, 0], Y,      color=CYAN, s=60, zorder=3, label="Aktual")
ax3.scatter(X_raw[:, 0], Y_pred, color=GOLD, s=40, marker="^", zorder=4, label="Prediksi")
sort_idx = np.argsort(X_raw[:, 0])
ax3.plot(X_raw[sort_idx, 0], Y_pred[sort_idx], color=GOLD, lw=1.2, ls="--", alpha=0.6)
ax3.set_title("Irradiasi vs Daya")
ax3.set_xlabel("Irradiasi (kWh/m²/hari)")
ax3.set_ylabel("Daya (Watt)")
ax3.legend(fontsize=7, facecolor=PANEL, edgecolor="#30363d", labelcolor="#e6edf3")

# ── Plot 4: Suhu vs Daya ──
ax4.scatter(X_raw[:, 1], Y,      color=CYAN, s=60, zorder=3, label="Aktual")
ax4.scatter(X_raw[:, 1], Y_pred, color=GOLD, s=40, marker="^", zorder=4, label="Prediksi")
sort_idx2 = np.argsort(X_raw[:, 1])
ax4.plot(X_raw[sort_idx2, 1], Y_pred[sort_idx2], color=GOLD, lw=1.2, ls="--", alpha=0.6)
ax4.set_title("Suhu vs Daya")
ax4.set_xlabel("Suhu (°C)")
ax4.set_ylabel("Daya (Watt)")
ax4.legend(fontsize=7, facecolor=PANEL, edgecolor="#30363d", labelcolor="#e6edf3")

# ── Plot 5: Distribusi Residual ──
ax5.hist(residuals, bins=6, color=CYAN, edgecolor=PANEL, alpha=0.85)
ax5.axvline(0, color=RED, lw=1.5, ls="--")
ax5.set_title("Distribusi Residual")
ax5.set_xlabel("Residual (Watt)")
ax5.set_ylabel("Frekuensi")

plt.savefig("/mnt/user-data/outputs/solar_prediction_plot.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")

print("\n[INFO] Plot disimpan → solar_prediction_plot.png")
print("\n" + "=" * 60)
print("  SELESAI — Semua tahap implementasi berhasil dijalankan.")
print("=" * 60)

plt.show()