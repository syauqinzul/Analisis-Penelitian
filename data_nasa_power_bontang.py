"""
================================================================================
PENGUNDUHAN DATA NASA POWER – WILAYAH BONTANG
================================================================================
Lokasi        : Bontang, Kalimantan Timur
Koordinat     : 0.1333° LU, 117.5000° BT
Periode       : 2020 - 2024 (5 tahun / 60 bulan)
Temporal      : Monthly (rata-rata bulanan)
Community     : Renewable Energy (RE)
Parameter     : Radiasi, Suhu, Kelembaban, Kecepatan Angin

Parameter NASA POWER yang digunakan:
1. ALLSKY_SFC_SW_DWN  → Radiasi matahari (MJ/m²/day) → dikonversi ke kWh/m²/day
2. T2M                → Suhu udara 2 meter (°C)
3. QV2M               → Kelembaban spesifik 2 meter (g/kg)
4. WS10M              → Kecepatan angin 10 meter (m/s)

Referensi dokumentasi:
- https://power.larc.nasa.gov/docs/services/api/
- https://power.larc.nasa.gov/parameters/
================================================================================
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI WILAYAH BONTANG
# ============================================================================
LATITUDE = 0.1333          # Lintang Utara
LONGITUDE = 117.5000       # Bujur Timur
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# Parameter NASA POWER (4 variabel)
PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",   # Radiasi matahari (MJ/m²/day)
    "T2M",                 # Suhu udara 2 meter (°C)
    "QV2M",                # Kelembaban spesifik 2 meter (g/kg)
    "WS10M"                # Kecepatan angin 10 meter (m/s)
]

# Konfigurasi tambahan
WIND_ELEVATION = 10        # Ketinggian angin 10 meter
WIND_SURFACE = "airportgrass"  # Airport: flat rough grass
COMMUNITY = "RE"           # Renewable Energy community

# ============================================================================
# FUNGSI 1: MENGUNDUH DATA DARI NASA POWER API
# ============================================================================
def download_nasa_power_bontang():
    """
    Mengunduh data NASA POWER untuk wilayah Bontang
    Berdasarkan spesifikasi penelitian
    """
    
    # Format tanggal ke YYYYMMDD
    start = START_DATE.replace("-", "")
    end = END_DATE.replace("-", "")
    
    # URL endpoint API
    base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    
    # Parameter request
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": COMMUNITY,
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "LST",
        "user": WIND_SURFACE,
        "wind_elevation": WIND_ELEVATION
    }
    
    print("="*70)
    print("🌞 NASA POWER DATA DOWNLOADER - BONTANG")
    print("="*70)
    print(f"\n📌 Konfigurasi Penelitian:")
    print(f"   Lokasi        : Bontang, Kalimantan Timur")
    print(f"   Koordinat     : {LATITUDE}° LU, {LONGITUDE}° BT")
    print(f"   Periode       : {START_DATE} s/d {END_DATE}")
    print(f"   Temporal      : Monthly")
    print(f"   Community     : {COMMUNITY} (Renewable Energy)")
    print(f"   Wind Elevation: {WIND_ELEVATION} m")
    print(f"   Wind Surface  : {WIND_SURFACE}")
    
    print("\n📋 Parameter yang diunduh:")
    print(f"   • ALLSKY_SFC_SW_DWN  → Radiasi matahari (MJ/m²/day)")
    print(f"   • T2M                → Suhu udara 2 meter (°C)")
    print(f"   • QV2M               → Kelembaban spesifik 2 meter (g/kg)")
    print(f"   • WS10M              → Kecepatan angin 10 meter (m/s)")
    
    print("\n📡 Mengirim request ke NASA POWER API...")
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        print("✅ Data berhasil diunduh!")
        return data
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================================
# FUNGSI 2: EKSTRAKSI DATA KE DATAFRAME
# ============================================================================
def extract_to_dataframe(api_response):
    """
    Mengekstrak data dari response API ke Pandas DataFrame
    """
    if api_response is None:
        return None
    
    properties = api_response.get("properties", {})
    parameter_data = properties.get("parameter", {})
    
    # Ekstrak data per parameter
    all_data = {}
    dates = None
    
    for param in PARAMETERS:
        if param in parameter_data:
            param_values = parameter_data[param]
            dates = list(param_values.keys())
            values = list(param_values.values())
            all_data[param] = values
    
    if dates is None:
        print("❌ Tidak ada data ditemukan")
        return None
    
    # Konversi ke DataFrame
    df = pd.DataFrame(all_data)
    
    # Konversi tanggal
    df["date"] = pd.to_datetime(dates, format="%Y%m%d")
    
    # Konversi radiasi: MJ/m²/day → kWh/m²/day (÷ 3.6)
    if "ALLSKY_SFC_SW_DWN" in df.columns:
        df["radiation_kWh"] = df["ALLSKY_SFC_SW_DWN"] / 3.6
        df = df.drop(columns=["ALLSKY_SFC_SW_DWN"])
        print("   ✓ Radiasi dikonversi: MJ/m² → kWh/m² (faktor 3.6)")
    
    # Rename kolom untuk memudahkan
    column_mapping = {
        "T2M": "temperature_C",
        "QV2M": "humidity_gkg",
        "WS10M": "wind_speed_ms"
    }
    df = df.rename(columns=column_mapping)
    
    # Urutkan berdasarkan tanggal
    df = df.sort_values("date").reset_index(drop=True)
    
    return df

# ============================================================================
# FUNGSI 3: GENERATE DATA DAYA (SIMULASI)
# ============================================================================
def generate_power_data(df):
    """
    Membangkitkan data daya listrik (simulasi) berdasarkan model teoritis
    
    Model: Daya = a·Radiasi + b·Suhu + c·Kelembaban + d·KecepatanAngin + ε
    
    Koefisien berdasarkan studi literatur:
    - a (radiasi) = 18-22 (Watt per kWh/m²)
    - b (suhu)    = 4-6   (Watt per °C)
    - c (humid)   = 0.5-1 (Watt per g/kg)
    - d (angin)   = 1-2   (Watt per m/s) - efek pendinginan positif
    
    Catatan: Ini adalah data simulasi. Untuk penelitian sesungguhnya,
    gunakan data daya aktual dari pengukuran PLTS di Bontang.
    """
    
    np.random.seed(42)
    
    # Koefisien model
    a = 20.0      # koefisien radiasi
    b = 5.0       # koefisien suhu
    c = 0.8       # koefisien kelembaban
    d = 1.5       # koefisien kecepatan angin (efek pendinginan)
    
    # Hitung daya berdasarkan model
    power = (a * df['radiation_kWh'] + 
             b * df['temperature_C'] + 
             c * df['humidity_gkg'] + 
             d * df['wind_speed_ms'])
    
    # Tambahkan noise untuk realistis
    noise = np.random.normal(0, 5, len(df))
    power = power + noise
    
    # Bulatkan
    df['power_Watt'] = np.round(power, 1)
    
    print(f"\n   📊 Model simulasi daya: y = {a}·x₁ + {b}·x₂ + {c}·x₃ + {d}·x₄")
    print(f"   ⚠️  Data daya ini adalah SIMULASI untuk keperluan pemodelan")
    print(f"   💡 Untuk penelitian aktual, gunakan data pengukuran PLTS")
    
    return df

# ============================================================================
# FUNGSI 4: REGRESI LINEAR DENGAN NORMAL EQUATION
# ============================================================================
def linear_regression_normal_equation(X, y):
    """
    Regresi linear menggunakan Normal Equation: θ = (Xᵀ X)⁻¹ Xᵀ y
    Pendekatan Aljabar Linear murni
    """
    # Tambahkan kolom bias (intercept)
    X_with_intercept = np.c_[np.ones(X.shape[0]), X]
    
    # Normal Equation
    X_T = X_with_intercept.T
    theta = np.linalg.inv(X_T @ X_with_intercept) @ X_T @ y
    
    return theta, X_with_intercept

# ============================================================================
# FUNGSI 5: EVALUASI MODEL
# ============================================================================
def evaluate_model(y_actual, y_pred):
    """
    Menghitung metrik evaluasi: MAE, MAPE, RMSE, R²
    """
    n = len(y_actual)
    residuals = y_actual - y_pred
    
    # MAE
    mae = np.mean(np.abs(residuals))
    
    # MAPE (hindari division by zero)
    y_nonzero = y_actual[y_actual != 0]
    residuals_nonzero = residuals[y_actual != 0]
    mape = 100 * np.mean(np.abs(residuals_nonzero / y_nonzero))
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mae, mape, rmse, r2

# ============================================================================
# FUNGSI 6: VISUALISASI HASIL
# ============================================================================
def create_visualizations(df, y_actual, y_pred, mae, mape, rmse, r2):
    """
    Membuat grafik visualisasi hasil analisis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analisis Model Prediksi Daya Energi Surya\nBontang, Kalimantan Timur (Data NASA POWER 2020-2024)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Aktual vs Prediksi
    ax1 = axes[0, 0]
    ax1.scatter(y_actual, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Garis Ideal (y=x)')
    ax1.set_xlabel('Daya Aktual (Watt)')
    ax1.set_ylabel('Daya Prediksi (Watt)')
    ax1.set_title(f'Aktual vs Prediksi (R² = {r2:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Perbandingan Aktual vs Prediksi (Line Chart)
    ax2 = axes[0, 1]
    indices = range(len(y_actual))
    ax2.plot(indices, y_actual, 'o-', label='Aktual', markersize=4, linewidth=1.5)
    ax2.plot(indices, y_pred, 's-', label='Prediksi', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Urutan Sampel (Bulan ke-)')
    ax2.set_ylabel('Daya (Watt)')
    ax2.set_title('Perbandingan Daya Aktual vs Prediksi')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual Error
    ax3 = axes[1, 0]
    residuals = y_actual - y_pred
    ax3.bar(indices, residuals, alpha=0.7, color='skyblue')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Urutan Sampel (Bulan ke-)')
    ax3.set_ylabel('Residual Error (Watt)')
    ax3.set_title(f'Residual Error (MAE = {mae:.2f} Watt)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap Korelasi
    ax4 = axes[1, 1]
    df_corr = df[['radiation_kWh', 'temperature_C', 'humidity_gkg', 'wind_speed_ms', 'power_Watt']].copy()
    df_corr.columns = ['Radiasi (kWh/m²)', 'Suhu (°C)', 'Kelembaban (g/kg)', 'Kecepatan Angin (m/s)', 'Daya (Watt)']
    corr_matrix = df_corr.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.3f', square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Heatmap Korelasi Antar Variabel')
    
    plt.tight_layout()
    plt.savefig('hasil_analisis_bontang.png', dpi=300, bbox_inches='tight')
    print("\n💾 Grafik disimpan ke: hasil_analisis_bontang.png")
    plt.close(fig)

# ============================================================================
# EKSEKUSI UTAMA
# ============================================================================
def main():
    """
    Fungsi utama untuk menjalankan seluruh proses
    """
    print("\n" + "="*70)
    print("🚀 MEMULAI PROSES ANALISIS")
    print("="*70)
    
    # Step 1: Unduh data dari NASA POWER
    print("\n[STEP 1] Mengunduh data NASA POWER...")
    api_response = download_nasa_power_bontang()
    df = extract_to_dataframe(api_response)
    
    if df is None or len(df) == 0:
        print("\n⚠️ Gagal mengunduh data dari API. Menggunakan data sampel...")
        df = generate_sample_data_bontang()
    
    # Step 2: Generate data daya (jika tidak ada data aktual)
    print("\n[STEP 2] Menyiapkan data daya listrik...")
    if 'power_Watt' not in df.columns:
        df = generate_power_data(df)
    
    print(f"\n   Jumlah sampel: {len(df)} bulan")
    print(f"   Periode: {df['date'].min()} s/d {df['date'].max()}")
    
    # Step 3: Siapkan matriks X dan y
    print("\n[STEP 3] Membentuk matriks untuk regresi linear...")
    feature_columns = ['radiation_kWh', 'temperature_C', 'humidity_gkg', 'wind_speed_ms']
    X = df[feature_columns].values
    y = df['power_Watt'].values
    
    print(f"   Matriks X: {X.shape[0]} baris × {X.shape[1]} kolom")
    print(f"   Vektor y: {y.shape[0]} sampel")
    
    # Step 4: Regresi linear dengan Normal Equation
    print("\n[STEP 4] Menyelesaikan sistem persamaan linear (Normal Equation)...")
    theta, X_with_intercept = linear_regression_normal_equation(X, y)
    
    # Ekstrak koefisien
    intercept = theta[0]
    coefficients = theta[1:]
    
    print(f"\n   Model yang diperoleh:")
    print(f"   y = {coefficients[0]:.4f}·x₁ + {coefficients[1]:.4f}·x₂ + {coefficients[2]:.4f}·x₃ + {coefficients[3]:.4f}·x₄ + {intercept:.4f}")
    
    print(f"\n   Interpretasi koefisien:")
    print(f"   • Radiasi (x₁): setiap kenaikan 1 kWh/m² → daya naik {coefficients[0]:.2f} Watt")
    print(f"   • Suhu (x₂): setiap kenaikan 1°C → daya naik {coefficients[1]:.2f} Watt")
    print(f"   • Kelembaban (x₃): setiap kenaikan 1 g/kg → daya berubah {coefficients[2]:.2f} Watt")
    print(f"   • Kecepatan Angin (x₄): setiap kenaikan 1 m/s → daya berubah {coefficients[3]:.2f} Watt")
    
    # Step 5: Prediksi
    print("\n[STEP 5] Menghitung prediksi...")
    y_pred = X_with_intercept @ theta
    
    # Step 6: Evaluasi model
    print("\n[STEP 6] Evaluasi model...")
    mae, mape, rmse, r2 = evaluate_model(y, y_pred)
    
    print(f"\n   HASIL EVALUASI MODEL:")
    print("   " + "-"*50)
    print(f"   • MAE  (Mean Absolute Error)     : {mae:.4f} Watt")
    print(f"   • MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"   • RMSE (Root Mean Square Error)  : {rmse:.4f} Watt")
    print(f"   • R²   (Koefisien Determinasi)   : {r2:.4f}")
    print("   " + "-"*50)
    
    # Interpretasi R²
    if r2 >= 0.9:
        print(f"\n   ✅ R² = {r2:.4f} → Model sangat baik (menjelaskan {(r2*100):.1f}% variasi data)")
    elif r2 >= 0.7:
        print(f"\n   ✅ R² = {r2:.4f} → Model baik (menjelaskan {(r2*100):.1f}% variasi data)")
    elif r2 >= 0.5:
        print(f"\n   ⚠️ R² = {r2:.4f} → Model cukup (menjelaskan {(r2*100):.1f}% variasi data)")
    else:
        print(f"\n   ❌ R² = {r2:.4f} → Model kurang baik (perlu perbaikan)")
    
    # Interpretasi MAPE
    if mape < 10:
        print(f"   ✅ MAPE = {mape:.2f}% → Tingkat kesalahan sangat rendah (akurasi tinggi)")
    elif mape < 20:
        print(f"   ✅ MAPE = {mape:.2f}% → Tingkat kesalahan rendah (akurasi baik)")
    elif mape < 30:
        print(f"   ⚠️ MAPE = {mape:.2f}% → Tingkat kesalahan sedang")
    else:
        print(f"   ❌ MAPE = {mape:.2f}% → Tingkat kesalahan tinggi (perlu perbaikan)")
    
    # Step 7: Simpan data ke CSV
    print("\n[STEP 7] Menyimpan data...")
    df['predicted_power_Watt'] = y_pred
    df['error_Watt'] = np.abs(y - y_pred)
    df['error_percent'] = (np.abs(y - y_pred) / y) * 100
    
    df.to_csv("data_nasa_bontang_lengkap.csv", index=False)
    print("   💾 Data lengkap disimpan ke: data_nasa_bontang_lengkap.csv")
    
    # Step 8: Tampilkan tabel hasil
    print("\n[STEP 8] Menampilkan hasil...")
    print("\n📋 Tabel Perbandingan Aktual vs Prediksi (15 data pertama):")
    display_df = df[['date', 'radiation_kWh', 'temperature_C', 'humidity_gkg', 
                    'wind_speed_ms', 'power_Watt', 'predicted_power_Watt', 'error_Watt']].head(15)
    print(display_df.round(2).to_string(index=False))
    
    # Step 9: Visualisasi
    print("\n[STEP 9] Membuat visualisasi...")
    create_visualizations(df, y, y_pred, mae, mape, rmse, r2)
    
    # Step 10: Ringkasan akhir
    print("\n" + "="*70)
    print("✅ ANALISIS SELESAI")
    print("="*70)
    print("\n📊 RINGKASAN EKSEKUTIF:")
    print(f"   • Model: y = {coefficients[0]:.2f}·Radiasi + {coefficients[1]:.2f}·Suhu + {coefficients[2]:.2f}·Kelembaban + {coefficients[3]:.2f}·Angin + {intercept:.2f}")
    print(f"   • Akurasi: MAPE = {mape:.2f}%, R² = {r2:.4f}")
    print(f"   • Total sampel: {len(df)} bulan data (2020-2024)")
    print(f"   • File output: data_nasa_bontang_lengkap.csv, hasil_analisis_bontang.png")
    
    return df, coefficients, mae, mape, rmse, r2

# ============================================================================
# FUNGSI ALTERNATIF: DATA SAMPEL (JIKA API OFFLINE)
# ============================================================================
def generate_sample_data_bontang():
    """
    Membangkitkan data sampel untuk wilayah Bontang
    (Gunakan jika API NASA POWER sedang offline)
    """
    print("\n   Membangkitkan data sampel untuk Bontang...")
    
    np.random.seed(42)
    n_samples = 60  # 5 tahun × 12 bulan
    
    # Radiasi: 4.0 - 6.0 kWh/m²/day (wilayah tropis)
    radiation = np.random.uniform(4.0, 6.0, n_samples)
    
    # Suhu: 26 - 31 °C (Bontang)
    temperature = np.random.uniform(26, 31, n_samples)
    
    # Kelembaban: 15 - 24 g/kg (wilayah lembab)
    humidity = np.random.uniform(15, 24, n_samples)
    
    # Kecepatan angin: 1.5 - 3.5 m/s
    wind_speed = np.random.uniform(1.5, 3.5, n_samples)
    
    # Model simulasi daya
    a, b, c, d = 20, 5, 0.8, 1.5
    power = (a * radiation + b * temperature + c * humidity + d * wind_speed)
    noise = np.random.normal(0, 5, n_samples)
    power = power + noise
    
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=n_samples, freq='MS'),
        'radiation_kWh': np.round(radiation, 2),
        'temperature_C': np.round(temperature, 1),
        'humidity_gkg': np.round(humidity, 1),
        'wind_speed_ms': np.round(wind_speed, 1),
        'power_Watt': np.round(power, 1)
    })
    
    print(f"   ✅ {n_samples} sampel data siap digunakan")
    return df

# ============================================================================
# JALANKAN PROGRAM
# ============================================================================
if __name__ == "__main__":
    df_result, coeff, mae, mape, rmse, r2 = main()