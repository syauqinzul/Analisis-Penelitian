"""
PENGUNDUHAN DATA NASA POWER UNTUK WILAYAH SAMARINDA
===================================================
Dokumentasi API: https://power.larc.nasa.gov/docs/services/api/
Parameter dictionary: https://power.larc.nasa.gov/parameters/
"""

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# ============================================================
# KONFIGURASI WILAYAH SAMARINDA
# ============================================================
LATITUDE = -0.5022   # Lintang Selatan (-)
LONGITUDE = 117.1536 # Bujur Timur (+)
START_DATE = "2024-01-01"  # Mulai dari Januari 2024
END_DATE = "2024-02-29"    # Hingga Februari 2024 (2 bulan ~ 59-60 hari)

# Parameter yang akan diunduh
PARAMETERS = [
    "ALLSKY_SFC_SW_DWN",  # Radiasi matahari (MJ/m²/day)
    "T2M",                # Suhu 2 meter (°C)
    "QV2M",               # Kelembaban spesifik (g/kg)
    "PS"                  # Tekanan permukaan (kPa)
]

# ============================================================
# FUNGSI UNDUH DATA DARI NASA POWER API
# ============================================================
def download_nasa_power_data(latitude, longitude, start_date, end_date, parameters):
    """
    Mengunduh data dari NASA POWER API
    
    Referensi: NASA POWER API Documentation [citation:5]
    
    Args:
        latitude (float): Lintang lokasi
        longitude (float): Bujur lokasi
        start_date (str): Tanggal mulai (YYYY-MM-DD)
        end_date (str): Tanggal akhir (YYYY-MM-DD)
        parameters (list): Daftar parameter yang diinginkan
    
    Returns:
        pd.DataFrame: Dataframe berisi data yang diunduh
    """
    
    # Konversi tanggal ke format YYYYMMDD (format API NASA POWER)
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    
    # Buat URL request sesuai format API [citation:5]
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        "parameters": ",".join(parameters),
        "community": "RE",  # Renewable Energy community
        "longitude": longitude,
        "latitude": latitude,
        "start": start,
        "end": end,
        "format": "JSON"
    }
    
    print(f"📡 Mengirim request ke NASA POWER API...")
    print(f"   Lokasi: {latitude}, {longitude}")
    print(f"   Periode: {start_date} s/d {end_date}")
    print(f"   Parameter: {', '.join(parameters)}")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()  # Raise error jika request gagal
        
        data = response.json()
        print(f"✅ Data berhasil diunduh!")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error mengunduh data: {e}")
        return None

# ============================================================
# FUNGSI EKSTRAK DATA KE DATAFRAME
# ============================================================
def extract_to_dataframe(api_response, parameters):
    """
    Mengekstrak data dari response API ke Pandas DataFrame
    
    Args:
        api_response (dict): Response JSON dari NASA POWER API
        parameters (list): Daftar parameter yang diunduh
    
    Returns:
        pd.DataFrame: Dataframe dengan kolom tanggal dan parameter
    """
    if api_response is None:
        return None
    
    # Ekstrak data dari struktur JSON
    # Struktur response: properties -> parameter -> [parameter] -> values
    properties = api_response.get("properties", {})
    parameter_data = properties.get("parameter", {})
    
    # Buat dictionary untuk menyimpan data per parameter
    all_data = {}
    dates = None
    
    for param in parameters:
        if param in parameter_data:
            param_values = parameter_data[param]
            # Ambil tanggal dan nilai
            dates = list(param_values.keys())
            values = list(param_values.values())
            all_data[param] = values
    
    if dates is None:
        print("❌ Tidak ada data yang ditemukan")
        return None
    
    # Konversi ke DataFrame
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(dates)
    
    # Konversi radiasi dari MJ/m²/day ke kWh/m²/day (bagi dengan 3.6)
    if "ALLSKY_SFC_SW_DWN" in df.columns:
        df["radiation_kWh"] = df["ALLSKY_SFC_SW_DWN"] / 3.6
        print(f"   Radiasi dikonversi: MJ/m² → kWh/m² (faktor 3.6)")
    
    # Urutkan berdasarkan tanggal
    df = df.sort_values("date").reset_index(drop=True)
    
    return df

# ============================================================
# EKSEKUSI UNDUH DATA
# ============================================================
print("\n" + "="*60)
print("🌞 NASA POWER DATA DOWNLOADER - SAMARINDA")
print("="*60 + "\n")

# Unduh data
api_response = download_nasa_power_data(
    latitude=LATITUDE,
    longitude=LONGITUDE,
    start_date=START_DATE,
    end_date=END_DATE,
    parameters=PARAMETERS
)

# Ekstrak ke DataFrame
df_raw = extract_to_dataframe(api_response, PARAMETERS)

if df_raw is not None:
    print(f"\n📊 Dataframe berhasil dibuat:")
    print(f"   Jumlah sampel: {len(df_raw)} hari")
    print(f"   Kolom: {list(df_raw.columns)}")
    
    # Tampilkan 5 data pertama
    print(f"\n📋 5 data teratas:")
    print(df_raw.head())
    
    # Simpan ke CSV
    df_raw.to_csv("data_nasa_samarinda.csv", index=False)
    print(f"\n💾 Data disimpan ke: data_nasa_samarinda.csv")
else:
    print("❌ Gagal mengunduh data")

# ============================================================
# SIMULASI DATA ALTERNATIF (JIKA API OFFLINE)
# ============================================================
def generate_sample_data(n_samples=50):
    """
    Membangkitkan data sampel untuk pengujian
    (Gunakan jika API sedang offline atau untuk testing)
    
    Data ini disimulasikan berdasarkan pola data dari penelitian Pabate (2025)
    dan karakteristik wilayah tropis Samarinda.
    
    Args:
        n_samples (int): Jumlah sampel yang diinginkan
    
    Returns:
        pd.DataFrame: Data sampel
    """
    np.random.seed(42)  # Untuk reproducibility
    
    # Radiasi: 4.0 - 6.5 kWh/m²/day (data NASA untuk wilayah tropis)
    radiation = np.random.uniform(4.0, 6.5, n_samples)
    
    # Suhu: 26 - 32 °C (wilayah tropis Samarinda)
    temperature = np.random.uniform(26, 32, n_samples)
    
    # Kelembaban spesifik: 15 - 25 g/kg (wilayah tropis lembab)
    humidity = np.random.uniform(15, 25, n_samples)
    
    # Tekanan: 100.5 - 101.2 kPa
    pressure = np.random.uniform(100.5, 101.2, n_samples)
    
    # Daya: simulasi berdasarkan model y = a*x1 + b*x2 + c*x3 + d*x4 + noise
    # Koefisien dari studi literatur [citation:3]
    a, b, c, d = 18, 5, 0.5, -0.2  # radiasi dominan, tekanan pengaruh negatif
    noise = np.random.normal(0, 5, n_samples)  # noise kecil
    
    power = (a * radiation + 
             b * temperature + 
             c * humidity + 
             d * pressure + 
             noise)
    
    df = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=n_samples, freq="D"),
        "radiation_kWh": np.round(radiation, 2),
        "temperature_C": np.round(temperature, 1),
        "humidity_gkg": np.round(humidity, 1),
        "pressure_kPa": np.round(pressure, 2),
        "power_Watt": np.round(power, 1)
    })
    
    return df

# Jika API gagal, gunakan data sampel
if df_raw is None or len(df_raw) == 0:
    print("\n⚠️ Menggunakan data sampel untuk pengujian...")
    df_sample = generate_sample_data(50)
    print(f"📊 Data sampel: {len(df_sample)} sampel")
    print(df_sample.head())
    df_raw = df_sample