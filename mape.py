import matplotlib.pyplot as plt
import numpy as np

# Data MAPE per tahun (hitung dari error_percent di Tabel 3)
years = ['2020', '2021', '2022', '2023', '2024']
mape_per_year = [0.1000, 0.1073, 0.1269, 0.1552, 0.1278]  # dalam %

plt.figure(figsize=(8, 5))
bars = plt.bar(years, mape_per_year, color='steelblue', edgecolor='black')
plt.axhline(y=0.04204, color='red', linestyle='--', linewidth=2, label=f'MAPE Total = 0.0420%')
plt.xlabel('Tahun')
plt.ylabel('MAPE (%)')
plt.title('Rekapitulasi MAPE per Tahun (2020-2024)')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Tambahkan nilai di atas bar
for bar, val in zip(bars, mape_per_year):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.4f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('mape_per_tahun.png', dpi=300, bbox_inches='tight')
plt.show()