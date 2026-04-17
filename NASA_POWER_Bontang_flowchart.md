# Flowchart Analisis NASA POWER Bontang

```mermaid
flowchart TD
    A[Mulai] --> B[Baca data dari nasa_power_bontang.csv]
    B --> C[Ekstrak parameter:
ALLSKY_SFC_SW_DWN, T2M, WS10M,
QV2M, PS, WSC]
    C --> D[Hitung estimasi daya Y]
    D --> E[Pembangunan model OLS
(XtX, XtY, solve koefisien)]
    E --> F[Prediksi Y_pred]
    F --> G[Evaluasi: MAE, MAPE, R2, RMSE]
    G --> H[Statistik deskriptif 6 parameter]
    H --> I[Rata-rata tahunan per parameter]
    G --> J[Hitung residual dan distribusi]
    I --> K[Plot 3 panel:
- Rata-rata tahunan (kiri)
- Box plot bulanan (tengah)
- Distribusi residual (kanan)]
    J --> K
    K --> L[Simpan grafik dan CSV]
    L --> M[Selesai]
```
