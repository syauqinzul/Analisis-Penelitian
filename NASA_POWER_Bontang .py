import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nasa_file = 'nasa_power_bontang.csv'
fallback_file = 'data_nasa_bontang_lengkap.csv'

if os.path.exists(nasa_file):
    df = pd.read_csv(nasa_file, skiprows=18)
    months_csv = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    years_all = list(range(2015, 2025))  # 10 tahun

    def get_param(param_name, years):
        data = []
        for y in years:
            row = df[(df['PARAMETER'] == param_name) & (df['YEAR'] == y)].iloc[0]
            for m in months_csv:
                data.append(float(row[m]))
        return np.array(data)

    # 6 Parameter NASA POWER
    ALLSKY = get_param('ALLSKY_SFC_SW_DWN', years_all)  # x1
    T2M    = get_param('T2M',   years_all)               # x2
    WS10M  = get_param('WS10M', years_all)               # x3
    QV2M   = get_param('QV2M',  years_all)               # x4
    PS     = get_param('PS',    years_all)               # x5
    WSC    = get_param('WSC',   years_all)               # x6

    # Estimasi daya panel surya
    eta_STC = 0.18
    beta_T = 0.004
    T_STC = 25.0
    Y = ALLSKY * eta_STC * (1 - beta_T * (T2M - T_STC))

    # Aljabar Linear: Normal Equations OLS (6x6)
    X = np.column_stack([ALLSKY, T2M, WS10M, QV2M, PS, WSC])
    XtX = X.T @ X
    XtY = X.T @ Y
    koef = np.linalg.solve(XtX, XtY)
    a, b, c, d, e, f = koef

    # Prediksi dan evaluasi
    Y_pred = X @ koef
    error = Y - Y_pred
    MAE = np.mean(np.abs(error))
    MAPE = np.mean(np.abs(error / Y)) * 100
    SST = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - np.sum(error**2) / SST
    RMSE = np.sqrt(np.mean(error**2))

    print('Using NASA POWER monthly data:', nasa_file)
    print(f'Model coefficients: a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}, e={e:.6f}, f={f:.6f}')
    print(f'MAE={MAE:.8f}, MAPE={MAPE:.6f}%, R2={R2:.8f}, RMSE={RMSE:.8f}')

    # Statistik deskriptif 6 parameter
    param_list = ['ALLSKY_SFC_SW_DWN', 'T2M', 'WS10M', 'QV2M', 'PS', 'WSC']
    df_params = df[df['PARAMETER'].isin(param_list)].copy()
    df_params['YEAR'] = df_params['YEAR'].astype(int)

    summary_rows = []
    for param in param_list:
        values = df_params.loc[df_params['PARAMETER'] == param, months_csv].to_numpy().astype(float).ravel()
        summary_rows.append({
            'Parameter': param,
            'Count': values.size,
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values, ddof=1),
            'Min': np.min(values),
            'Max': np.max(values)
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('NASA_POWER_Bontang_descriptive_statistics.csv', index=False)
    print('\nDescriptive Statistics for 6 parameters:')
    print(summary_df.to_string(index=False, float_format='%.4f'))

    yearly_means = df_params.set_index(['YEAR', 'PARAMETER'])[months_csv].mean(axis=1).unstack('PARAMETER')
    yearly_means.to_csv('NASA_POWER_Bontang_yearly_means.csv')

    # Visualisasi: 3 panel
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    yearly_means.plot(ax=axes[0], marker='o')
    axes[0].set_title('Rata-rata Tahunan 6 Parameter')
    axes[0].set_xlabel('Tahun')
    axes[0].set_ylabel('Rata-rata nilai')
    axes[0].legend(loc='best', fontsize='small')
    axes[0].grid(alpha=0.3)

    allsky_monthly = df_params[df_params['PARAMETER'] == 'ALLSKY_SFC_SW_DWN'][months_csv]
    axes[1].boxplot(allsky_monthly.values, labels=months_csv, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='steelblue'), medianprops=dict(color='red'))
    axes[1].set_title('Box Plot Bulanan: ALLSKY SFC SW DWN')
    axes[1].set_xlabel('Bulan')
    axes[1].set_ylabel('Radiasi (kWh/m^2/day)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3)

    residuals_percent = np.abs(error / Y) * 100
    axes[2].hist(residuals_percent, bins=15, color='seagreen', edgecolor='black', alpha=0.8)
    axes[2].set_title('Distribusi Residual (%)')
    axes[2].set_xlabel('Residual (%)')
    axes[2].set_ylabel('Frekuensi')
    axes[2].grid(alpha=0.3)

    fig.suptitle('Statistik Deskriptif & Distribusi Residual', fontsize=16)
    fig.savefig('NASA_POWER_Bontang_statistik.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.figure(figsize=(10, 6))
    plt.plot(Y, label='Y Actual', linewidth=2)
    plt.plot(Y_pred, label='Y Predicted', linestyle='--', linewidth=2)
    plt.title('NASA POWER: Actual vs Predicted Daya Panel Surya')
    plt.xlabel('Index Bulanan')
    plt.ylabel('Daya (estimasi)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('NASA_POWER_Bontang_output.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.bar(years_all, [np.mean(get_param('ALLSKY_SFC_SW_DWN', [y])) for y in years_all], color='steelblue', edgecolor='black')
    plt.title('Rata-rata ALLSKY SFC SW DWN per Tahun')
    plt.xlabel('Tahun')
    plt.ylabel('Radiasi (kWh/m^2/day)')
    plt.tight_layout()
    plt.savefig('NASA_POWER_Bontang_allsky_per_year.png', dpi=300, bbox_inches='tight')
    plt.show()

    error_percent = np.abs(error / Y) * 100
    plt.figure(figsize=(12, 5))
    plt.plot(error_percent, marker='o', linestyle='-', color='crimson', linewidth=1.5)
    plt.title('Persentase Error per Bulan (NASA POWER Data)')
    plt.xlabel('Index Bulanan')
    plt.ylabel('Error (%)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('NASA_POWER_Bontang_error_percent.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(Y, Y_pred, color='darkgreen', alpha=0.7, edgecolors='black', linewidths=0.5)
    min_val = min(Y.min(), Y_pred.min())
    max_val = max(Y.max(), Y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5)
    plt.title('Actual vs Predicted Scatter Plot')
    plt.xlabel('Y Actual')
    plt.ylabel('Y Predicted')
    plt.tight_layout()
    plt.savefig('NASA_POWER_Bontang_actual_vs_predicted_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    if not os.path.exists(fallback_file):
        raise FileNotFoundError(
            f"Tidak ditemukan file data. Letakkan salah satu file berikut di folder kerja:\n"
            f"- {nasa_file}\n"
            f"- {fallback_file}"
        )

    df = pd.read_csv(fallback_file)
    df['date'] = pd.to_datetime(df['date'])
    Y = df['power_Watt'].to_numpy()
    Y_pred = df['predicted_power_Watt'].to_numpy()
    error = Y - Y_pred

    MAE = np.mean(np.abs(error))
    MAPE = np.mean(np.abs(error / Y)) * 100
    SST = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - np.sum(error**2) / SST
    RMSE = np.sqrt(np.mean(error**2))

    print('Using fallback data:', fallback_file)
    print(f'MAE={MAE:.8f}, MAPE={MAPE:.6f}%, R2={R2:.8f}, RMSE={RMSE:.8f}')

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], Y, label='Actual Power', linewidth=2)
    plt.plot(df['date'], Y_pred, label='Predicted Power', linestyle='--', linewidth=2)
    plt.title('Actual vs Predicted Power (Fallback Data)')
    plt.xlabel('Tanggal')
    plt.ylabel('Daya (Watt)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('NASA_POWER_Bontang_fallback_output.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.bar(df['date'].dt.strftime('%Y-%m'), df['error_percent'], color='steelblue', edgecolor='black')
    plt.title('Error Percent per Bulan (Fallback Data)')
    plt.xlabel('Bulan')
    plt.ylabel('Error (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('NASA_POWER_Bontang_error_percent.png', dpi=300, bbox_inches='tight')
    plt.show()

