import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
# Load data CSV NASA POWER (10 tahun)
df = pd.read_csv('POWER_Point_Monthly_20150101_20241231_000d13N_117d50E_UTC.csv', skiprows=18)
months_csv = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
years_all  = list(range(2015, 2025))
 
def get_param(param_name, years):
    data = []
    for y in years:
        row = df[(df['PARAMETER']==param_name) & (df['YEAR']==y)].iloc[0]
        for m in months_csv: data.append(float(row[m]))
    return np.array(data)
 
# 6 Parameter NASA POWER
ALLSKY = get_param('ALLSKY_SFC_SW_DWN', years_all)  # x1
T2M    = get_param('T2M',   years_all)               # x2
WS10M  = get_param('WS10M', years_all)               # x3
QV2M   = get_param('QV2M',  years_all)               # x4
PS     = get_param('PS',    years_all)               # x5
WSC    = get_param('WSC',   years_all)               # x6
 
# Estimasi daya panel surya
eta_STC=0.18; beta_T=0.004; T_STC=25.0
Y = ALLSKY * eta_STC * (1 - beta_T * (T2M - T_STC))
 
# ── Train/Test Split 80:20 (n=96/24) ─────────────────────────────────────────
X_all = np.column_stack([ALLSKY, T2M, WS10M, QV2M, PS, WSC])
n_train = 96  # 2015-2022
X_train, X_test = X_all[:n_train], X_all[n_train:]
Y_train, Y_test  = Y[:n_train],     Y[n_train:]
 
# ── OLS pada data training ────────────────────────────────────────────────────
XtX   = X_train.T @ X_train
XtY   = X_train.T @ Y_train
koef  = np.linalg.solve(XtX, XtY)
a,b,c,d,e,f = koef
 
# ── Evaluasi pada data test ───────────────────────────────────────────────────
Y_pred_test = X_test @ koef
test_errors = Y_test - Y_pred_test
MAE_test  = np.mean(np.abs(test_errors))
MAPE_test = np.mean(np.abs(test_errors / Y_test)) * 100
R2_test   = 1 - np.sum(test_errors**2)/np.sum((Y_test-np.mean(Y_test))**2)
RMSE_test = np.sqrt(np.mean(test_errors**2))
 
# ── Uji Signifikansi Koefisien (t-test) ──────────────────────────────────────
Y_pred_all = X_all @ koef
residuals  = Y - Y_pred_all
sigma2     = np.sum(residuals**2) / (len(Y) - 6)
XtX_full   = X_all.T @ X_all
XtX_inv    = np.linalg.inv(XtX_full)
SE_coef    = np.sqrt(sigma2 * np.diag(XtX_inv))
t_stats    = koef / SE_coef
p_values   = 2 * stats.t.sf(np.abs(t_stats), df=len(Y)-6)
 
# ── VIF (Variance Inflation Factor) ─────────────────────────────────────────
from numpy.linalg import solve
VIF = []
for i in range(6):
    Xi = X_all[:,i]; Xothers = np.delete(X_all,i,axis=1)
    b_i = solve(Xothers.T@Xothers, Xothers.T@Xi)
    R2_i = 1 - np.sum((Xi - Xothers@b_i)**2)/np.sum((Xi-Xi.mean())**2)
    VIF.append(1/(1-R2_i))
 
# ── Uji Durbin-Watson ────────────────────────────────────────────────────────
dw_num = np.sum(np.diff(residuals)**2)
dw_den = np.sum(residuals**2)
DW = dw_num / dw_den  # Nilai mendekati 2 = tidak ada autokorelasi
 
# ── Uji Normalitas Residual (Shapiro-Wilk) ───────────────────────────────────
W_stat, p_normal = stats.shapiro(residuals)
