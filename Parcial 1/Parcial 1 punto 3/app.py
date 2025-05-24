import streamlit as st
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, make_scorer
import matplotlib.pyplot as plt

st.title("Comparación de Modelos: Kernel Ridge, Gaussian Process y Lasso")

@st.cache_data
def cargar_datos():
    # Cambia aquí la ruta a tu CSV con el dataset preprocesado
    return pd.read_csv("X_final_cleaned.csv")

df = cargar_datos()
st.subheader("Dataset preprocesado")
st.dataframe(df, use_container_width=True)

target = "SalePrice"
X = df.drop(columns=[target])
y = df[target]

models = {
    "Kernel Ridge": KernelRidge(alpha=0.1, degree=3, gamma=0.0031338741453090678, kernel='poly'),
    "Gaussian Process": GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        alpha=4.452048365748848e-07,
        random_state=42
    ),
    "Lasso": Lasso(alpha=11.594392087725058, max_iter=10000)
}

scoring = {
    'MAE': make_scorer(mean_absolute_error),
    'MSE': make_scorer(mean_squared_error),
    'R2': make_scorer(r2_score),
    'MAPE': make_scorer(mean_absolute_percentage_error)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = {}
    for metric in scoring.keys():
        mean_score = np.mean(scores[f'test_{metric}'])
        std_score = np.std(scores[f'test_{metric}'])
        results[name][f'{metric} mean'] = mean_score
        results[name][f'{metric} std'] = std_score

df_results = pd.DataFrame(results).T
st.subheader("Resultados de Validación Cruzada (5 folds)")
st.dataframe(df_results.style.format("{:.4f}"), use_container_width=True)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
df_results["MAE mean"].plot(kind='bar', ax=axs[0, 0], title="MAE (menor mejor)")
df_results["MSE mean"].plot(kind='bar', ax=axs[0, 1], title="MSE (menor mejor)")
df_results["R2 mean"].plot(kind='bar', ax=axs[1, 0], title="R2 (mayor mejor)")
df_results["MAPE mean"].plot(kind='bar', ax=axs[1, 1], title="MAPE (menor mejor)")
for ax in axs.flatten():
    ax.grid(True)

st.pyplot(fig)
