import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Cartella contenente i JSON.
# "." = stessa cartella dello script
INPUT_DIR = Path("./liralab/training_analysis")

# Cartella dove salvare i risultati
OUTPUT_DIR = Path("./liralab/training_analysis/analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PARAMETERS = [
    "num_epochs",
    "lr_backbone",
    "batch_size",
    "chunk_size",
    "dim_feedforward",
    "lr",
    "kl_weight",
    "num_encoder_layers",
    "nhead",
    "position_embedding",
    "latent_dim",
]

TARGET = "score"


# ============================================================
# 1. CARICAMENTO DEI JSON
# ============================================================

rows = []

json_files = sorted(INPUT_DIR.glob("*.json"))

if not json_files:
    raise RuntimeError(f"Non ho trovato nessun file .json nella cartella: {INPUT_DIR.glob("*.json")}")

print(f"Trovati {len(json_files)} file JSON.")

for file in json_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Controllo che ci sia lo score
        if TARGET not in data:
            print(f"[WARNING] {file.name}: manca '{TARGET}', salto.")
            continue

        row = {
            "file": file.name,
            TARGET: data[TARGET],
        }

        for param in PARAMETERS:
            row[param] = data.get(param, np.nan)

        rows.append(row)

    except Exception as e:
        print(f"[ERROR] Problema con {file.name}: {e}")


df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("Nessun training valido trovato.")

print("\nDataset caricato:")
print(df[["file"] + PARAMETERS + [TARGET]].to_string(index=False))


# ============================================================
# 2. PULIZIA / CONVERSIONE
# ============================================================

# position_embedding:
# learned -> 0
# sine    -> 1
#
# Se in futuro hai altri valori, verranno trattati come NaN.
embedding_mapping = {
    "learned": 0,
    "sine": 1,
}

df["position_embedding_numeric"] = (
    df["position_embedding"]
    .map(embedding_mapping)
)

# Lista delle feature numeriche da utilizzare
numeric_parameters = [
    "num_epochs",
    "lr_backbone",
    "batch_size",
    "chunk_size",
    "dim_feedforward",
    "lr",
    "kl_weight",
    "num_encoder_layers",
    "nhead",
    "position_embedding_numeric",
    "latent_dim",
]

# Conversione numerica
for col in numeric_parameters + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Eliminiamo righe con dati mancanti
analysis_df = df[numeric_parameters + [TARGET]].dropna().copy()

if len(analysis_df) < 3:
    raise RuntimeError(
        "Servono almeno 3 training validi per fare un'analisi delle correlazioni."
    )

print(f"\nTraining utilizzabili per l'analisi: {len(analysis_df)}")


# ============================================================
# 3. CORRELAZIONE SPEARMAN
# ============================================================

# Spearman è generalmente più adatta di Pearson per questo tipo
# di esperimenti, perché permette relazioni monotone non lineari.

spearman_corr = analysis_df.corr(method="spearman")

score_correlations = (
    spearman_corr[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda x: abs(x), ascending=False)
)

print("\n" + "=" * 70)
print("CORRELAZIONE SPEARMAN CON LO SCORE")
print("=" * 70)

for parameter, correlation in score_correlations.items():

    # Nome leggibile
    display_name = parameter.replace(
        "position_embedding_numeric",
        "position_embedding"
    )

    print(f"{display_name:25s}: {correlation:+.3f}")


# ============================================================
# 4. HEATMAP
# ============================================================

plt.figure(figsize=(12, 10))

sns.heatmap(
    spearman_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
)

plt.title("Correlazioni Spearman tra parametri e score")
plt.tight_layout()

heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=200)
plt.close()

print(f"\nSalvata: {heatmap_path}")


# ============================================================
# 5. BARPLOT DELLE CORRELAZIONI CON LO SCORE
# ============================================================

plot_df = score_correlations.reset_index()
plot_df.columns = ["parameter", "correlation"]

# Nome leggibile
plot_df["parameter"] = plot_df["parameter"].replace({
    "position_embedding_numeric": "position_embedding"
})

plt.figure(figsize=(10, 7))

colors = [
    "#d62728" if x < 0 else "#2ca02c"
    for x in plot_df["correlation"]
]

sns.barplot(
    data=plot_df,
    x="correlation",
    y="parameter",
    palette=colors,
    hue="parameter",
    legend=False,
)

plt.axvline(0, color="black", linewidth=1)
plt.xlim(-1, 1)
plt.xlabel("Correlazione Spearman con score")
plt.ylabel("")
plt.title("Quanto ogni parametro è correlato allo score")
plt.tight_layout()

correlation_bar_path = OUTPUT_DIR / "score_correlations.png"
plt.savefig(correlation_bar_path, dpi=200)
plt.close()

print(f"Salvata: {correlation_bar_path}")


# ============================================================
# 6. RANDOM FOREST + PERMUTATION IMPORTANCE
# ============================================================

X = analysis_df[numeric_parameters]
y = analysis_df[TARGET]

# Random Forest
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
)

rf.fit(X, y)

# Permutation importance:
# misura quanto peggiora il modello quando una feature viene
# casualmente permutata.
#
# Usiamo più ripetizioni per avere una stima più stabile.
perm = permutation_importance(
    rf,
    X,
    y,
    n_repeats=30,
    random_state=42,
    scoring="neg_mean_squared_error",
)

importance_df = pd.DataFrame({
    "parameter": numeric_parameters,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False,
)

importance_df["parameter"] = importance_df["parameter"].replace({
    "position_embedding_numeric": "position_embedding"
})

print("\n" + "=" * 70)
print("RANDOM FOREST - PERMUTATION IMPORTANCE")
print("=" * 70)

for _, row in importance_df.iterrows():
    print(
        f"{row['parameter']:25s}: "
        f"{row['importance_mean']:.5f} "
        f"+/- {row['importance_std']:.5f}"
    )


# ============================================================
# 7. GRAFICO RANDOM FOREST
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["parameter"],
    importance_df["importance_mean"],
    xerr=importance_df["importance_std"],
    color="#4c72b0",
    alpha=0.85,
)

plt.gca().invert_yaxis()

plt.xlabel("Permutation importance")
plt.ylabel("")
plt.title("Importanza dei parametri secondo Random Forest")

plt.tight_layout()

rf_path = OUTPUT_DIR / "random_forest_importance.png"
plt.savefig(rf_path, dpi=200)
plt.close()

print(f"\nSalvata: {rf_path}")


# ============================================================
# 8. SCORE VS OGNI PARAMETRO
# ============================================================

# Creiamo un grafico separato per ogni parametro.

n_params = len(PARAMETERS)

fig, axes = plt.subplots(
    4,
    3,
    figsize=(16, 16),
)

axes = axes.flatten()

for i, parameter in enumerate(PARAMETERS):

    ax = axes[i]

    if parameter == "position_embedding":

        # Variabile categorica
        sns.boxplot(
            data=df,
            x=parameter,
            y=TARGET,
            ax=ax,
            color="#8ecae6",
        )

        sns.stripplot(
            data=df,
            x=parameter,
            y=TARGET,
            ax=ax,
            color="black",
            size=5,
            jitter=True,
        )

    else:

        sns.scatterplot(
            data=df,
            x=parameter,
            y=TARGET,
            ax=ax,
            s=70,
        )

        # Linea di regressione LOWESS, se possibile
        try:
            sns.regplot(
                data=df,
                x=parameter,
                y=TARGET,
                ax=ax,
                scatter=False,
                lowess=True,
                color="red",
            )
        except Exception:
            pass

        # Learning rate e simili è spesso meglio visualizzarli
        # in scala logaritmica
        if parameter in ["lr", "lr_backbone"]:

            positive_values = df[parameter].dropna()

            if len(positive_values) > 0 and (positive_values > 0).all():
                ax.set_xscale("log")

    ax.set_title(parameter)
    ax.grid(alpha=0.2)

# Eliminiamo subplot inutilizzato
for j in range(n_params, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(
    "Score in funzione dei parametri di training",
    fontsize=18,
    y=1.01,
)

plt.tight_layout()

scatter_path = OUTPUT_DIR / "score_vs_parameters.png"
plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Salvata: {scatter_path}")


# ============================================================
# 9. SALVATAGGIO DEI RISULTATI IN CSV
# ============================================================

summary = pd.DataFrame({
    "parameter": score_correlations.index,
    "spearman_correlation": score_correlations.values,
})

summary["abs_spearman"] = summary["spearman_correlation"].abs()

summary = summary.merge(
    importance_df[
        ["parameter", "importance_mean", "importance_std"]
    ],
    on="parameter",
    how="left",
)

summary = summary.sort_values(
    "abs_spearman",
    ascending=False,
)

csv_path = OUTPUT_DIR / "parameter_analysis.csv"

summary.to_csv(
    csv_path,
    index=False,
)

print(f"Salvato: {csv_path}")


# ============================================================
# 10. REPORT FINALE
# ============================================================

print("\n" + "=" * 70)
print("PARAMETRI PIÙ CORRELATI ALLO SCORE")
print("=" * 70)

for i, row in summary.iterrows():

    correlation = row["spearman_correlation"]

    if correlation > 0:
        direction = "↑ score"
    else:
        direction = "↓ score"

    print(
        f"{row['parameter']:25s} "
        f"Spearman={correlation:+.3f}  {direction}"
    )

print("\nAnalisi completata.")
print(f"Risultati nella cartella: {OUTPUT_DIR.resolve()}")
