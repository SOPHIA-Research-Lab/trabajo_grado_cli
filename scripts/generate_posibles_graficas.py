#!/usr/bin/env python3
"""Genera un conjunto ampliado de gráficas candidatas para el informe final."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, learning_curve
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from anomaly_detector import DistanceBasedAnomalyDetector  # noqa: E402
from hologram_analysis import HologramAnalyzer  # noqa: E402

sns.set_context("talk")
sns.set_style("whitegrid")


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def prepare_config(raw_config: Dict) -> Dict:
    config = raw_config.copy()
    execution = config.setdefault("execution", {})
    execution["mode"] = "full"
    execution["progress_bar"] = False
    execution.setdefault("cache_features", True)

    model_cfg = config.setdefault("model", {})
    model_cfg["auto_optimize"] = False
    model_cfg["top_k_features"] = int(model_cfg.get("top_k_features", 35))

    return config


def get_top_feature_indices(feature_names: List[str], selected_names: Iterable[str], top_k: int) -> List[int]:
    indices: List[int] = []
    for name in selected_names:
        if name in feature_names:
            indices.append(feature_names.index(name))
        if len(indices) == top_k:
            break
    return indices


def compute_feature_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    stats_records = []
    labels = labels.astype(int)
    for idx, name in enumerate(feature_names):
        column = features[:, idx]
        healthy = column[labels == 0]
        scd = column[labels == 1]
        if healthy.size == 0 or scd.size == 0:
            continue
        mean_healthy = healthy.mean()
        mean_scd = scd.mean()
        diff = mean_scd - mean_healthy
        pooled_std = np.sqrt(((healthy.size - 1) * healthy.var(ddof=1) + (scd.size - 1) * scd.var(ddof=1)) / (healthy.size + scd.size - 2))
        cohens_d = diff / pooled_std if pooled_std > 0 else 0.0
        if np.std(column) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(column, labels)[0, 1])
        try:
            auc = roc_auc_score(labels, column)
        except ValueError:
            auc = 0.5
        stats_records.append(
            {
                "feature": name,
                "mean_healthy": mean_healthy,
                "mean_scd": mean_scd,
                "mean_diff": diff,
                "cohens_d": cohens_d,
                "abs_d": abs(cohens_d),
                "point_biserial": correlation,
                "abs_corr": abs(correlation),
                "univariate_auc": auc,
                "feature_idx": idx,
            }
        )
    stats_df = pd.DataFrame(stats_records)
    stats_df.sort_values("abs_d", ascending=False, inplace=True)
    return stats_df


def plot_learning_curve(
    pipeline: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    cv: StratifiedKFold,
    output_path: Path,
) -> None:
    train_sizes = np.linspace(0.2, 1.0, 6)
    train_sizes, train_scores, val_scores = learning_curve(
        clone(pipeline),
        features,
        labels,
        cv=cv,
        train_sizes=train_sizes,
        scoring="accuracy",
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_mean, "o-", color="#1b9e77", label="Entrenamiento")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="#1b9e77")
    plt.plot(train_sizes, val_mean, "o-", color="#d95f02", label="Validación cruzada")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="#d95f02")

    plt.title("Curva de aprendizaje (accuracy)")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("Exactitud")
    plt.ylim(0.7, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cv_metrics(
    pipeline: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    cv: StratifiedKFold,
    output_path: Path,
) -> None:
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
    }
    cv_results = cross_validate(
        clone(pipeline),
        features,
        labels,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    records = []
    for metric, values in cv_results.items():
        if metric.startswith("test_"):
            clean_name = metric.replace("test_", "")
            for fold_idx, value in enumerate(values, start=1):
                records.append({"fold": fold_idx, "metric": clean_name, "score": value})

    df = pd.DataFrame(records)

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="metric", y="score", color="#7cb342")
    sns.stripplot(data=df, x="metric", y="score", color="#1b5e20", size=6, alpha=0.7)
    plt.ylim(0.6, 1.05)
    plt.xlabel("Métrica")
    plt.ylabel("Valor por pliegue")
    plt.title("Distribución de métricas en validación cruzada")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_correlation(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    selected_features: Iterable[str],
    output_path: Path,
    top_k: int = 12,
) -> None:
    indices = get_top_feature_indices(feature_names, selected_features, top_k)
    if len(indices) < 2:
        return
    df = pd.DataFrame(features[:, indices], columns=[feature_names[i] for i in indices])
    df = df.astype(float)

    corr = df.corr()

    plt.figure(figsize=(min(1 + 0.6 * len(indices), 14), 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Mapa de correlaciones entre características clave")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_pairplot(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    selected_features: Iterable[str],
    output_path: Path,
    top_k: int = 4,
) -> None:
    indices = get_top_feature_indices(feature_names, selected_features, top_k)
    if len(indices) < 2:
        return

    selected_names = [feature_names[i] for i in indices]
    df = pd.DataFrame(features[:, indices], columns=selected_names)
    df["Clase"] = np.where(labels == 1, "SCD", "Healthy")

    grid = sns.pairplot(df, hue="Clase", diag_kind="kde", corner=True, plot_kws={"alpha": 0.6, "s": 25})
    grid.fig.suptitle("Relaciones entre características más discriminativas", y=1.02)
    grid.fig.set_size_inches(12, 10)
    grid.savefig(output_path, dpi=300)
    plt.close(grid.fig)


def plot_probability_distribution(
    pipeline: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    cv: StratifiedKFold,
    output_path: Path,
) -> None:
    probabilities = cross_val_predict(
        clone(pipeline),
        features,
        labels,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    df = pd.DataFrame(
        {
            "Probabilidad": probabilities,
            "Clase": np.where(labels == 1, "SCD", "Healthy"),
        }
    )

    plt.figure(figsize=(9, 6))
    sns.kdeplot(data=df, x="Probabilidad", hue="Clase", fill=True, common_norm=False, alpha=0.4)
    plt.title("Distribución de probabilidades predichas (validación cruzada)")
    plt.xlabel("Probabilidad predicha de SCD")
    plt.ylabel("Densidad")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_anomaly_scores(
    features: np.ndarray,
    labels: np.ndarray,
    detector: DistanceBasedAnomalyDetector | None,
    output_path: Path,
) -> None:
    if detector is None:
        return

    distances: List[float] = []
    scores: List[float] = []
    classes: List[str] = []

    for feature_vector, label in zip(features, labels):
        result = detector.predict(feature_vector)
        distances.append(result.get("mahalanobis_distance", np.nan))
        scores.append(result.get("anomaly_score", np.nan))
        classes.append("SCD" if label == 1 else "Healthy")

    df = pd.DataFrame({"Distancia": distances, "Score": scores, "Clase": classes})

    plt.figure(figsize=(9, 6))
    sns.violinplot(data=df, x="Clase", y="Distancia", inner="quartile", palette="Set2")
    plt.axhline(detector.threshold, color="#d32f2f", linestyle="--", label=f"Umbral ({detector.threshold:.2f})")
    plt.yscale("symlog")
    plt.title("Distribución de distancias de Mahalanobis por clase")
    plt.ylabel("Distancia de Mahalanobis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Clase", y="Score", palette="Set2")
    plt.axhline(1.0, color="#d32f2f", linestyle="--", linewidth=1, alpha=0.6)
    plt.title("Distribución de scores de anomalía")
    plt.ylabel("Score normalizado")
    plt.tight_layout()
    plt.savefig(output_path.with_name("anomaly_score_distribution.png"), dpi=300)
    plt.close()


def plot_top_feature_effects(
    feature_stats: pd.DataFrame,
    output_path: Path,
    top_k: int = 15,
) -> None:
    if feature_stats.empty:
        return
    top = feature_stats.head(top_k)
    plt.figure(figsize=(12, 7))
    colors = ["#d95f02" if row["cohens_d"] > 0 else "#1b9e77" for _, row in top.iterrows()]
    sns.barplot(
        data=top,
        x="cohens_d",
        y="feature",
        palette=colors,
        orient="h",
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Cohen's d (positivo ⇒ SCD > Healthy)")
    plt.ylabel("Característica")
    plt.title("Top características por tamaño del efecto")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_effect_vs_correlation(
    feature_stats: pd.DataFrame,
    output_path: Path,
    top_labels: int = 20,
) -> None:
    if feature_stats.empty:
        return
    plt.figure(figsize=(9, 7))
    subset = feature_stats.head(top_labels)
    scatter = plt.scatter(
        subset["point_biserial"],
        subset["cohens_d"],
        s=120,
        c=subset["univariate_auc"],
        cmap="viridis",
        edgecolor="black",
        alpha=0.8,
    )
    for _, row in subset.iterrows():
        plt.text(row["point_biserial"] + 0.01, row["cohens_d"] + 0.01, row["feature"], fontsize=9)
    plt.axhline(0, color="grey", linestyle="--", linewidth=1)
    plt.axvline(0, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Correlación punto-biserial con clase")
    plt.ylabel("Cohen's d")
    plt.title("Relación entre correlación y tamaño del efecto")
    cbar = plt.colorbar(scatter)
    cbar.set_label("AUC univariado")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_top_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    feature_stats: pd.DataFrame,
    output_path: Path,
    top_k: int = 5,
) -> None:
    if feature_stats.empty:
        return
    top = feature_stats.head(top_k)

    n_cols = min(top_k, 3)
    n_rows = int(np.ceil(top_k / n_cols))
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for subplot_idx, (_, row) in enumerate(top.iterrows(), start=1):
        plt.subplot(n_rows, n_cols, subplot_idx)
        column_idx = int(row["feature_idx"])
        feature_name = row["feature"]
        column_data = features[:, column_idx]
        df = pd.DataFrame({feature_name: column_data, "Clase": np.where(labels == 1, "SCD", "Healthy")})
        sns.kdeplot(data=df, x=feature_name, hue="Clase", fill=True, common_norm=False, alpha=0.3)
        plt.title(f"{feature_name}\nCohen's d = {row['cohens_d']:.2f}")
        plt.xlabel("Valor")
        plt.ylabel("Densidad")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    output_dir = ROOT / "posibles_gráficas"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = ROOT / "config.yaml"
    config = prepare_config(load_config(config_path))

    analyzer = HologramAnalyzer(config_dict=config)
    images, labels, _ = analyzer._load_dataset()
    features = analyzer._extract_features(images)

    pipeline, training_results = analyzer._train_model(features, labels)
    analyzer._validate_model(features, labels, pipeline)

    cv = StratifiedKFold(n_splits=config.get("validation", {}).get("n_cv_folds", 5), shuffle=True, random_state=42)

    feature_stats = compute_feature_statistics(features, labels, analyzer.feature_names)

    plot_learning_curve(
        pipeline,
        features,
        labels,
        cv,
        output_dir / "learning_curve_accuracy.png",
    )
    plot_cv_metrics(
        pipeline,
        features,
        labels,
        cv,
        output_dir / "cv_metric_distribution.png",
    )
    plot_feature_correlation(
        features,
        labels,
        analyzer.feature_names,
        training_results.get("selected_features", []),
        output_dir / "feature_correlation_heatmap.png",
    )
    plot_feature_pairplot(
        features,
        labels,
        analyzer.feature_names,
        training_results.get("selected_features", []),
        output_dir / "feature_pairplot.png",
    )
    plot_probability_distribution(
        pipeline,
        features,
        labels,
        cv,
        output_dir / "probability_density.png",
    )
    plot_anomaly_scores(
        features,
        labels,
        analyzer.anomaly_detector,
        output_dir / "anomaly_distance_distribution.png",
    )
    plot_top_feature_effects(
        feature_stats,
        output_dir / "top_features_effect_size.png",
    )
    plot_effect_vs_correlation(
        feature_stats,
        output_dir / "feature_effect_vs_correlation.png",
    )
    plot_top_feature_distributions(
        features,
        labels,
        feature_stats,
        output_dir / "top_feature_distributions.png",
    )

    print("✅ Gráficas generadas en:", output_dir)


if __name__ == "__main__":
    main()
