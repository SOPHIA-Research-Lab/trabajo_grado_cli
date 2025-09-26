#!/usr/bin/env python3
"""Generate supplementary figures for the hologram classification thesis."""

from __future__ import annotations

import copy
from pathlib import Path
import pickle
from typing import Dict, Tuple
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import patches
from matplotlib.colors import Normalize
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern, graycomatrix
from skimage.filters import gabor, threshold_otsu
from skimage.transform import resize
import pywt
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from hologram_analysis import HologramAnalyzer
from anomaly_detector import DistanceBasedAnomalyDetector

OUTPUT_DIR = ROOT / "manuscrito" / "proyecto_final" / "figuras"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / "config.yaml"
MODEL_PATH = ROOT / "results" / "hologram_model.pkl"
ANOMALY_PATH = ROOT / "results" / "anomaly_detector.pkl"

sns.set_context("talk")


def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_analyzer(
    config: Dict,
) -> Tuple[HologramAnalyzer, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], list[str]]:
    analyzer = HologramAnalyzer(config_dict=config)
    images, labels, _ = analyzer._load_dataset()
    features = analyzer._extract_features(images)
    feature_list = analyzer.feature_names.copy()
    feature_indices = {name: idx for idx, name in enumerate(feature_list)}
    return analyzer, images, features, labels, feature_indices, feature_list


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] == 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def figure_descriptor_examples(
    images: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Dict[str, int],
) -> None:
    class_map = {0: "Healthy", 1: "SCD"}
    lbp_indices = [feature_names[name] for name in feature_names if name.startswith("lbp_hist_")]
    fft_indices = [feature_names[name] for name in feature_names if name.startswith("fft_ring_")]
    wavelet_indices = [feature_names[name] for name in feature_names if name.startswith("wavelet_") and name.endswith("_energy")]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    for row, class_id in enumerate([0, 1]):
        idx = int(np.where(labels == class_id)[0][0])
        grayscale = convert_to_grayscale(images[idx])

        axes[row, 0].imshow(grayscale, cmap="gray")
        axes[row, 0].set_title(f"Holograma {class_map[class_id]}")
        axes[row, 0].axis("off")

        lbp_values = features[idx, lbp_indices]
        axes[row, 1].bar(range(len(lbp_values)), lbp_values, color="#1f77b4")
        axes[row, 1].set_title("Histograma LBP")
        axes[row, 1].set_xlabel("Bin")
        axes[row, 1].set_ylabel("Probabilidad")

        fft_values = features[idx, fft_indices]
        axes[row, 2].bar([f"Anillo {i+1}" for i in range(len(fft_values))], fft_values, color="#ff7f0e")
        axes[row, 2].set_title("Energía espectral")
        axes[row, 2].set_ylabel("Magnitud media")

        wavelet_values = features[idx, wavelet_indices]
        wavelet_labels = [name.replace("wavelet_", "").replace("_energy", "") for name in feature_names if name.startswith("wavelet_") and name.endswith("_energy")]
        axes[row, 3].bar(wavelet_labels, wavelet_values, color="#2ca02c")
        axes[row, 3].set_title("Energía wavelet")
        axes[row, 3].set_ylabel("Energía acumulada")
        axes[row, 3].tick_params(axis="x", rotation=30)

    fig.suptitle("Descriptores holográficos comparativos", fontweight="bold")
    fig.savefig(OUTPUT_DIR / "descriptor_profiles.png", dpi=300)
    plt.close(fig)


def figure_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    steps = [
        "Unificación\ny control de calidad",
        "Extracción\nde 84 descriptores",
        "Escalado +\nselección",
        "Ensamble ML",
        "Calibración +\nMahalanobis",
    ]

    x_positions = np.linspace(0.05, 0.85, len(steps))
    box_width = 0.16
    box_height = 0.55

    for i, (x, label) in enumerate(zip(x_positions, steps)):
        box = patches.FancyBboxPatch(
            (x, 0.2),
            box_width,
            box_height,
            boxstyle="round,pad=0.03",
            linewidth=2,
            edgecolor="#1f77b4",
            facecolor="#e3f2fd",
        )
        ax.add_patch(box)
        ax.text(x + box_width / 2, 0.47, label, ha="center", va="center", fontsize=11)

        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch(
                (x + box_width, 0.475),
                (x_positions[i + 1], 0.475),
                connectionstyle="arc3,rad=0.0",
                arrowstyle="->",
                mutation_scale=20,
                linewidth=2,
                color="#424242",
            )
            ax.add_patch(arrow)

    ax.text(0.5, 0.05, "Resumen del pipeline de clasificación holográfica", ha="center", fontsize=12, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "pipeline_overview.png", dpi=300)
    plt.close(fig)


def figure_tsne_anomaly(features: np.ndarray, labels: np.ndarray) -> None:
    model = joblib.load(MODEL_PATH)
    scaled = model.named_steps["scaler"].transform(features)
    selected = model.named_steps["selector"].transform(scaled)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    embedding = tsne.fit_transform(selected)

    detector = DistanceBasedAnomalyDetector.load(ANOMALY_PATH)
    distances = np.array([detector.predict(vec)["mahalanobis_distance"] for vec in features])
    threshold = detector.threshold

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(np.min(distances)), vmax=float(np.percentile(distances, 99)))

    markers = {0: "o", 1: "^"}
    class_labels = {0: "Healthy", 1: "SCD"}

    for class_id in np.unique(labels):
        mask = labels == class_id
        sc = ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=distances[mask],
            cmap=cmap,
            norm=norm,
            s=60,
            marker=markers[class_id],
            edgecolors="#212121",
            linewidths=0.4,
            alpha=0.9,
            label=f"{class_labels[class_id]} (n={mask.sum()})",
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Distancia de Mahalanobis")
    ax.set_title("Proyección t-SNE del espacio de características")
    ax.set_xlabel("Componente t-SNE 1")
    ax.set_ylabel("Componente t-SNE 2")
    ax.legend(loc="upper right")
    ax.text(0.02, 0.02, f"Umbral percentil 95 = {threshold:.2f}", transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.savefig(OUTPUT_DIR / "tsne_mahalanobis.png", dpi=300)
    plt.close(fig)


def normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    return arr


def figure_hologram_transforms(images: np.ndarray, labels: np.ndarray) -> None:
    idx = int(np.where(labels == 1)[0][0])  # seleccionar un holograma falciforme
    raw = images[idx]
    gray = convert_to_grayscale(raw).astype(np.float32)
    gray_norm = gray / 255.0

    fft_mag = np.fft.fftshift(np.fft.fft2(gray_norm))
    fft_mag = np.log1p(np.abs(fft_mag))
    fft_img = normalize_array(fft_mag)

    lbp_map = local_binary_pattern(gray_norm, P=24, R=3, method="uniform")
    lbp_img = normalize_array(lbp_map)

    gabor_real, gabor_imag = gabor(gray_norm, frequency=0.18)
    gabor_mag = normalize_array(np.sqrt(gabor_real**2 + gabor_imag**2))

    coeffs = pywt.dwt2(gray_norm, "db2")
    _, (cH, cV, cD) = coeffs
    wavelet_energy = np.sqrt(cH**2 + cV**2 + cD**2)
    wavelet_energy = resize(wavelet_energy, gray_norm.shape, mode="reflect", anti_aliasing=True)
    wavelet_img = normalize_array(wavelet_energy)

    thresh = threshold_otsu(gray_norm)
    mask = gray_norm > thresh
    mask = ndimage.binary_fill_holes(mask)
    mask = normalize_array(mask.astype(np.float32))

    quantized = np.floor(gray_norm * 31).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[5], angles=[0], levels=32, symmetric=True, normed=True)
    glcm_matrix = glcm[:, :, 0, 0]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Holograma original")
    axes[0, 0].axis("off")

    im_fft = axes[0, 1].imshow(fft_img, cmap="magma")
    axes[0, 1].set_title("Magnitud logarítmica de la FFT")
    axes[0, 1].axis("off")
    fig.colorbar(im_fft, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_lbp = axes[0, 2].imshow(lbp_img, cmap="viridis")
    axes[0, 2].set_title("Mapa LBP uniforme")
    axes[0, 2].axis("off")
    fig.colorbar(im_lbp, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_gabor = axes[1, 0].imshow(gabor_mag, cmap="inferno")
    axes[1, 0].set_title("Respuesta de filtro de Gabor")
    axes[1, 0].axis("off")
    fig.colorbar(im_gabor, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_wavelet = axes[1, 1].imshow(wavelet_img, cmap="cividis")
    axes[1, 1].set_title("Energía de coeficientes wavelet")
    axes[1, 1].axis("off")
    fig.colorbar(im_wavelet, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_glcm = axes[1, 2].imshow(glcm_matrix, cmap="plasma")
    axes[1, 2].set_title("Matriz de coocurrencia (GLCM)")
    axes[1, 2].set_xlabel("Nivel $i$")
    axes[1, 2].set_ylabel("Nivel $j$")
    fig.colorbar(im_glcm, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Transformaciones clave para la caracterización holográfica", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "hologram_transforms.png", dpi=220)
    plt.close(fig)


def bootstrap_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 2000) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(42)
    n = len(y_true)
    indices = np.arange(n)
    scores = []
    for _ in range(n_bootstrap):
        sample = rng.choice(indices, size=n, replace=True)
        scores.append(accuracy_score(y_true[sample], y_pred[sample]))
    scores = np.array(scores)
    mean_score = float(np.mean(scores))
    ci_low, ci_high = np.percentile(scores, [2.5, 97.5])
    return mean_score, (float(ci_low), float(ci_high))


def figure_performance_uncertainty(
    base_config: Dict,
    features: np.ndarray,
    labels: np.ndarray,
    feature_list: list[str],
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    config_mode = copy.deepcopy(base_config)
    config_mode.setdefault("execution", {})["mode"] = "quick"
    analyzer = HologramAnalyzer(config_dict=config_mode)
    analyzer._adjust_config_for_mode("quick")
    analyzer.anomaly_detector = object()
    analyzer.feature_names = feature_list
    _, results = analyzer._train_model(features, labels)

    y_test = results["y_test"]
    y_pred = results["y_pred"]
    y_proba = results["y_proba"]

    acc_mean, (ci_low, ci_high) = bootstrap_accuracy(y_test, y_pred)
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=8)
    brier = float(np.mean((y_proba - y_test) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0 = axes[0]
    ax0.bar(["Modelo final"], [acc_mean], color="#1976d2", alpha=0.9)
    ax0.errorbar(
        0,
        acc_mean,
        yerr=[[acc_mean - ci_low], [ci_high - acc_mean]],
        fmt="none",
        ecolor="#212121",
        capsize=6,
        linewidth=1.4,
    )
    ax0.set_ylim(0.8, 1.0)
    ax0.set_ylabel("Exactitud")
    ax0.set_title("Exactitud con intervalo bootstrap 95%")

    ax1 = axes[1]
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración ideal")
    ax1.plot(mean_pred, frac_pos, marker="o", color="#d32f2f", label="Modelo final")
    ax1.set_xlabel("Probabilidad pronosticada")
    ax1.set_ylabel("Frecuencia observada")
    ax1.set_title(f"Curva de confiabilidad (Brier = {brier:.3f})")
    ax1.legend()

    fig.suptitle("Desempeño, incertidumbre y calibración", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "performance_uncertainty.png", dpi=300)
    plt.close(fig)

    print(
        f"Métricas modelo final: exactitud bootstrap = {acc_mean:.3f}"
        f" [{ci_low:.3f}, {ci_high:.3f}], Brier = {brier:.3f}"
    )

    return acc_mean, ci_low, ci_high, mean_pred, frac_pos


def main() -> None:
    base_config = load_config()
    analyzer, images, features, labels, feature_map, feature_list = prepare_analyzer(copy.deepcopy(base_config))

    figure_descriptor_examples(images, features, labels, feature_map)
    figure_pipeline_overview()
    figure_tsne_anomaly(features, labels)
    figure_hologram_transforms(images, labels)
    acc, ci_low, ci_high, mean_pred, frac_pos = figure_performance_uncertainty(
        base_config, features, labels, feature_list
    )

    print("Resumen calibración (promedio prob.):", mean_pred)
    print("Resumen calibración (frecuencia obs.):", frac_pos)

    print(f"Figuras generadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
