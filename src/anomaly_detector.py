#!/usr/bin/env python3
"""
Anomaly Detector para Hologramas
Sistema de detección de anomalías basado en distancia de Mahalanobis
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

class DistanceBasedAnomalyDetector:
    """
    Detector de anomalías basado en distancia de Mahalanobis
    
    Entrena con datos de células sanguíneas (Healthy + SCD) y detecta
    muestras que están significativamente lejos del espacio de características normal.
    """
    
    def __init__(self, threshold_percentile: float = 95.0):
        """
        Inicializa el detector de anomalías
        
        Args:
            threshold_percentile: Percentil para definir el umbral de anomalía (95.0 = 5% falsos positivos)
        """
        self.threshold_percentile = threshold_percentile
        self.is_trained = False
        
        # Estadísticas de referencia
        self.mean_features = None
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.threshold = None
        self.feature_names = None
        
        # Estadísticas por clase (para análisis detallado)
        self.class_stats = {}
        
    def fit(self, features: np.ndarray, labels: np.ndarray = None, feature_names: List[str] = None):
        """
        Entrena el detector con datos de células sanguíneas
        
        Args:
            features: Array de características (n_samples, n_features)
            labels: Etiquetas opcionales para análisis por clase
            feature_names: Nombres de las características
        """
        features = np.array(features)
        
        if features.ndim != 2:
            raise ValueError(f"Features debe ser 2D, recibido: {features.shape}")
        
        print(f"🔧 Entrenando detector de anomalías con {features.shape[0]} muestras y {features.shape[1]} características")
        
        # Calcular estadísticas globales
        self.mean_features = np.mean(features, axis=0)
        self.cov_matrix = np.cov(features, rowvar=False)
        
        # Añadir regularización para evitar matrices singulares
        regularization = 1e-6
        self.cov_matrix += regularization * np.eye(self.cov_matrix.shape[0])
        
        try:
            self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            print("⚠️  Matriz de covarianza singular, usando pseudoinversa")
            self.inv_cov_matrix = np.linalg.pinv(self.cov_matrix)
        
        # Calcular distancias para todas las muestras de entrenamiento
        train_distances = []
        for i, sample in enumerate(features):
            distance = mahalanobis(sample, self.mean_features, self.inv_cov_matrix)
            train_distances.append(distance)
        
        train_distances = np.array(train_distances)
        
        # Establecer umbral basado en percentil
        self.threshold = np.percentile(train_distances, self.threshold_percentile)
        
        # Guardar nombres de características
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
        
        # Calcular estadísticas por clase si se proporcionan etiquetas
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                class_mask = labels == label
                class_features = features[class_mask]
                
                self.class_stats[label] = {
                    'mean': np.mean(class_features, axis=0),
                    'cov': np.cov(class_features, rowvar=False),
                    'count': np.sum(class_mask),
                    'distances': train_distances[class_mask]
                }
        
        self.is_trained = True
        
        print(f"✅ Detector entrenado")
        print(f"   - Umbral de anomalía: {self.threshold:.3f}")
        print(f"   - Percentil usado: {self.threshold_percentile}%")
        print(f"   - Distancia media en entrenamiento: {np.mean(train_distances):.3f}")
        print(f"   - Distancia máxima en entrenamiento: {np.max(train_distances):.3f}")
        
    def predict(self, features: np.ndarray) -> Dict[str, Union[bool, float, np.ndarray]]:
        """
        Detecta si una muestra es anómala
        
        Args:
            features: Características de la muestra (1D array)
            
        Returns:
            Dict con información de anomalía
        """
        if not self.is_trained:
            raise ValueError("El detector debe ser entrenado primero")
        
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        results = []
        
        for sample in features:
            # Calcular distancia de Mahalanobis
            distance = mahalanobis(sample, self.mean_features, self.inv_cov_matrix)
            
            # Determinar si es anomalía
            is_anomaly = distance > self.threshold
            
            # Calcular p-value aproximado (asumiendo distribución chi-cuadrada)
            # Los grados de libertad son el número de características
            dof = len(sample)
            p_value = 1 - chi2.cdf(distance**2, dof)
            
            # Calcular score normalizado (0-1, donde 1 es más anómalo)
            anomaly_score = min(distance / self.threshold, 2.0) / 2.0  # Cap at 2x threshold
            
            # Calcular distancias a cada clase si están disponibles
            class_distances = {}
            if self.class_stats:
                for class_name, stats in self.class_stats.items():
                    try:
                        class_inv_cov = np.linalg.inv(stats['cov'] + 1e-6 * np.eye(stats['cov'].shape[0]))
                        class_distance = mahalanobis(sample, stats['mean'], class_inv_cov)
                        class_distances[class_name] = class_distance
                    except np.linalg.LinAlgError:
                        class_distances[class_name] = float('inf')
            
            result = {
                'is_anomaly': is_anomaly,
                'mahalanobis_distance': distance,
                'threshold': self.threshold,
                'anomaly_score': anomaly_score,
                'p_value': p_value,
                'class_distances': class_distances,
                'closest_class': min(class_distances.items(), key=lambda x: x[1])[0] if class_distances else None
            }
            
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def get_feature_contributions(self, features: np.ndarray, top_n: int = 10) -> Dict[str, float]:
        """
        Identifica qué características contribuyen más a la anomalía
        
        Args:
            features: Características de la muestra
            top_n: Número de características principales a retornar
            
        Returns:
            Dict con contribuciones de características
        """
        if not self.is_trained:
            raise ValueError("El detector debe ser entrenado primero")
        
        features = np.array(features).flatten()
        
        # Calcular diferencias respecto a la media
        diff = features - self.mean_features
        
        # Calcular contribuciones usando la forma cuadrática de Mahalanobis
        # d² = (x-μ)ᵀ Σ⁻¹ (x-μ)
        contributions = np.abs(diff * np.diag(self.inv_cov_matrix))
        
        # Crear diccionario con nombres de características
        feature_contributions = dict(zip(self.feature_names, contributions))
        
        # Ordenar por contribución descendente
        sorted_contributions = dict(
            sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        return sorted_contributions
    
    def save(self, filepath: Union[str, Path]):
        """Guarda el detector entrenado"""
        filepath = Path(filepath)
        
        detector_data = {
            'threshold_percentile': self.threshold_percentile,
            'is_trained': self.is_trained,
            'mean_features': self.mean_features,
            'cov_matrix': self.cov_matrix,
            'inv_cov_matrix': self.inv_cov_matrix,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'class_stats': self.class_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(detector_data, f)
        
        print(f"💾 Detector guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """Carga un detector previamente entrenado"""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            detector_data = pickle.load(f)
        
        detector = cls(detector_data['threshold_percentile'])
        
        # Restaurar estado
        detector.is_trained = detector_data['is_trained']
        detector.mean_features = detector_data['mean_features']
        detector.cov_matrix = detector_data['cov_matrix']
        detector.inv_cov_matrix = detector_data['inv_cov_matrix']
        detector.threshold = detector_data['threshold']
        detector.feature_names = detector_data['feature_names']
        detector.class_stats = detector_data['class_stats']
        
        print(f"📂 Detector cargado desde: {filepath}")
        return detector
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del detector"""
        if not self.is_trained:
            return {"error": "Detector no entrenado"}
        
        stats = {
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        if self.class_stats:
            stats['class_statistics'] = {}
            for class_name, class_stat in self.class_stats.items():
                stats['class_statistics'][class_name] = {
                    'count': class_stat['count'],
                    'mean_distance': np.mean(class_stat['distances']),
                    'max_distance': np.max(class_stat['distances'])
                }
        
        return stats


def create_anomaly_detector_from_training_data(data_dir: str = "data/unified") -> DistanceBasedAnomalyDetector:
    """
    Crea y entrena un detector de anomalías usando los datos de entrenamiento existentes
    
    Args:
        data_dir: Directorio con datos unificados
        
    Returns:
        Detector entrenado
    """
    from hologram_analysis import HologramAnalyzer
    import os
    
    # Inicializar analizador para extraer características
    analyzer = HologramAnalyzer("config.yaml")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directorio de datos no encontrado: {data_path}")
    
    print(f"🔍 Cargando datos de entrenamiento desde: {data_path}")
    
    features_list = []
    labels_list = []
    
    # Procesar cada clase
    for class_dir in ["Healthy", "SCD"]:
        class_path = data_path / class_dir
        
        if not class_path.exists():
            continue
            
        image_files = list(class_path.glob("*.png"))
        
        print(f"📁 Procesando {class_dir}: {len(image_files)} imágenes")
        
        for img_path in image_files[:5]:  # Limitar para demo
            try:
                # Cargar y procesar imagen
                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Redimensionar a tamaño estándar (similar al analizador)
                img_resized = cv2.resize(img, (512, 512))
                
                # Crear batch con una imagen
                images_batch = np.expand_dims(img_resized, axis=0)
                
                # Extraer características usando el analizador
                features_batch = analyzer._extract_features(images_batch)
                features = features_batch[0]  # Tomar primera muestra
                
                features_list.append(features)
                labels_list.append(class_dir)
                
            except Exception as e:
                print(f"⚠️  Error procesando {img_path}: {e}")
                continue
    
    if not features_list:
        raise ValueError("No se pudieron cargar características de entrenamiento")
    
    # Convertir a arrays numpy
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    print(f"✅ Datos cargados: {features_array.shape[0]} muestras, {features_array.shape[1]} características")
    
    # Crear y entrenar detector
    detector = DistanceBasedAnomalyDetector(threshold_percentile=95.0)
    detector.fit(features_array, labels_array, analyzer.feature_names)
    
    return detector


if __name__ == "__main__":
    # Demo del detector de anomalías
    print("🧪 DEMO: Detector de Anomalías Basado en Distancia")
    print("=" * 60)
    
    try:
        # Crear detector desde datos de entrenamiento
        detector = create_anomaly_detector_from_training_data()
        
        # Guardar detector
        detector.save("results/anomaly_detector.pkl")
        
        print("✅ Detector creado y guardado exitosamente")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()