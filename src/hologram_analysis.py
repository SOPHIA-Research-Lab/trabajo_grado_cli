#!/usr/bin/env python3
"""
Hologram Classifier
Sistema unificado de clasificación de hologramas con interpretabilidad básica
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports de ML
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import filters, measure, morphology
from skimage.filters import gabor
from skimage.segmentation import clear_border
from scipy import ndimage, stats
from scipy.stats import entropy
import pywt
import joblib
from multiprocessing import Pool, cpu_count
import time
import hashlib
import pickle
from tqdm import tqdm
# Import dataset unifier
try:
    from .dataset_unifier import create_unified_dataset_if_needed
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from dataset_unifier import create_unified_dataset_if_needed

class HologramAnalyzer:
    """
    Clase que realiza el analisis
    """
    
    def __init__(self, config_path: str = "config.yaml", config_dict: Optional[Dict] = None):
        """
        Inicializar analizador con configuración
        """
        if config_dict:
            self.config = config_dict
        else:
            self.config = self._load_config(config_path)
        
        self.results = {}
        self.feature_names = []
        
        # Crear directorio de resultados
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        print("Inicializando HologramAnalyzer")
        print("="*60)
    
    def _load_config(self, config_path: str) -> Dict:
        """Cargar configuración desde YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuración cargada desde: {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ No se encontró {config_path}, usando configuración por defecto")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuración por defecto"""
        return {
            'dataset_path': '../unified_dataset',
            'output_dir': './results',
            'model': {'target_size': [512, 512], 'test_size': 0.2, 'random_state': 42, 'top_k_features': 20},
            'features': {'lbp': {'radius': 3, 'points': 24}, 'glcm': {'distances': [5], 'angles': [0]}, 'fft': {'n_rings': 3}},
            'validation': {'use_loocv': True, 'n_cv_folds': 5, 'stability_iterations': 30},
            'visualization': {'save_plots': True, 'show_top_features': 10, 'include_interpretability': True}
        }
    
    def _adjust_config_for_mode(self, mode: str):
        """Ajustar configuración según el modo de ejecución"""
        if mode == 'quick':
            # TEST Modo rápido: menos características, sin optimización de hyprerparámetros
            self.config['model']['auto_optimize'] = False
            self.config['model']['top_k_features'] = min(20, self.config['model'].get('top_k_features', 20))
            self.config['validation']['use_loocv'] = False
            self.config['validation']['n_cv_folds'] = 3
            self.config['visualization']['show_top_features'] = 5
            print(f"   ⚡ Modo rápido activado: análisis optimizado para velocidad")
            
        elif mode == 'deep':
            # Modo profundo: máximo análisis, todas las características
            self.config['model']['auto_optimize'] = True
            self.config['model']['optimization_time'] = 10  # más tiempo para optimización
            self.config['model']['top_k_features'] = min(50, self.config['model'].get('top_k_features', 35) + 15)
            self.config['validation']['stability_iterations'] = 50
            self.config['visualization']['show_top_features'] = 15
            print(f"   🔬 Modo profundo activado: análisis exhaustivo con optimización máxima")
            
        
    
    def run_complete_analysis(self) -> Dict:
        """
        Ejecutar análisis completo end-to-end con modos optimizados
        """
        execution_mode = self.config.get('execution', {}).get('mode', 'full')
        
        mode_icons = {'quick': '⚡', 'full': '🚀', 'deep': '🔬'}
        mode_names = {'quick': 'RÁPIDO', 'full': 'COMPLETO', 'deep': 'PROFUNDO'}
        
        print(f"\n{mode_icons.get(execution_mode, '🚀')} INICIANDO ANÁLISIS {mode_names.get(execution_mode, 'COMPLETO')}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Ajustar configuraciones según el modo
        self._adjust_config_for_mode(execution_mode)
        
        # 1. Cargar datos
        print(f"\n1️⃣ Cargando dataset...")
        X, y, metadata = self._load_dataset()
        
        # 2. Extraer características
        print(f"\n2️⃣ Extrayendo características...")
        X_features = self._extract_features(X)
        
        # 3. Entrenar modelo
        print(f"\n3️⃣ Entrenando modelo...")
        model, train_results = self._train_model(X_features, y)
        
        # 4. Validación robusta
        print(f"\n4️⃣ Validación robusta...")
        validation_results = self._validate_model(X_features, y, model)
        
        # 5. Análisis de interpretabilidad
        print(f"\n5️⃣ Análisis de interpretabilidad...")
        interpretation_results = self._analyze_interpretability(X_features, y, model)
        
        # Consolidar resultados primero para visualizaciones
        self.results = {
            'model': model,
            'training': train_results,
            'validation': validation_results,
            'interpretation': interpretation_results,
            'metadata': metadata,
        }
        
        # 6. Generar visualizaciones
        print(f"\n6️⃣ Generando visualizaciones...")
        self._create_visualizations(X_features, y, model, interpretation_results)
        
        # 7. Generar reporte
        print(f"\n7️⃣ Generando reporte final...")
        report = self._generate_report(train_results, validation_results, interpretation_results, metadata)
        
        # Añadir reporte a resultados
        self.results['report'] = report
        
        print(f"\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"📁 Resultados guardados en: {self.config['output_dir']}")
        
        return self.results
    
    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Cargar dataset unificado"""
        dataset_path = self.config['dataset_path']
        
        # Si el dataset unificado no existe, crearlo automáticamente
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset unificado no encontrado en: {dataset_path}")
            # Intentar crear desde los datos base
            data_base_path = self.config.get('data_dir', './data')
            if os.path.exists(data_base_path):
                print(f"🔧 Creando dataset unificado desde: {data_base_path}")
                dataset_path = create_unified_dataset_if_needed(data_base_path)
                print(f"✅ Dataset unificado creado en: {dataset_path}")
            else:
                raise FileNotFoundError(f"No se encuentra ni el dataset unificado ni los datos base en: {data_base_path}")
        
        images = []
        labels = []
        filenames = []
        
        # Cargar clases
        classes = {'Healthy': 0, 'SCD': 1}
        
        for class_name, class_label in classes.items():
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith('.png')]
            print(f"   {class_name}: {len(image_files)} imágenes")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Redimensionar
                    target_size = tuple(self.config['model']['target_size'])
                    img_resized = cv2.resize(img, target_size)
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_normalized = img_rgb.astype(np.float32) / 255.0
                    
                    images.append(img_normalized)
                    labels.append(class_label)
                    filenames.append(img_file)
        
        metadata = {
            'total_images': len(images),
            'healthy_count': sum(1 for l in labels if l == 0),
            'scd_count': sum(1 for l in labels if l == 1),
            'filenames': filenames,
            'classes': list(classes.keys())
        }
        
        print(f"   Total: {metadata['total_images']} imágenes")
        print(f"   Healthy: {metadata['healthy_count']}, SCD: {metadata['scd_count']}")
        
        return np.array(images), np.array(labels), metadata
    
    def _get_features_cache_key(self, images: np.ndarray) -> str:
        """Generar clave única para cache de características"""
        # Hash basado en configuración y datos
        config_str = str(sorted(self.config['features'].items()))
        data_hash = hashlib.md5(images.tobytes()).hexdigest()[:16]
        return f"features_{data_hash}_{hashlib.md5(config_str.encode()).hexdigest()[:8]}"
    
    def _load_cached_features(self, cache_key: str) -> Optional[np.ndarray]:
        """Cargar características desde cache"""
        cache_file = f"{self.config['output_dir']}/cache/{cache_key}.pkl"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"   ✅ Características cargadas desde cache")
                return cached_data['features'], cached_data['feature_names']
            except:
                pass
        return None, None
    
    def _save_features_to_cache(self, cache_key: str, features: np.ndarray, feature_names: List[str]):
        """Guardar características en cache"""
        cache_dir = f"{self.config['output_dir']}/cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{cache_key}.pkl"
        
        cached_data = {
            'features': features,
            'feature_names': feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except:
            pass  # Si no se puede guardar cache, continuar
    
    def _extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extraer características avanzadas mejoradas con cache"""
        
        # Verificar cache si está habilitado
        if self.config.get('execution', {}).get('cache_features', False):
            cache_key = self._get_features_cache_key(images)
            cached_features, cached_names = self._load_cached_features(cache_key)
            if cached_features is not None:
                self.feature_names = cached_names
                return cached_features
        
        features_list = []
        
        # Nombres de características expandidas
        lbp_names = [f'lbp_hist_{i}' for i in range(self.config['features']['lbp']['points'] + 2)]
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        glcm_names = [f'glcm_{prop}' for prop in glcm_props]
        fft_names = [f'fft_ring_{i}' for i in range(self.config['features']['fft']['n_rings'])]
        hu_names = [f'hu_moment_{i}' for i in range(7)]
        

        gabor_names = [f'gabor_mean_{i}' for i in range(8)] + [f'gabor_std_{i}' for i in range(8)]
        stat_names = ['mean_intensity', 'std_intensity', 'skewness', 'kurtosis', 'entropy']
        edge_names = ['edge_density', 'edge_mean', 'edge_std']
        wavelet_names = [f'wavelet_{comp}_{stat}' for comp in ['cA', 'cH', 'cV', 'cD'] for stat in ['mean', 'std', 'energy']]
        morpho_names = ['solidity', 'extent', 'aspect_ratio', 'eccentricity', 'convex_area_ratio']
        
        self.feature_names = (lbp_names + glcm_names + fft_names + hu_names + 
                             ['circularity'] + gabor_names + stat_names + 
                             edge_names + wavelet_names + morpho_names)
        

        use_progress = self.config.get('execution', {}).get('progress_bar', True)
        iterator = tqdm(enumerate(images), total=len(images), desc="   Extrayendo características") if use_progress else enumerate(images)
        
        for i, image in iterator:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (image * 255).astype(np.uint8)
            
            # Normalizar para algunas operaciones
            gray_norm = gray / 255.0
            
            features = []
            
            # 1. Local Binary Pattern
            lbp_radius = self.config['features']['lbp']['radius']
            lbp_points = self.config['features']['lbp']['points']
            lbp = local_binary_pattern(gray, lbp_points, lbp_radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_points+2, range=(0, lbp_points+2), density=True)
            features.extend(lbp_hist)
            
            # 2. GLCM
            distances = self.config['features']['glcm']['distances']
            angles = self.config['features']['glcm']['angles']
            glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            
            for prop in glcm_props:
                glcm_prop = graycoprops(glcm, prop).ravel()[0]
                features.append(glcm_prop)
            
            # 3. FFT Features
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift) + 1)
            
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            
            # Crear anillos
            y, x = np.ogrid[:rows, :cols]
            center_dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
            
            n_rings = self.config['features']['fft']['n_rings']
            for ring_idx in range(n_rings):
                r_inner = ring_idx * cols // (2 * n_rings)
                r_outer = (ring_idx + 1) * cols // (2 * n_rings)
                ring_mask = (center_dist >= r_inner) & (center_dist < r_outer)
                ring_energy = np.mean(magnitude_spectrum[ring_mask])
                features.append(ring_energy)
            
            # 4. Hu Moments + Circularity 
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                moments = cv2.moments(main_contour)
                hu_moments = cv2.HuMoments(moments).flatten()
                
                # Circularity
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                circularity = 0 if perimeter == 0 else (4 * np.pi * area) / (perimeter**2)
                
                features.extend(hu_moments)
                features.append(circularity)
            else:
                # Si no hay contornos, llenar con zeros
                features.extend([0] * 8)
            

            
            # 5.1 Gabor Filters (8 orientaciones)
            for theta in np.arange(0, np.pi, np.pi / 8):
                filt_real, filt_imag = gabor(gray_norm, frequency=0.6, theta=theta)
                features.append(np.mean(filt_real))
                features.append(np.std(filt_real))
            
            # 5.2 Statistical Features
            features.append(np.mean(gray_norm))  # mean_intensity
            features.append(np.std(gray_norm))   # std_intensity
            features.append(stats.skew(gray_norm.ravel()))  # skewness
            features.append(stats.kurtosis(gray_norm.ravel()))  # kurtosis
            
            # Entropy
            hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
            hist = hist[hist > 0]  # Remove zeros for log
            features.append(entropy(hist))  # entropy
            
            # 5.3 Edge Features
            edges = filters.sobel(gray_norm)
            features.append(np.mean(edges > 0.1))  # edge_density
            features.append(np.mean(edges))        # edge_mean
            features.append(np.std(edges))         # edge_std
            
            # 5.4 Wavelet Features
            coeffs = pywt.dwt2(gray_norm, 'db4')
            cA, (cH, cV, cD) = coeffs
            
            for comp in [cA, cH, cV, cD]:
                features.append(np.mean(comp))     # mean
                features.append(np.std(comp))      # std
                features.append(np.sum(comp**2))   # energy
            
            # 5.5 Advanced Morphological Features
            if contours:
                # Convex hull
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(main_contour)
                rect_area = w * h
                
                # Features
                features.append(area / hull_area if hull_area > 0 else 0)  # solidity
                features.append(area / rect_area if rect_area > 0 else 0)  # extent
                features.append(w / h if h > 0 else 0)  # aspect_ratio
                
                # Eccentricity from fitted ellipse
                if len(main_contour) >= 5:
                    ellipse = cv2.fitEllipse(main_contour)
                    (center, axes, orientation) = ellipse
                    majoraxis_length = max(axes)
                    minoraxis_length = min(axes)
                    eccentricity = np.sqrt(1 - (minoraxis_length/majoraxis_length)**2) if majoraxis_length > 0 else 0
                    features.append(eccentricity)
                else:
                    features.append(0)
                
                features.append(hull_area / area if area > 0 else 0)  # convex_area_ratio
            else:
                # Si no hay contornos
                features.extend([0] * 5)
            
            features_list.append(features)
            
            # Progreso
            if (i + 1) % 50 == 0:
                print(f"   Procesadas {i + 1}/{len(images)} imágenes...")
        
        X_features = np.array(features_list)
        X_features = np.nan_to_num(X_features)  # Manejar NaNs
        
        # Guardar en cache si está habilitado
        if self.config.get('execution', {}).get('cache_features', False):
            cache_key = self._get_features_cache_key(images)
            self._save_features_to_cache(cache_key, X_features, self.feature_names)
            print(f"   💾 Características guardadas en cache")
        
        print(f"   Características extraídas: {X_features.shape}")
        print(f"   Nombres de características: {len(self.feature_names)}")
        
        return X_features
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Optimización automática de hiperparámetros"""
        if not self.config.get('model', {}).get('auto_optimize', False):
            return {}
        
        print(f"   🔧 Optimizando hiperparámetros...")
        start_time = time.time()
        max_time = self.config.get('model', {}).get('optimization_time', 5) * 60  # minutos a segundos
        
        # Definir espacios de búsqueda
        param_distributions = {
            'selector__k': [20, 25, 30, 35, 40, 45, 50],
            'classifier__lr__C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'classifier__rf__n_estimators': [50, 100, 150],
            'classifier__rf__max_depth': [5, 10, 15, None],
            'classifier__svm__C': [0.5, 1.0, 2.0],
            'classifier__svm__gamma': ['scale', 'auto'],
            'classifier__gb__n_estimators': [50, 100, 150],
            'classifier__gb__learning_rate': [0.05, 0.1, 0.15]
        }
        
        # Pipeline base para optimización
        base_ensemble = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(random_state=42)),
                ('svm', SVC(probability=True, random_state=42)),
                ('gb', GradientBoostingClassifier(random_state=42))
            ],
            voting='soft'
        )
        
        base_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('selector', SelectKBest(f_classif)),
            ('classifier', base_ensemble)
        ])
        
        # Búsqueda aleatoria con tiempo limitado
        random_search = RandomizedSearchCV(
            base_pipeline,
            param_distributions,
            n_iter=20,  # Número limitado para velocidad
            cv=3,       # CV reducido para velocidad
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1
        )
        
        try:
            random_search.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            
            if elapsed_time < max_time:
                best_params = random_search.best_params_
                best_score = random_search.best_score_
                print(f"   ✅ Optimización completada en {elapsed_time:.1f}s")
                print(f"   📈 Mejor CV Score: {best_score:.3f}")
                return best_params
            else:
                print(f"    Tiempo límite alcanzado, usando parámetros por defecto")
                return {}
                
        except Exception as e:
            print(f"    Error en optimización, usando parámetros por defecto")
            return {}
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Pipeline, Dict]:
        """Entrenar modelo con ensemble methods"""
        
        # Dividir datos
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']
        k_features = self.config['model'].get('top_k_features', 30)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Optimización de hiperparámetros
        best_params = self._optimize_hyperparameters(X_train, y_train)
        
        # Aplicar mejores parámetros si están disponibles
        if best_params:
            k_features = best_params.get('selector__k', k_features)
        
        # Crear clasificadores base
        lr_classifier = LogisticRegression(penalty='l2', C=1.0, random_state=random_state, max_iter=1000)
        rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
        svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=random_state)
        gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
        
        # Ensemble con Voting Classifier
        ensemble_classifier = VotingClassifier(
            estimators=[
                ('lr', lr_classifier),
                ('rf', rf_classifier), 
                ('svm', svm_classifier),
                ('gb', gb_classifier)
            ],
            voting='soft'
        )
        
        # Pipeline con selección de características mejorada
        model = Pipeline([
            ('scaler', RobustScaler()),  # Más robusto a outliers
            ('selector', SelectKBest(f_classif, k=k_features)),
            ('classifier', ensemble_classifier)
        ])
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Evaluar
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Características seleccionadas
        selected_features = model.named_steps['selector'].get_support()
        selected_feature_names = [self.feature_names[i] for i, selected in enumerate(selected_features) if selected]
        
        # Evaluación adicional de clasificadores individuales
        individual_scores = {}
        for name, clf in [('LogisticRegression', lr_classifier), 
                         ('RandomForest', rf_classifier),
                         ('SVM', svm_classifier), 
                         ('GradientBoosting', gb_classifier)]:
            # Pipeline individual
            individual_model = Pipeline([
                ('scaler', RobustScaler()),
                ('selector', SelectKBest(f_classif, k=k_features)),
                ('classifier', clf)
            ])
            individual_model.fit(X_train, y_train)
            individual_scores[name] = {
                'train_score': individual_model.score(X_train, y_train),
                'test_score': individual_model.score(X_test, y_test),
                'auc': roc_auc_score(y_test, individual_model.predict_proba(X_test)[:, 1])
            }
        
        results = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'auc_score': auc_score,
            'selected_features': selected_feature_names,
            'individual_scores': individual_scores,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"   Train Score: {train_score:.3f}")
        print(f"   Test Score: {test_score:.3f}")
        print(f"   AUC Score: {auc_score:.3f}")
        print(f"   Características seleccionadas: {len(selected_feature_names)}")
        
        return model, results
    
    def _validate_model(self, X: np.ndarray, y: np.ndarray, model: Pipeline) -> Dict:
        """Validación robusta simplificada"""
        results = {}
        
        # 1. Cross-Validation Estratificada
        n_folds = self.config['validation']['n_cv_folds']
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_scores = cross_validate(
            model, X, y, cv=cv,
            scoring=['roc_auc', 'accuracy', 'precision', 'recall'],
            n_jobs=-1
        )
        
        results['cv_scores'] = {
            'auc_mean': np.mean(cv_scores['test_roc_auc']),
            'auc_std': np.std(cv_scores['test_roc_auc']),
            'accuracy_mean': np.mean(cv_scores['test_accuracy']),
            'accuracy_std': np.std(cv_scores['test_accuracy']),
            'precision_mean': np.mean(cv_scores['test_precision']),
            'precision_std': np.std(cv_scores['test_precision']),
            'recall_mean': np.mean(cv_scores['test_recall']),
            'recall_std': np.std(cv_scores['test_recall'])
        }
        
        print(f"   CV AUC: {results['cv_scores']['auc_mean']:.3f} ± {results['cv_scores']['auc_std']:.3f}")
        print(f"   CV Accuracy: {results['cv_scores']['accuracy_mean']:.3f} ± {results['cv_scores']['accuracy_std']:.3f}")
        
        # 2. Leave-One-Out (si está habilitado y el dataset no es muy grande)
        if self.config['validation']['use_loocv'] and len(X) <= 500:
            print(f"   Ejecutando Leave-One-Out CV...")
            loo = LeaveOneOut()
            
            loo_predictions = []
            loo_true = []
            
            for i, (train_idx, test_idx) in enumerate(loo.split(X)):
                X_train_loo, X_test_loo = X[train_idx], X[test_idx]
                y_train_loo, y_test_loo = y[train_idx], y[test_idx]
                
                model.fit(X_train_loo, y_train_loo)
                pred = model.predict(X_test_loo)[0]
                
                loo_predictions.append(pred)
                loo_true.append(y_test_loo[0])
                
                if (i + 1) % 50 == 0:
                    print(f"     LOOCV: {i + 1}/{len(X)} completadas...")
            
            loo_accuracy = np.mean(np.array(loo_predictions) == np.array(loo_true))
            loo_errors = np.sum(np.array(loo_predictions) != np.array(loo_true))
            
            results['loocv'] = {
                'accuracy': loo_accuracy,
                'errors': loo_errors,
                'total_samples': len(X),
                'predictions': loo_predictions,
                'true_labels': loo_true
            }
            
            print(f"   LOOCV Accuracy: {loo_accuracy:.1%}")
            print(f"   LOOCV Errors: {loo_errors}/{len(X)}")
        
        return results
    
    def _analyze_interpretability(self, X: np.ndarray, y: np.ndarray, model: Pipeline) -> Dict:
        """Análisis básico de interpretabilidad"""
        results = {}
        
        # 1. Importancia de características (coeficientes del modelo)
        try:
            # Obtener coeficientes del clasificador
            classifier = model.named_steps['classifier']
            selector = model.named_steps['selector']
            
            if hasattr(classifier, 'coef_'):
                # Mapear coeficientes a características originales
                selected_mask = selector.get_support()
                selected_feature_names = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
                
                coefficients = classifier.coef_[0]
                
                # Crear DataFrame de importancia
                importance_df = pd.DataFrame({
                    'feature': selected_feature_names,
                    'coefficient': coefficients,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
                
                results['feature_importance'] = importance_df
                
                print(f"   Top 5 características más importantes:")
                for i, (_, row) in enumerate(importance_df.head().iterrows()):
                    print(f"     {i+1}. {row['feature']}: {row['coefficient']:.3f}")
        
        except Exception as e:
            print(f"   ⚠️ No se pudo extraer importancia de características: {e}")
            results['feature_importance'] = None
        
        # 2. Análisis estadístico básico por clase
        feature_stats = []
        
        for i, feature_name in enumerate(self.feature_names):
            if i < X.shape[1]:  # Asegurar que el índice es válido
                healthy_values = X[y == 0, i]
                scd_values = X[y == 1, i]
                
                # Estadísticas básicas
                healthy_mean = np.mean(healthy_values)
                scd_mean = np.mean(scd_values)
                
                # Cohen's d (tamaño del efecto)
                pooled_std = np.sqrt((np.var(healthy_values) + np.var(scd_values)) / 2)
                cohens_d = (scd_mean - healthy_mean) / pooled_std if pooled_std > 0 else 0
                
                feature_stats.append({
                    'feature': feature_name,
                    'healthy_mean': healthy_mean,
                    'scd_mean': scd_mean,
                    'cohens_d': cohens_d,
                    'abs_cohens_d': abs(cohens_d)
                })
        
        stats_df = pd.DataFrame(feature_stats).sort_values('abs_cohens_d', ascending=False)
        results['feature_statistics'] = stats_df
        
        print(f"   Top 5 características más discriminativas (Cohen's d):")
        for i, (_, row) in enumerate(stats_df.head().iterrows()):
            print(f"     {i+1}. {row['feature']}: d = {row['cohens_d']:.2f}")
        
        return results
    
    def _create_visualizations(self, X: np.ndarray, y: np.ndarray, model: Pipeline, interpretation_results: Dict):
        """Crear visualizaciones avanzadas mejoradas"""
        
        if not self.config['visualization']['save_plots']:
            return
        
        # 1. Matriz de confusión mejorada
        if 'confusion_matrix' in self.results.get('training', {}):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Matriz de confusión simple
            cm = self.results['training']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Healthy', 'SCD'], 
                       yticklabels=['Healthy', 'SCD'], ax=ax1)
            ax1.set_title('Matriz de Confusión')
            ax1.set_ylabel('Etiqueta Real')
            ax1.set_xlabel('Predicción')
            
            # Matriz de confusión normalizada
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges',
                       xticklabels=['Healthy', 'SCD'], 
                       yticklabels=['Healthy', 'SCD'], ax=ax2)
            ax2.set_title('Matriz de Confusión Normalizada')
            ax2.set_ylabel('Etiqueta Real')
            ax2.set_xlabel('Predicción')
            
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/confusion_matrix.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Curva ROC con comparación de modelos individuales
        if 'individual_scores' in self.results.get('training', {}):
            plt.figure(figsize=(10, 8))
            
            # ROC del ensemble
            y_test = self.results['training']['y_test']
            y_proba = self.results['training']['y_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, linewidth=3, label=f'Ensemble (AUC = {self.results["training"]["auc_score"]:.3f})')
            
            # Línea diagonal
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC - Comparación de Modelos')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/roc_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Comparación de rendimiento de modelos individuales
        if 'individual_scores' in self.results.get('training', {}):
            individual_scores = self.results['training']['individual_scores']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy comparison
            models = list(individual_scores.keys()) + ['Ensemble']
            train_accs = [individual_scores[m]['train_score'] for m in models[:-1]] + [self.results['training']['train_score']]
            test_accs = [individual_scores[m]['test_score'] for m in models[:-1]] + [self.results['training']['test_score']]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, train_accs, width, label='Train', alpha=0.7)
            ax1.bar(x + width/2, test_accs, width, label='Test', alpha=0.7)
            ax1.set_xlabel('Modelos')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Comparación de Accuracy por Modelo')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # AUC comparison
            aucs = [individual_scores[m]['auc'] for m in models[:-1]] + [self.results['training']['auc_score']]
            bars = ax2.bar(models, aucs, color=['skyblue', 'lightgreen', 'orange', 'pink', 'gold'])
            ax2.set_xlabel('Modelos')
            ax2.set_ylabel('AUC Score')
            ax2.set_title('Comparación de AUC por Modelo')
            ax2.set_xticklabels(models, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Añadir valores en las barras
            for bar, auc in zip(bars, aucs):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{auc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/model_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Importancia de características mejorada
        if interpretation_results.get('feature_importance') is not None:
            importance_df = interpretation_results['feature_importance']
            
            plt.figure(figsize=(12, 10))
            top_features = importance_df.head(self.config['visualization']['show_top_features'])
            
            # Crear gráfico horizontal con colores mejorados
            colors = ['darkred' if coef < 0 else 'darkblue' for coef in top_features['coefficient']]
            bars = plt.barh(range(len(top_features)), top_features['abs_coefficient'], color=colors)
            
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importancia Absoluta (|Coeficiente|)')
            plt.title(f'Top {len(top_features)} Características Más Importantes')
            plt.gca().invert_yaxis()
            
            # Añadir valores en las barras
            for i, (bar, coef) in enumerate(zip(bars, top_features['abs_coefficient'])):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{coef:.3f}', va='center', ha='left')
            
            # Añadir leyenda
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='darkred', label='Favorece Healthy')
            blue_patch = mpatches.Patch(color='darkblue', label='Favorece SCD')
            plt.legend(handles=[red_patch, blue_patch], loc='lower right')
            
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. Distribución de características discriminativas mejorada
        if interpretation_results.get('feature_statistics') is not None:
            stats_df = interpretation_results['feature_statistics']
            top_discriminative = stats_df.head(6)  # Top 6 para subplot 2x3
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
            
            for i, (_, row) in enumerate(top_discriminative.iterrows()):
                if i >= 6:
                    break
                    
                feature_idx = self.feature_names.index(row['feature'])
                if feature_idx < X.shape[1]:
                    healthy_vals = X[y == 0, feature_idx]
                    scd_vals = X[y == 1, feature_idx]
                    
                    # Histogramas con mejor estilo
                    axes[i].hist(healthy_vals, alpha=0.6, label='Healthy', bins=25, 
                               color='skyblue', edgecolor='navy')
                    axes[i].hist(scd_vals, alpha=0.6, label='SCD', bins=25,
                               color='orange', edgecolor='darkred')
                    
                    # Líneas de media
                    axes[i].axvline(np.mean(healthy_vals), color='blue', linestyle='--', alpha=0.8)
                    axes[i].axvline(np.mean(scd_vals), color='red', linestyle='--', alpha=0.8)
                    
                    axes[i].set_title(f"{row['feature']}\n(Cohen's d = {row['cohens_d']:.2f})", fontsize=10)
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle('Distribución de Características Más Discriminativas', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/feature_distributions.png", dpi=150, bbox_inches='tight')
            plt.close()
        
      
        if interpretation_results.get('stability_results') is not None:
            stability_data = interpretation_results['stability_results']
            if 'feature_stability' in stability_data:
                stability_df = pd.DataFrame(stability_data['feature_stability'])
                top_stable = stability_df.nlargest(15, 'selection_frequency')
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_stable)), top_stable['selection_frequency'])
                plt.yticks(range(len(top_stable)), top_stable['feature'])
                plt.xlabel('Frecuencia de Selección (%)')
                plt.title('Estabilidad de Características (Top 15)')
                plt.gca().invert_yaxis()
                
                # Añadir línea de referencia en 80%
                plt.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{self.config['output_dir']}/feature_stability.png", dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"   ✅ Visualizaciones avanzadas guardadas en {self.config['output_dir']}")
    
    def _generate_report(self, train_results: Dict, validation_results: Dict, 
                        interpretation_results: Dict, metadata: Dict) -> Dict:
        """Generar reporte final simplificado"""
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': metadata,
            'model_performance': {
                'train_accuracy': train_results['train_score'],
                'test_accuracy': train_results['test_score'],
                'auc_score': train_results['auc_score']
            }
        }
        
        # Agregar resultados de CV
        if 'cv_scores' in validation_results:
            report['cross_validation'] = validation_results['cv_scores']
        
        # Agregar LOOCV si existe
        if 'loocv' in validation_results:
            report['loocv'] = validation_results['loocv']
        
        # Guardar reporte en texto
        report_path = f"{self.config['output_dir']}/analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE ANÁLISIS - HOLOGRAM CLASSIFIER V2.0\n")
            f.write("="*80 + "\n")
            f.write(f"Fecha: {report['timestamp']}\n\n")
            
            # Dataset info
            f.write("INFORMACIÓN DEL DATASET\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total de imágenes: {metadata['total_images']}\n")
            f.write(f"Células sanas: {metadata['healthy_count']}\n")
            f.write(f"Células SCD: {metadata['scd_count']}\n")
            f.write(f"Balance: {metadata['healthy_count']/metadata['total_images']:.1%} / {metadata['scd_count']/metadata['total_images']:.1%}\n\n")
            
            # Performance
            f.write("RENDIMIENTO DEL MODELO\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy (Entrenamiento): {train_results['train_score']:.3f}\n")
            f.write(f"Accuracy (Test): {train_results['test_score']:.3f}\n")
            f.write(f"AUC Score: {train_results['auc_score']:.3f}\n\n")
            
            # Cross-validation
            if 'cv_scores' in validation_results:
                cv = validation_results['cv_scores']
                f.write("VALIDACIÓN CRUZADA (5-FOLD)\n")
                f.write("-" * 40 + "\n")
                f.write(f"AUC: {cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}\n")
                f.write(f"Accuracy: {cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}\n")
                f.write(f"Precision: {cv['precision_mean']:.3f} ± {cv['precision_std']:.3f}\n")
                f.write(f"Recall: {cv['recall_mean']:.3f} ± {cv['recall_std']:.3f}\n\n")
            
            # LOOCV
            if 'loocv' in validation_results:
                loo = validation_results['loocv']
                f.write("LEAVE-ONE-OUT CROSS-VALIDATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy: {loo['accuracy']:.1%}\n")
                f.write(f"Errores: {loo['errors']}/{loo['total_samples']}\n\n")
            
            # Top características
            if interpretation_results.get('feature_statistics') is not None:
                stats_df = interpretation_results['feature_statistics']
                f.write("TOP 10 CARACTERÍSTICAS MÁS DISCRIMINATIVAS\n")
                f.write("-" * 40 + "\n")
                for i, (_, row) in enumerate(stats_df.head(10).iterrows()):
                    f.write(f"{i+1:2d}. {row['feature']}: Cohen's d = {row['cohens_d']:.3f}\n")
        
        print(f"   ✅ Reporte guardado en: {report_path}")
        
        return report

    def predict_single_image(self, image_path: str, save_report: bool = False, save_visualization: bool = False) -> Dict:
        """
        Predecir y analizar completamente una sola imagen
        
        Args:
            image_path: Ruta a la imagen a analizar
            save_report: Si guardar reporte detallado en archivo
            save_visualization: Si guardar visualizaciones de características
            
        Returns:
            Dict con predicción, probabilidades y análisis completo
        """
        print(f"🔍 ANÁLISIS DE IMAGEN INDIVIDUAL")
        print(f"Imagen: {image_path}")
        print("="*60)
        
        # Verificar que existe modelo entrenado
        model_path = f"{self.config['output_dir']}/hologram_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modelo no encontrado en {model_path}. Ejecuta primero el entrenamiento completo.")
        
        # Cargar modelo
        print("📥 Cargando modelo entrenado...")
        model = joblib.load(model_path)
        
        # Cargar estadísticas de entrenamiento si existen
        training_stats = None
        if hasattr(self, 'results') and 'interpretation' in self.results:
            training_stats = self.results['interpretation'].get('feature_statistics')
        
        # 1. Cargar y preprocesar imagen
        print("🖼️  Cargando imagen...")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Imagen no encontrada: {image_path}")
        
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"❌ No se pudo cargar la imagen: {image_path}")
        
        # Redimensionar según configuración
        target_size = tuple(self.config['model']['target_size'])
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        print(f"   Tamaño original: {img.shape}")
        print(f"   Tamaño procesado: {img_normalized.shape}")
        
        # 2. Extraer características
        print("🔬 Extrayendo características...")
        img_array = np.array([img_normalized])  # Agregar dimensión batch
        features = self._extract_features(img_array)[0]  # Obtener features del primer elemento
        
        print(f"   Características extraídas: {len(features)}")
        
        # 3. Hacer predicción
        print("🤖 Realizando predicción...")
        
        # Reshape para predicción individual
        features_reshaped = features.reshape(1, -1)
        
        # Predicción
        prediction = model.predict(features_reshaped)[0]
        probabilities = model.predict_proba(features_reshaped)[0]
        confidence = np.max(probabilities)
        
        # Mapear predicción a etiqueta
        class_names = ['Healthy', 'SCD']
        predicted_class = class_names[prediction]
        
        print(f"   Predicción: {predicted_class}")
        print(f"   Confianza: {confidence:.3f}")
        print(f"   Probabilidades: Healthy={probabilities[0]:.3f}, SCD={probabilities[1]:.3f}")
        
        # 4. Análisis detallado de características
        print("📊 Analizando características...")
        
        # Características seleccionadas por el modelo
        if hasattr(model.named_steps['selector'], 'get_support'):
            selected_mask = model.named_steps['selector'].get_support()
            selected_features = features[selected_mask]
            selected_feature_names = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        else:
            selected_features = features
            selected_feature_names = self.feature_names
        
        # Crear DataFrame de características
        feature_analysis = []
        for i, (feat_name, feat_value) in enumerate(zip(selected_feature_names, selected_features)):
            analysis_item = {
                'feature': feat_name,
                'value': float(feat_value),
                'rank': i + 1
            }
            
            # Si tenemos estadísticas de entrenamiento, agregar comparaciones
            if training_stats is not None:
                matching_stat = training_stats[training_stats['feature'] == feat_name]
                if len(matching_stat) > 0:
                    stat_row = matching_stat.iloc[0]
                    analysis_item.update({
                        'healthy_mean': stat_row['healthy_mean'],
                        'scd_mean': stat_row['scd_mean'],
                        'cohens_d': stat_row['cohens_d'],
                        'discriminative_power': abs(stat_row['cohens_d'])
                    })
                    
                    # Determinar hacia qué clase tiende este valor
                    if feat_value > stat_row['healthy_mean'] and feat_value > stat_row['scd_mean']:
                        tendency = 'High' if stat_row['cohens_d'] > 0 else 'High (favors Healthy)'
                    elif feat_value < stat_row['healthy_mean'] and feat_value < stat_row['scd_mean']:
                        tendency = 'Low' if stat_row['cohens_d'] < 0 else 'Low (favors SCD)'
                    else:
                        # Determinar cuál está más cerca
                        dist_healthy = abs(feat_value - stat_row['healthy_mean'])
                        dist_scd = abs(feat_value - stat_row['scd_mean'])
                        tendency = 'Closer to Healthy' if dist_healthy < dist_scd else 'Closer to SCD'
                    
                    analysis_item['tendency'] = tendency
            
            feature_analysis.append(analysis_item)
        
        # Ordenar por poder discriminativo si tenemos las estadísticas
        if training_stats is not None:
            feature_analysis.sort(key=lambda x: x.get('discriminative_power', 0), reverse=True)
        
        # 5. Crear resultado completo
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            'metadata': {
                'image_path': image_path,
                'analysis_timestamp': timestamp,
                'model_used': model_path,
                'image_size_original': img.shape,
                'image_size_processed': img_normalized.shape
            },
            'prediction': {
                'class': predicted_class,
                'class_index': int(prediction),
                'confidence': float(confidence),
                'probabilities': {
                    'Healthy': float(probabilities[0]),
                    'SCD': float(probabilities[1])
                }
            },
            'features': {
                'total_extracted': len(features),
                'total_selected': len(selected_features),
                'raw_features': features.tolist(),
                'selected_features': selected_features.tolist(),
                'feature_names': self.feature_names,
                'selected_feature_names': selected_feature_names
            },
            'analysis': {
                'top_discriminative_features': feature_analysis[:10],
                'all_features': feature_analysis
            }
        }
        
        # 6. Mostrar resumen de características más importantes
        print("\n📈 TOP 5 CARACTERÍSTICAS MÁS DISCRIMINATIVAS:")
        for i, feat in enumerate(feature_analysis[:5]):
            print(f"   {i+1}. {feat['feature']}: {feat['value']:.4f}")
            if 'tendency' in feat:
                print(f"      → {feat['tendency']} (Cohen's d: {feat.get('cohens_d', 'N/A'):.3f})")
        
        # 7. Guardar reporte si se solicita
        if save_report:
            report_path = f"{self.config['output_dir']}/single_image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            self._save_single_image_report(result, report_path)
            result['report_path'] = report_path
        
        # 8. Crear visualización si se solicita
        if save_visualization:
            viz_path = self._create_single_image_visualization(img_normalized, result)
            result['visualization_path'] = viz_path
        
        print(f"\n✅ Análisis completado exitosamente")
        print(f"   Predicción final: {predicted_class} (confianza: {confidence:.1%})")
        
        return result
    
    def _save_single_image_report(self, result: Dict, report_path: str):
        """Guardar reporte detallado de análisis individual"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ANÁLISIS INDIVIDUAL DE IMAGEN - HOLOGRAM CLASSIFIER V2.0\n")
            f.write("="*80 + "\n")
            f.write(f"Fecha: {result['metadata']['analysis_timestamp']}\n")
            f.write(f"Imagen: {result['metadata']['image_path']}\n\n")
            
            # Predicción
            f.write("PREDICCIÓN\n")
            f.write("-" * 40 + "\n")
            f.write(f"Clase predicha: {result['prediction']['class']}\n")
            f.write(f"Confianza: {result['prediction']['confidence']:.1%}\n")
            f.write(f"Probabilidad Healthy: {result['prediction']['probabilities']['Healthy']:.3f}\n")
            f.write(f"Probabilidad SCD: {result['prediction']['probabilities']['SCD']:.3f}\n\n")
            
            # Características
            f.write("ANÁLISIS DE CARACTERÍSTICAS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Características totales extraídas: {result['features']['total_extracted']}\n")
            f.write(f"Características seleccionadas por modelo: {result['features']['total_selected']}\n\n")
            
            # Top características
            f.write("TOP 15 CARACTERÍSTICAS MÁS DISCRIMINATIVAS\n")
            f.write("-" * 40 + "\n")
            for i, feat in enumerate(result['analysis']['top_discriminative_features'][:15]):
                f.write(f"{i+1:2d}. {feat['feature']}: {feat['value']:.4f}")
                if 'cohens_d' in feat:
                    f.write(f" (Cohen's d: {feat['cohens_d']:.3f})")
                if 'tendency' in feat:
                    f.write(f" - {feat['tendency']}")
                f.write("\n")
        
        print(f"   📄 Reporte detallado guardado en: {report_path}")
    
    def _create_single_image_visualization(self, image: np.ndarray, result: Dict) -> str:
        """Crear visualización para análisis individual"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Imagen original
        ax1.imshow(image)
        ax1.set_title(f'Imagen Analizada\nPredicción: {result["prediction"]["class"]} ({result["prediction"]["confidence"]:.1%})')
        ax1.axis('off')
        
        # 2. Probabilidades
        classes = ['Healthy', 'SCD']
        probs = [result['prediction']['probabilities']['Healthy'], result['prediction']['probabilities']['SCD']]
        colors = ['lightblue' if result['prediction']['class'] == 'Healthy' else 'lightgray',
                  'orange' if result['prediction']['class'] == 'SCD' else 'lightgray']
        bars = ax2.bar(classes, probs, color=colors)
        ax2.set_title('Probabilidades de Clase')
        ax2.set_ylabel('Probabilidad')
        ax2.set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, prob in zip(bars, probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # 3. Top características por valor (siempre disponible)
        top_features = result['analysis']['top_discriminative_features'][:8]
        if top_features:
            # Si tenemos Cohen's d, usarlo; si no, mostrar por valores más altos
            if 'cohens_d' in top_features[0]:
                feature_names = [f['feature'][:15] + '...' if len(f['feature']) > 15 else f['feature'] for f in top_features]
                cohens_d_values = [f['cohens_d'] for f in top_features]
                
                colors_feat = ['red' if d < 0 else 'blue' for d in cohens_d_values]
                bars = ax3.barh(range(len(feature_names)), [abs(d) for d in cohens_d_values], color=colors_feat)
                ax3.set_xlabel('|Cohen\'s d| (Poder Discriminativo)')
                ax3.set_title('Top Características Discriminativas')
            else:
                # Mostrar características por valor absoluto más alto
                feature_names = [f['feature'][:15] + '...' if len(f['feature']) > 15 else f['feature'] for f in top_features]
                feature_values = [abs(f['value']) for f in top_features]
                
                bars = ax3.barh(range(len(feature_names)), feature_values, color='purple', alpha=0.7)
                ax3.set_xlabel('|Valor de Característica|')
                ax3.set_title('Características con Valores Más Altos')
            
            ax3.set_yticks(range(len(feature_names)))
            ax3.set_yticklabels(feature_names)
            ax3.invert_yaxis()
        else:
            # Fallback: mostrar características básicas del modelo
            selected_names = result['features']['selected_feature_names'][:8]
            selected_values = result['features']['selected_features'][:8]
            
            feature_names_short = [name[:15] + '...' if len(name) > 15 else name for name in selected_names]
            bars = ax3.barh(range(len(feature_names_short)), [abs(v) for v in selected_values], 
                           color='gray', alpha=0.7)
            ax3.set_yticks(range(len(feature_names_short)))
            ax3.set_yticklabels(feature_names_short)
            ax3.set_xlabel('|Valor|')
            ax3.set_title('Características Seleccionadas')
            ax3.invert_yaxis()
        
        # 4. Información del análisis (siempre útil)
        # Mostrar distribución de tipos de características
        feature_types = {}
        for feat_name in result['features']['selected_feature_names']:
            if feat_name.startswith('lbp_'):
                feature_types['LBP'] = feature_types.get('LBP', 0) + 1
            elif feat_name.startswith('glcm_'):
                feature_types['GLCM'] = feature_types.get('GLCM', 0) + 1
            elif feat_name.startswith('fft_'):
                feature_types['FFT'] = feature_types.get('FFT', 0) + 1
            elif feat_name.startswith('hu_'):
                feature_types['Hu Moments'] = feature_types.get('Hu Moments', 0) + 1
            elif feat_name.startswith('gabor_'):
                feature_types['Gabor'] = feature_types.get('Gabor', 0) + 1
            elif feat_name.startswith('wavelet_'):
                feature_types['Wavelet'] = feature_types.get('Wavelet', 0) + 1
            else:
                feature_types['Morfológicas'] = feature_types.get('Morfológicas', 0) + 1
        
        if feature_types:
            types = list(feature_types.keys())
            counts = list(feature_types.values())
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(types)))
            
            wedges, texts, autotexts = ax4.pie(counts, labels=types, autopct='%1.1f%%', 
                                             colors=colors_pie, startangle=90)
            ax4.set_title('Distribución de Tipos de\nCaracterísticas Seleccionadas')
            
            # Mejorar legibilidad
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            # Fallback: mostrar estadísticas básicas
            stats_data = {
                'Total Extraídas': result['features']['total_extracted'],
                'Seleccionadas': result['features']['total_selected'],
                'Confianza %': int(result['prediction']['confidence'] * 100)
            }
            
            bars = ax4.bar(stats_data.keys(), stats_data.values(), 
                          color=['lightcoral', 'lightgreen', 'gold'])
            ax4.set_title('Estadísticas del Análisis')
            ax4.set_ylabel('Cantidad / Porcentaje')
            
            # Añadir valores
            for bar, value in zip(bars, stats_data.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(value), ha='center', va='bottom')
        
        plt.suptitle(f'Análisis Completo: {os.path.basename(result["metadata"]["image_path"])}', fontsize=16)
        plt.tight_layout()
        
        # Guardar visualización
        viz_path = f"{self.config['output_dir']}/single_image_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   🎨 Visualización guardada en: {viz_path}")
        return viz_path

    def save_model(self, filename: str = None):
        """Guardar modelo entrenado"""
        if 'model' not in self.results.get('training', {}):
            print("❌ No hay modelo para guardar")
            return
        
        if filename is None:
            filename = f"{self.config['output_dir']}/hologram_model.pkl"
        
        joblib.dump(self.results['training']['model'], filename)
        print(f"✅ Modelo guardado en: {filename}")

def main():
    """Función principal - Punto de entrada único"""
    
    # Configurar paths
    config_path = "config.yaml"
    
    print("🎯 HOLOGRAM CLASSIFIER V2.0 - TRABAJO DE GRADO")
    print("   Ejecución end-to-end con un solo comando")
    print("   Versión simplificada y práctica\n")
    
    try:
        # Crear analizador
        analyzer = HologramAnalyzer(config_path)
        
        # Ejecutar análisis completo
        results = analyzer.run_complete_analysis()
        
        # Guardar modelo
        analyzer.save_model()
        
        # Resumen final
        print(f"\n🎉 ANÁLISIS COMPLETADO CON ÉXITO")
        print(f"📊 Performance Summary:")
        
        if 'training' in results:
            train = results['training']
            print(f"   • Test Accuracy: {train['test_score']:.1%}")
            print(f"   • AUC Score: {train['auc_score']:.3f}")
        
        if 'validation' in results and 'cv_scores' in results['validation']:
            cv = results['validation']['cv_scores']
            print(f"   • CV Accuracy: {cv['accuracy_mean']:.1%} ± {cv['accuracy_std']:.1%}")
        
        if 'validation' in results and 'loocv' in results['validation']:
            loo = results['validation']['loocv']
            print(f"   • LOOCV Accuracy: {loo['accuracy']:.1%}")
        
        print(f"\n📁 Todos los resultados disponibles en: ./results/")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR EN LA EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()