#!/usr/bin/env python3
"""
🔬 Hologram Classifier CLI v2.0
Sistema de línea de comandos profesional para análisis de hologramas
"""

import argparse
import sys
import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Imports del sistema
from src.hologram_analysis import HologramAnalyzer

class HologramCLI:
    """CLI profesional para Hologram Classifier"""
    
    def __init__(self):
        self.config_path = "config.yaml"
        self.version = "2.0"
        
    def print_banner(self):
        """Banner del CLI"""
        print(f"""
🔬======================================================================🔬
   HOLOGRAM CLASSIFIER CLI v{self.version}
   Sistema de Análisis de Hologramas para Anemia Falciforme
   Trabajo de Grado
🔬======================================================================🔬
""")

    def load_config(self) -> Dict[str, Any]:
        """Cargar configuración actual"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"❌ No se encontró {self.config_path}")
            return {}

    def save_config(self, config: Dict[str, Any]):
        """Guardar configuración"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            print(f"✅ Configuración guardada en {self.config_path}")
        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")

    def cmd_analyze(self, args):
        """Comando principal de análisis"""
        self.print_banner()
        
        # Cargar configuración base
        config = self.load_config()
        if not config:
            print("❌ No se pudo cargar la configuración")
            return 1
        
        # Aplicar argumentos de línea de comandos
        if args.mode:
            config.setdefault('execution', {})['mode'] = args.mode
            
        if args.features:
            config.setdefault('model', {})['top_k_features'] = args.features
            
        if args.no_cache:
            config.setdefault('execution', {})['cache_features'] = False
            
        if args.no_optimize:
            config.setdefault('model', {})['auto_optimize'] = False
            
        if args.no_progress:
            config.setdefault('execution', {})['progress_bar'] = False
            
        if args.output:
            config['output_dir'] = args.output

        # Mostrar configuración si se solicita
        if args.show_config:
            print("📋 CONFIGURACIÓN ACTUAL:")
            print(f"   Modo: {config.get('execution', {}).get('mode', 'full')}")
            print(f"   Características: {config.get('model', {}).get('top_k_features', 35)}")
            print(f"   Cache: {config.get('execution', {}).get('cache_features', True)}")
            print(f"   Optimización: {config.get('model', {}).get('auto_optimize', True)}")
            print(f"   Directorio de salida: {config.get('output_dir', './results')}")
            print()

        # Ejecutar análisis
        try:
            analyzer = HologramAnalyzer(config_dict=config)
            results = analyzer.run_complete_analysis()
            
            # Mostrar resumen final
            training_results = results.get('training', {})
            validation_results = results.get('validation', {})
            
            print("\n ANÁLISIS COMPLETADO EXITOSAMENTE")
            print(f"    Test Accuracy: {training_results.get('test_score', 0)*100:.1f}%")
            print(f"    AUC Score: {training_results.get('auc_score', 0):.3f}")
            
            if 'loocv' in validation_results:
                loocv_acc = validation_results['loocv'].get('accuracy', 0) * 100
                loocv_errors = validation_results['loocv'].get('errors', 0)
                print(f"   🔄 LOOCV Accuracy: {loocv_acc:.1f}% ({loocv_errors} errores)")
            
            return 0
            
        except Exception as e:
            print(f" Error durante el análisis: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    def cmd_cache(self, args):
        """Gestión del cache"""
        cache_dir = Path("./results/cache")
        
        if args.clear:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(" Cache limpiado correctamente")
            else:
                print(" No hay cache que limpiar")
                
        elif args.info:
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.pkl"))
                total_size = sum(f.stat().st_size for f in cache_files)
                print(f" INFORMACIÓN DEL CACHE:")
                print(f"   Archivos: {len(cache_files)}")
                print(f"   Tamaño total: {total_size/1024/1024:.1f} MB")
                print(f"   Ubicación: {cache_dir}")
                
                for cache_file in cache_files:
                    size_mb = cache_file.stat().st_size / 1024 / 1024
                    mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    print(f"   - {cache_file.name}: {size_mb:.1f} MB ({mod_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                print("  No hay cache disponible")
        else:
            self.cmd_cache_help()
            
        return 0

    def cmd_info(self, args):
        """Información del sistema"""
        if args.dataset:
            self._info_dataset()
        elif args.model:
            self._info_model()
        elif args.config:
            self._info_config()
        elif args.anomaly:
            self._info_anomaly()
        elif args.system:
            self._info_system()
        else:
            # Mostrar toda la información
            self._info_system()
            self._info_dataset()
            self._info_config()
            self._info_anomaly()
            
        return 0

    def _info_dataset(self):
        """Información del dataset"""
        print(" INFORMACIÓN DEL DATASET:")
        
        data_dir = Path("./data")
        if data_dir.exists():
            # Dataset 1
            ds1_path = data_dir / "dataset_1"
            if ds1_path.exists():
                healthy_1 = len(list((ds1_path / "Healthy").glob("*.png"))) if (ds1_path / "Healthy").exists() else 0
                scd_1 = len(list((ds1_path / "SCD").glob("*.png"))) if (ds1_path / "SCD").exists() else 0
                print(f"   Dataset 1: {healthy_1} Healthy, {scd_1} SCD")
            
            # Dataset 2
            ds2_path = data_dir / "dataset_2"
            if ds2_path.exists():
                healthy_2 = len(list((ds2_path / "H-RBC").glob("*.png"))) if (ds2_path / "H-RBC").exists() else 0
                scd_2 = len(list((ds2_path / "SCD-RBC").glob("*.png"))) if (ds2_path / "SCD-RBC").exists() else 0
                print(f"   Dataset 2: {healthy_2} H-RBC, {scd_2} SCD-RBC")
            
            # Dataset unificado
            unified_path = data_dir / "unified"
            if unified_path.exists():
                healthy_u = len(list((unified_path / "Healthy").glob("*.png"))) if (unified_path / "Healthy").exists() else 0
                scd_u = len(list((unified_path / "SCD").glob("*.png"))) if (unified_path / "SCD").exists() else 0
                print(f"   Dataset Unificado: {healthy_u} Healthy, {scd_u} SCD")
                print(f"   Total: {healthy_u + scd_u} imágenes")
        else:
            print("    No se encontró el directorio de datos")

    def _info_model(self):
        """Información del modelo"""
        print(" INFORMACIÓN DEL MODELO:")
        
        model_path = Path("./results/hologram_model.pkl")
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            print(f"   Archivo: {model_path}")
            print(f"   Tamaño: {size_mb:.1f} MB")
            print(f"   Última actualización: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Intentar cargar información del modelo
            try:
                import joblib
                model = joblib.load(model_path)
                if hasattr(model, 'named_steps'):
                    print(f"   Tipo: {type(model).__name__}")
                    steps = list(model.named_steps.keys())
                    print(f"   Pipeline: {' → '.join(steps)}")
                    
                    if 'selector' in model.named_steps:
                        selector = model.named_steps['selector']
                        n_features = getattr(selector, 'k', 'N/A')
                        print(f"   Características seleccionadas: {n_features}")
            except Exception as e:
                print(f"    No se pudo cargar detalles del modelo: {e}")
        else:
            print("    No se encontró modelo entrenado")
        
        # Información del detector de anomalías
        anomaly_path = Path("./results/anomaly_detector.pkl")
        if anomaly_path.exists():
            size_mb = anomaly_path.stat().st_size / 1024 / 1024
            mod_time = datetime.fromtimestamp(anomaly_path.stat().st_mtime)
            print(f"\n🔍 DETECTOR DE ANOMALÍAS:")
            print(f"   Archivo: {anomaly_path}")
            print(f"   Tamaño: {size_mb:.1f} MB") 
            print(f"   Última actualización: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                from src.anomaly_detector import DistanceBasedAnomalyDetector
                detector = DistanceBasedAnomalyDetector.load(anomaly_path)
                stats = detector.get_statistics()
                print(f"   Algoritmo: Distancia de Mahalanobis")
                print(f"   Umbral: {stats.get('threshold', 'N/A'):.3f}")
                print(f"   Características: {stats.get('n_features', 'N/A')}")
                print(f"   Percentil: {stats.get('threshold_percentile', 'N/A')}%")
            except Exception as e:
                print(f"   No se pudieron cargar detalles: {e}")
        else:
            print(f"\n🔍 DETECTOR DE ANOMALÍAS: No disponible")

    def _info_config(self):
        """Información de configuración"""
        print(" CONFIGURACIÓN ACTUAL:")
        
        config = self.load_config()
        if config:
            # Información principal
            mode = config.get('execution', {}).get('mode', 'full')
            features = config.get('model', {}).get('top_k_features', 35)
            cache = config.get('execution', {}).get('cache_features', True)
            optimize = config.get('model', {}).get('auto_optimize', True)
            
            print(f"   Modo de ejecución: {mode}")
            print(f"   Características máximas: {features}")
            print(f"   Cache habilitado: {'Sí' if cache else 'No'}")
            print(f"   Optimización automática: {'Sí' if optimize else 'No'}")
            print(f"   Directorio de salida: {config.get('output_dir', './results')}")
            
            # Configuración de características
            lbp_points = config.get('features', {}).get('lbp', {}).get('points', 24)
            fft_rings = config.get('features', {}).get('fft', {}).get('n_rings', 3)
            print(f"   LBP points: {lbp_points}")
            print(f"   FFT rings: {fft_rings}")
        else:
            print("    No se pudo cargar la configuración")

    def _info_system(self):
        """Información del sistema"""
        print(f"🔬 HOLOGRAM CLASSIFIER v{self.version}")
        print(f"   Directorio actual: {os.getcwd()}")
        print(f"   Python: {sys.version.split()[0]}")
        
        # Verificar dependencias
        required_packages = ['numpy', 'pandas', 'scikit-learn', 'opencv-python', 'matplotlib', 'pywavelets']
        missing = []
        
        for pkg in required_packages:
            try:
                __import__(pkg.replace('-', '_'))
            except ImportError:
                missing.append(pkg)
        
        if missing:
            print(f"    Dependencias faltantes: {', '.join(missing)}")
        else:
            print("    Todas las dependencias instaladas")
    
    def _info_anomaly(self):
        """Información del detector de anomalías"""
        print(" DETECTOR DE ANOMALÍAS:")
        
        anomaly_path = Path("./results/anomaly_detector.pkl")
        if anomaly_path.exists():
            size_mb = anomaly_path.stat().st_size / 1024 / 1024
            mod_time = datetime.fromtimestamp(anomaly_path.stat().st_mtime)
            print(f"   Estado: ✅ Disponible")
            print(f"   Archivo: {anomaly_path}")
            print(f"   Tamaño: {size_mb:.1f} MB") 
            print(f"   Última actualización: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                from src.anomaly_detector import DistanceBasedAnomalyDetector
                detector = DistanceBasedAnomalyDetector.load(anomaly_path)
                stats = detector.get_statistics()
                print(f"   Algoritmo: Distancia de Mahalanobis")
                print(f"   Umbral: {stats.get('threshold', 'N/A'):.3f}")
                print(f"   Características: {stats.get('n_features', 'N/A')}")
                print(f"   Percentil: {stats.get('threshold_percentile', 'N/A')}%")
                
                if 'class_statistics' in stats:
                    print(f"   Clases entrenadas:")
                    for class_name, class_stats in stats['class_statistics'].items():
                        print(f"     - {class_name}: {class_stats['count']} muestras (dist media: {class_stats['mean_distance']:.3f})")
            except Exception as e:
                print(f"   No se pudieron cargar detalles: {e}")
        else:
            print(f"   Estado: ❌ No disponible")
            print(f"   El detector se creará automáticamente después del entrenamiento")
    
    def cmd_validate(self, args):
        """Validar modelo con datos fuera de dominio"""
        self.print_banner()
        
        print(f"🔍 VALIDACIÓN CON DATOS FUERA DE DOMINIO")
        print("=" * 60)
        
        # Verificar directorio de imágenes
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"❌ Directorio no encontrado: {image_dir}")
            return 1
        
        # Verificar modelo entrenado
        model_path = Path("./results/hologram_model.pkl")
        if not model_path.exists():
            print("❌ No se encontró modelo entrenado. Ejecuta primero: python cli.py analyze")
            return 1
        
        output_dir = Path(args.output) if args.output else Path("./results")
        output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Directorio de imágenes: {image_dir}")
        print(f"📄 Directorio de resultados: {output_dir}")
        print(f"🔢 Límite por categoría: {args.limit}")
        print(f"🚨 Detección de anomalías: {'Deshabilitada' if args.no_anomaly else 'Habilitada'}")
        
        try:
            # Crear validador
            from src.hologram_analysis import HologramAnalyzer
            
            analyzer = HologramAnalyzer("config.yaml")
            
            # Si se deshabilita anomalías, remover detector
            if args.no_anomaly:
                analyzer.anomaly_detector = None
                print("⚠️  Detección de anomalías deshabilitada")
            
            # Buscar subcarpetas o imágenes
            results = []
            categories_found = []
            
            # Buscar subcarpetas
            for subdir in image_dir.iterdir():
                if subdir.is_dir():
                    image_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
                    if image_files:
                        categories_found.append((subdir.name, image_files[:args.limit]))
            
            # Si no hay subcarpetas, usar el directorio raíz
            if not categories_found:
                image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
                if image_files:
                    categories_found.append(("images", image_files[:args.limit]))
            
            if not categories_found:
                print("❌ No se encontraron imágenes .jpg o .png")
                return 1
            
            print(f"\n📊 Categorías encontradas: {len(categories_found)}")
            for cat_name, images in categories_found:
                print(f"   - {cat_name}: {len(images)} imágenes")
            
            # Procesar cada imagen
            print(f"\n🔬 PROCESANDO IMÁGENES...")
            total_processed = 0
            anomalies_detected = 0
            
            for category_name, image_files in categories_found:
                print(f"\n📂 Procesando {category_name}...")
                
                for img_path in image_files:
                    try:
                        result = analyzer.predict_single_image(
                            str(img_path),
                            save_report=False,
                            save_visualization=False
                        )
                        
                        prediction = result['prediction']['class']
                        confidence = result['prediction']['confidence']
                        
                        # Información de anomalías
                        anomaly_info = result.get('anomaly_detection', {})
                        is_anomaly = anomaly_info.get('is_anomaly', False)
                        anomaly_score = anomaly_info.get('anomaly_score', 0.0)
                        
                        if is_anomaly:
                            anomalies_detected += 1
                        
                        # Extraer información de anomalías detallada
                        anomaly_info = result.get('anomaly_detection', {})
                        mahalanobis_distance = anomaly_info.get('mahalanobis_distance', None)
                        threshold = anomaly_info.get('threshold', None)
                        p_value = anomaly_info.get('p_value', None)
                        
                        # Guardar resultado
                        results.append({
                            'category': category_name,
                            'image': img_path.name,
                            'prediction': prediction,
                            'confidence': confidence,
                            'is_anomaly': is_anomaly,
                            'anomaly_score': anomaly_score,
                            'mahalanobis_distance': mahalanobis_distance,
                            'threshold': threshold,
                            'p_value': p_value,
                            'path': str(img_path)
                        })
                        
                        # Mostrar resultado resumido
                        anomaly_flag = "🚨" if is_anomaly else "✅"
                        print(f"   {img_path.name}: {prediction} ({confidence:.3f}) {anomaly_flag}")
                        
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"   ❌ Error procesando {img_path.name}: {e}")
            
            # Resumen final
            print(f"\n📈 RESUMEN DE VALIDACIÓN")
            print("=" * 40)
            print(f"📊 Total procesadas: {total_processed}")
            print(f"🚨 Anomalías detectadas: {anomalies_detected} ({anomalies_detected/total_processed*100:.1f}%)")
            
            # Resumen por predicción
            if results:
                import pandas as pd
                df = pd.DataFrame(results)
                
                pred_counts = df['prediction'].value_counts()
                print(f"\n🎯 Distribución de predicciones:")
                for pred, count in pred_counts.items():
                    print(f"   - {pred}: {count} ({count/len(df)*100:.1f}%)")
                
                # Resumen por categoría
                if len(categories_found) > 1:
                    print(f"\n📂 Resultados por categoría:")
                    for category in df['category'].unique():
                        cat_df = df[df['category'] == category]
                        cat_anomalies = cat_df['is_anomaly'].sum() if 'is_anomaly' in cat_df.columns else 0
                        print(f"   {category}: {len(cat_df)} imágenes, {cat_anomalies} anomalías ({cat_anomalies/len(cat_df)*100:.1f}%)")
            
            # Guardar reporte si se solicita
            if args.save_report and results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = output_dir / f"validation_report_{timestamp}.csv"
                
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(report_path, index=False)
                print(f"\n💾 Reporte guardado en: {report_path}")
                
                # Generar visualizaciones de anomalías
                print(f"\n📊 Generando visualizaciones de anomalías...")
                self._generate_anomaly_visualizations(results, output_dir, timestamp)
            
            return 0
            
        except Exception as e:
            print(f"❌ Error durante la validación: {e}")
            return 1

    def _generate_anomaly_visualizations(self, results, output_dir, timestamp):
        """Generar visualizaciones avanzadas para análisis de anomalías"""
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        try:
            df = pd.DataFrame(results)
            
            # Configurar estilo
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Distribución de Distancias Mahalanobis
            if 'mahalanobis_distance' in df.columns:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Análisis de Detección de Anomalías', fontsize=16, fontweight='bold')
                
                # Histograma de distancias
                ax1 = axes[0, 0]
                ax1.hist(df['mahalanobis_distance'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                if 'threshold' in df.columns and not df['threshold'].empty:
                    threshold = df['threshold'].iloc[0]
                    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral: {threshold:.2f}')
                    ax1.legend()
                ax1.set_xlabel('Distancia de Mahalanobis')
                ax1.set_ylabel('Frecuencia')
                ax1.set_title('Distribución de Distancias de Mahalanobis')
                ax1.grid(True, alpha=0.3)
                
                # Box plot por categoría
                ax2 = axes[0, 1]
                if 'category' in df.columns:
                    df.boxplot(column='mahalanobis_distance', by='category', ax=ax2)
                    ax2.set_title('Distancias de Mahalanobis por Categoría')
                    ax2.set_xlabel('Categoría')
                    ax2.set_ylabel('Distancia de Mahalanobis')
                    plt.suptitle('')  # Remove automatic title
                
                # Scatter: Confianza vs Score de Anomalía
                ax3 = axes[1, 0]
                if 'confidence' in df.columns and 'anomaly_score' in df.columns:
                    colors = df['category'].astype('category').cat.codes if 'category' in df.columns else 'blue'
                    scatter = ax3.scatter(df['confidence'], df['anomaly_score'], 
                                        c=colors, alpha=0.7, s=50, cmap='tab10')
                    ax3.set_xlabel('Confianza de Predicción')
                    ax3.set_ylabel('Score de Anomalía')
                    ax3.set_title('Confianza vs Score de Anomalía')
                    ax3.grid(True, alpha=0.3)
                    
                    # Colorbar si hay categorías
                    if 'category' in df.columns:
                        categories = df['category'].unique()
                        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=plt.cm.tab10(i/len(categories)), 
                                            markersize=8, label=cat) 
                                 for i, cat in enumerate(categories)]
                        ax3.legend(handles=handles, title='Categorías', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Distribución de scores por categoría
                ax4 = axes[1, 1]
                if 'anomaly_score' in df.columns and 'category' in df.columns:
                    categories = df['category'].unique()
                    for i, cat in enumerate(categories):
                        cat_data = df[df['category'] == cat]['anomaly_score']
                        ax4.hist(cat_data, alpha=0.6, label=cat, bins=10, density=True)
                    ax4.set_xlabel('Score de Anomalía')
                    ax4.set_ylabel('Densidad')
                    ax4.set_title('Distribución de Scores de Anomalía')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                anomaly_path = output_dir / f"anomaly_analysis_{timestamp}.png"
                plt.savefig(anomaly_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Análisis de anomalías: {anomaly_path}")
            
            # 2. Análisis de Características (si están disponibles)
            if any('feature_' in col or col.startswith(('lbp_', 'glcm_', 'gabor_', 'hu_')) for col in df.columns):
                self._generate_feature_analysis(df, output_dir, timestamp)
            
            # 3. Mapa de calor de predicciones
            self._generate_prediction_heatmap(df, output_dir, timestamp)
            
        except Exception as e:
            print(f"   ⚠️ Error generando visualizaciones: {e}")

    def _generate_feature_analysis(self, df, output_dir, timestamp):
        """Generar análisis de características específicas"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        try:
            # Buscar columnas de características
            feature_cols = [col for col in df.columns if 
                           col.startswith(('lbp_', 'glcm_', 'gabor_', 'hu_', 'fft_', 'wavelet_')) 
                           and col in df.columns]
            
            if not feature_cols:
                return
            
            # Seleccionar top características más variables
            feature_data = df[feature_cols]
            variances = feature_data.var().sort_values(ascending=False)
            top_features = variances.head(15).index.tolist()
            
            if len(top_features) < 3:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Análisis de Características para Detección de Anomalías', fontsize=16, fontweight='bold')
            
            # Mapa de calor de correlaciones
            ax1 = axes[0, 0]
            corr_matrix = feature_data[top_features[:10]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax1, fmt='.2f', cbar_kws={'shrink': 0.8})
            ax1.set_title('Correlación entre Top Características')
            
            # Distribución de características por categoría
            ax2 = axes[0, 1]
            if 'category' in df.columns and len(top_features) >= 3:
                for cat in df['category'].unique():
                    cat_data = df[df['category'] == cat][top_features[0]]
                    ax2.hist(cat_data, alpha=0.6, label=cat, bins=15, density=True)
                ax2.set_xlabel(f'Valores de {top_features[0]}')
                ax2.set_ylabel('Densidad')
                ax2.set_title(f'Distribución de {top_features[0]} por Categoría')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Scatter de dos características principales
            ax3 = axes[1, 0]
            if len(top_features) >= 2 and 'category' in df.columns:
                categories = df['category'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                for i, cat in enumerate(categories):
                    cat_data = df[df['category'] == cat]
                    ax3.scatter(cat_data[top_features[0]], cat_data[top_features[1]], 
                              alpha=0.7, label=cat, c=[colors[i]], s=50)
                ax3.set_xlabel(top_features[0])
                ax3.set_ylabel(top_features[1])
                ax3.set_title('Espacio de Características Principales')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Box plot de características más discriminativas
            ax4 = axes[1, 1]
            if 'category' in df.columns and len(top_features) >= 3:
                # Preparar datos para box plot
                plot_data = []
                categories = []
                feature_names = []
                
                for feature in top_features[:5]:  # Top 5 características
                    for cat in df['category'].unique():
                        cat_values = df[df['category'] == cat][feature]
                        plot_data.extend(cat_values.tolist())
                        categories.extend([cat] * len(cat_values))
                        feature_names.extend([feature] * len(cat_values))
                
                plot_df = pd.DataFrame({
                    'value': plot_data,
                    'category': categories,
                    'feature': feature_names
                })
                
                # Crear box plot agrupado
                sns.boxplot(data=plot_df, x='feature', y='value', hue='category', ax=ax4)
                ax4.set_title('Características Discriminativas por Categoría')
                ax4.set_xlabel('Características')
                ax4.set_ylabel('Valores')
                ax4.tick_params(axis='x', rotation=45)
                ax4.legend(title='Categoría')
            
            plt.tight_layout()
            features_path = output_dir / f"feature_analysis_{timestamp}.png"
            plt.savefig(features_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Análisis de características: {features_path}")
            
        except Exception as e:
            print(f"   ⚠️ Error en análisis de características: {e}")

    def _generate_prediction_heatmap(self, df, output_dir, timestamp):
        """Generar mapa de calor de predicciones y anomalías"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        
        try:
            if 'category' not in df.columns or 'prediction' not in df.columns:
                return
            
            # Crear matriz de confusión para anomalías y predicciones
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('Mapas de Calor de Predicciones y Anomalías', fontsize=16, fontweight='bold')
            
            # Mapa de calor: Categoría vs Predicción
            ax1 = axes[0]
            confusion_matrix = pd.crosstab(df['category'], df['prediction'], normalize='index') * 100
            sns.heatmap(confusion_matrix, annot=True, fmt='.1f', cmap='Blues', ax=ax1, cbar_kws={'label': 'Porcentaje'})
            ax1.set_title('Predicciones por Categoría de Organismo (%)')
            ax1.set_xlabel('Predicción del Modelo')
            ax1.set_ylabel('Categoría Real')
            
            # Mapa de calor: Métricas de anomalía por categoría
            ax2 = axes[1]
            if 'anomaly_score' in df.columns:
                anomaly_stats = df.groupby('category').agg({
                    'anomaly_score': ['mean', 'std'],
                    'mahalanobis_distance': ['mean', 'std'] if 'mahalanobis_distance' in df.columns else ['mean'],
                    'confidence': ['mean', 'std'] if 'confidence' in df.columns else ['mean']
                }).round(3)
                
                # Flatten column names
                anomaly_stats.columns = ['_'.join(col).strip() for col in anomaly_stats.columns]
                
                # Transponer para mejor visualización
                anomaly_stats_T = anomaly_stats.T
                sns.heatmap(anomaly_stats_T, annot=True, fmt='.3f', cmap='Reds', ax=ax2, cbar_kws={'label': 'Valor'})
                ax2.set_title('Métricas de Anomalía por Categoría')
                ax2.set_xlabel('Categoría')
                ax2.set_ylabel('Métricas')
            
            plt.tight_layout()
            heatmap_path = output_dir / f"prediction_heatmap_{timestamp}.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Mapas de calor: {heatmap_path}")
            
        except Exception as e:
            print(f"   ⚠️ Error generando mapas de calor: {e}")

    def cmd_config(self, args):
        """Configuración del sistema"""
        config = self.load_config()
        if not config:
            print(" No se pudo cargar la configuración")
            return 1
        
        changed = False
        
        # Aplicar cambios
        if args.mode:
            config.setdefault('execution', {})['mode'] = args.mode
            print(f" Modo cambiado a: {args.mode}")
            changed = True
            
        if args.features:
            config.setdefault('model', {})['top_k_features'] = args.features
            print(f" Características máximas: {args.features}")
            changed = True
            
        if args.cache is not None:
            config.setdefault('execution', {})['cache_features'] = args.cache
            print(f" Cache: {'habilitado' if args.cache else 'deshabilitado'}")
            changed = True
            
        if args.optimize is not None:
            config.setdefault('model', {})['auto_optimize'] = args.optimize
            print(f" Optimización automática: {'habilitada' if args.optimize else 'deshabilitada'}")
            changed = True
            
        if args.progress is not None:
            config.setdefault('execution', {})['progress_bar'] = args.progress
            print(f" Barras de progreso: {'habilitadas' if args.progress else 'deshabilitadas'}")
            changed = True
            
        if args.output:
            config['output_dir'] = args.output
            print(f" Directorio de salida: {args.output}")
            changed = True
        
        # Guardar si hubo cambios
        if changed:
            self.save_config(config)
        else:
            # Mostrar configuración actual
            self._info_config()
            
        return 0

    def cmd_cache_help(self):
        """Ayuda del comando cache"""
        print(" GESTIÓN DEL CACHE:")
        print("   python cli.py cache --clear    # Limpiar cache")
        print("   python cli.py cache --info     # Información del cache")

def create_parser():
    """Crear el parser de argumentos"""
    parser = argparse.ArgumentParser(
        prog='cli.py',
        description=' Hologram Classifier CLI v2.0 - Sistema  de análisis de hologramas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  Análisis básico:
    python cli.py analyze
    
  Análisis rápido:
    python cli.py analyze --mode quick
    
  Análisis personalizado:
    python cli.py analyze --features 40 --no-cache --output ./mi_analisis
    
  Gestión del cache:
    python cli.py cache --info
    python cli.py cache --clear
    
  Información del sistema:
    python cli.py info --dataset
    python cli.py info --model
    
  Configuración:
    python cli.py config --mode deep --features 50 --cache
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando analyze
    analyze_parser = subparsers.add_parser('analyze', help='Ejecutar análisis de hologramas')
    analyze_parser.add_argument('--mode', choices=['quick', 'full', 'deep'], help='Modo de análisis')
    analyze_parser.add_argument('--features', type=int, help='Número máximo de características')
    analyze_parser.add_argument('--no-cache', action='store_true', help='Deshabilitar cache')
    analyze_parser.add_argument('--no-optimize', action='store_true', help='Deshabilitar optimización automática')
    analyze_parser.add_argument('--no-progress', action='store_true', help='Deshabilitar barras de progreso')
    analyze_parser.add_argument('--output', help='Directorio de salida personalizado')
    analyze_parser.add_argument('--show-config', action='store_true', help='Mostrar configuración antes del análisis')
    analyze_parser.add_argument('--debug', action='store_true', help='Modo debug con trazas completas')
    
    # Comando cache
    cache_parser = subparsers.add_parser('cache', help='Gestión del cache')
    cache_group = cache_parser.add_mutually_exclusive_group(required=True)
    cache_group.add_argument('--clear', action='store_true', help='Limpiar cache')
    cache_group.add_argument('--info', action='store_true', help='Información del cache')
    
    # Comando info
    info_parser = subparsers.add_parser('info', help='Información del sistema')
    info_parser.add_argument('--dataset', action='store_true', help='Información del dataset')
    info_parser.add_argument('--model', action='store_true', help='Información del modelo')
    info_parser.add_argument('--config', action='store_true', help='Información de configuración')
    info_parser.add_argument('--anomaly', action='store_true', help='Información del detector de anomalías')
    info_parser.add_argument('--system', action='store_true', help='Información del sistema')
    
    # Comando config
    config_parser = subparsers.add_parser('config', help='Configurar el sistema')
    config_parser.add_argument('--mode', choices=['quick', 'full', 'deep'], help='Cambiar modo de análisis')
    config_parser.add_argument('--features', type=int, help='Número máximo de características')
    config_parser.add_argument('--cache', type=bool, help='Habilitar/deshabilitar cache')
    config_parser.add_argument('--optimize', type=bool, help='Habilitar/deshabilitar optimización')
    config_parser.add_argument('--progress', type=bool, help='Habilitar/deshabilitar barras de progreso')
    config_parser.add_argument('--output', help='Directorio de salida')
    
    # Comando validate
    validate_parser = subparsers.add_parser('validate', help='Validar modelo con datos fuera de dominio')
    validate_parser.add_argument('--image-dir', required=True, help='Directorio con imágenes a validar')
    validate_parser.add_argument('--output', help='Directorio de salida para resultados')
    validate_parser.add_argument('--limit', type=int, default=10, help='Número máximo de imágenes por categoría')
    validate_parser.add_argument('--save-report', action='store_true', help='Guardar reporte detallado')
    validate_parser.add_argument('--no-anomaly', action='store_true', help='Deshabilitar detección de anomalías')
    
    return parser

def main():
    """Función principal del CLI"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = HologramCLI()
    
    try:
        if args.command == 'analyze':
            return cli.cmd_analyze(args)
        elif args.command == 'cache':
            return cli.cmd_cache(args)
        elif args.command == 'info':
            return cli.cmd_info(args)
        elif args.command == 'config':
            return cli.cmd_config(args)
        elif args.command == 'validate':
            return cli.cmd_validate(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\n  Operación cancelada por el usuario")
        return 1
    except Exception as e:
        print(f"\n Error inesperado: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())