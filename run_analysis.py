#!/usr/bin/env python3
"""
🎯 HOLOGRAM CLASSIFIER V2.0 - COMANDO ÚNICO
Ejecutar análisis completo con un solo comando
Versión simplificada para trabajo de grado
"""

import sys
import os
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """
    Punto de entrada principal - UN SOLO COMANDO PARA TODO
    """
    
    print("🚀" + "="*70 + "🚀")
    print("   HOLOGRAM CLASSIFIER V2.0 - ANÁLISIS COMPLETO")
    print("   Versión Simplificada para Trabajo de Grado")
    print("   Un solo comando ejecuta todo el pipeline")
    print("🚀" + "="*70 + "🚀")
    
    try:
        # Importar y ejecutar análisis
        from hologram_analysis import HologramAnalyzer
        
        # Verificar que existe el dataset
        config_path = Path(__file__).parent / "config.yaml"
        
        print(f"\n📋 Configuración: {config_path}")
        print(f"📁 Directorio de trabajo: {Path.cwd()}")
        
        # Crear y ejecutar analizador
        analyzer = HologramAnalyzer(str(config_path))
        results = analyzer.run_complete_analysis()
        analyzer.save_model()
        
        # Mostrar resumen final
        print("\n" + "🎉" + "="*68 + "🎉")
        print("   ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("🎉" + "="*68 + "🎉")
        
        if results:
            print(f"\n📊 RESUMEN DE RESULTADOS:")
            
            # Performance del modelo
            if 'training' in results:
                train = results['training']
                print(f"   🎯 Accuracy (Test): {train['test_score']:.1%}")
                print(f"   🎯 AUC Score: {train['auc_score']:.3f}")
                print(f"   🎯 Características Seleccionadas: {len(train['selected_features'])}")
            
            # Validación cruzada
            if 'validation' in results and 'cv_scores' in results['validation']:
                cv = results['validation']['cv_scores']
                print(f"   📈 Cross-Validation AUC: {cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}")
                print(f"   📈 Cross-Validation Accuracy: {cv['accuracy_mean']:.1%} ± {cv['accuracy_std']:.1%}")
            
            # LOOCV
            if 'validation' in results and 'loocv' in results['validation']:
                loo = results['validation']['loocv']
                print(f"   🔄 LOOCV Accuracy: {loo['accuracy']:.1%}")
                print(f"   🔄 LOOCV Errores: {loo['errors']}/{loo['total_samples']}")
            
            # Dataset info
            if 'metadata' in results:
                meta = results['metadata']
                print(f"\n📊 INFORMACIÓN DEL DATASET:")
                print(f"   📁 Total Imágenes: {meta['total_images']}")
                print(f"   🟢 Células Sanas: {meta['healthy_count']}")
                print(f"   🔴 Células SCD: {meta['scd_count']}")
            
            print(f"\n📂 ARCHIVOS GENERADOS:")
            print(f"   📄 Reporte: ./results/analysis_report.txt")
            print(f"   🤖 Modelo: ./results/hologram_model.pkl")
            print(f"   📊 Visualizaciones: ./results/*.png")
            
        else:
            print("⚠️ No se pudieron obtener resultados completos")
        
        print(f"\n✅ Para revisar resultados detallados:")
        print(f"   cat ./results/analysis_report.txt")
        print(f"   ls -la ./results/")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print(f"   Asegúrate de tener todas las dependencias instaladas:")
        print(f"   pip install numpy pandas scikit-learn opencv-python matplotlib seaborn pyyaml scikit-image")
        return 1
        
    except FileNotFoundError as e:
        print(f"❌ Archivo no encontrado: {e}")
        print(f"   Verifica que el dataset esté en la ubicación correcta")
        print(f"   El script busca el dataset en: '../unified_dataset'")
        return 1
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        print("\n🐛 Stack trace completo:")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)