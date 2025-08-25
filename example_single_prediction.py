#!/usr/bin/env python3
"""
🔍 EJEMPLO DE PREDICCIÓN DE IMAGEN INDIVIDUAL
Demuestra cómo usar el método predict_single_image del HologramAnalyzer
"""

import sys
import os
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hologram_analysis import HologramAnalyzer

def example_single_prediction():
    """
    Ejemplo de cómo usar predict_single_image para analizar una imagen individual
    """
    
    print("🔍 EJEMPLO DE PREDICCIÓN INDIVIDUAL")
    print("="*60)
    
    # 1. Crear analizador con configuración existente
    print("1️⃣ Inicializando HologramAnalyzer...")
    analyzer = HologramAnalyzer("config.yaml")
    
    # 2. Ejemplo con imagen de células sanas
    healthy_image = "./data/dataset_1/Healthy/20250707_220626.png"
    
    if os.path.exists(healthy_image):
        print(f"\n2️⃣ Analizando imagen HEALTHY: {healthy_image}")
        print("-" * 50)
        
        try:
            # Predicción básica
            result_healthy = analyzer.predict_single_image(healthy_image)
            
            # Mostrar resultados básicos
            pred = result_healthy['prediction']
            print(f"\n📊 RESULTADOS:")
            print(f"   Clase predicha: {pred['class']}")
            print(f"   Confianza: {pred['confidence']:.1%}")
            print(f"   Prob. Healthy: {pred['probabilities']['Healthy']:.3f}")
            print(f"   Prob. SCD: {pred['probabilities']['SCD']:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Tip: Asegúrate de haber ejecutado el entrenamiento primero:")
            print("   python run_analysis.py")
    else:
        print(f"❌ Imagen no encontrada: {healthy_image}")
    
    # 3. Ejemplo con imagen SCD
    scd_image = "./data/dataset_1/SCD/20250321_210836.png"
    
    if os.path.exists(scd_image):
        print(f"\n3️⃣ Analizando imagen SCD: {scd_image}")
        print("-" * 50)
        
        try:
            # Predicción con reporte y visualización
            result_scd = analyzer.predict_single_image(
                scd_image, 
                save_report=True, 
                save_visualization=True
            )
            
            # Mostrar resultados
            pred = result_scd['prediction']
            print(f"\n📊 RESULTADOS:")
            print(f"   Clase predicha: {pred['class']}")
            print(f"   Confianza: {pred['confidence']:.1%}")
            print(f"   Prob. Healthy: {pred['probabilities']['Healthy']:.3f}")
            print(f"   Prob. SCD: {pred['probabilities']['SCD']:.3f}")
            
            # Mostrar archivos generados
            if 'report_path' in result_scd:
                print(f"\n📄 Reporte: {result_scd['report_path']}")
            if 'visualization_path' in result_scd:
                print(f"🎨 Visualización: {result_scd['visualization_path']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ Imagen no encontrada: {scd_image}")

def analyze_custom_image(image_path: str):
    """
    Analizar una imagen personalizada
    
    Args:
        image_path: Ruta a la imagen a analizar
    """
    
    print(f"🔍 ANÁLISIS DE IMAGEN PERSONALIZADA")
    print(f"Imagen: {image_path}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ Imagen no encontrada: {image_path}")
        return
    
    try:
        # Crear analizador
        analyzer = HologramAnalyzer("config.yaml")
        
        # Análisis completo con todos los extras
        result = analyzer.predict_single_image(
            image_path,
            save_report=True,
            save_visualization=True
        )
        
        # Mostrar resultados detallados
        pred = result['prediction']
        meta = result['metadata']
        
        print(f"\n📊 RESULTADOS DETALLADOS:")
        print(f"   🎯 Predicción: {pred['class']}")
        print(f"   📈 Confianza: {pred['confidence']:.1%}")
        print(f"   🟢 Prob. Healthy: {pred['probabilities']['Healthy']:.3f}")
        print(f"   🔴 Prob. SCD: {pred['probabilities']['SCD']:.3f}")
        print(f"   📏 Tamaño original: {meta['image_size_original']}")
        print(f"   📐 Tamaño procesado: {meta['image_size_processed']}")
        
        # Top 3 características discriminativas
        top_features = result['analysis']['top_discriminative_features'][:3]
        print(f"\n🔬 TOP 3 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for i, feat in enumerate(top_features):
            print(f"   {i+1}. {feat['feature']}: {feat['value']:.4f}")
            if 'cohens_d' in feat:
                print(f"      → Cohen's d: {feat['cohens_d']:.3f}")
            if 'tendency' in feat:
                print(f"      → {feat['tendency']}")
        
        # Archivos generados
        print(f"\n📁 ARCHIVOS GENERADOS:")
        if 'report_path' in result:
            print(f"   📄 Reporte detallado: {result['report_path']}")
        if 'visualization_path' in result:
            print(f"   🎨 Visualización: {result['visualization_path']}")
        
        print(f"\n✅ Análisis completado exitosamente!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        print("\n💡 POSIBLES SOLUCIONES:")
        print("   • Ejecutar entrenamiento: python run_analysis.py")
        print("   • Verificar formato de imagen PNG")
        print("   • Revisar que la imagen sea legible")
        return None

def main():
    """
    Función principal con ejemplos
    """
    
    if len(sys.argv) > 1:
        # Analizar imagen personalizada desde argumentos
        image_path = sys.argv[1]
        analyze_custom_image(image_path)
    else:
        # Ejecutar ejemplos predeterminados
        example_single_prediction()
        
        print(f"\n" + "="*60)
        print(" CÓMO USAR ESTE SCRIPT:")
        print("   • Ejemplos predeterminados:")
        print("     python example_single_prediction.py")
        print("   • Imagen personalizada:")
        print("     python example_single_prediction.py path/to/your/image.png")
        print("="*60)

if __name__ == "__main__":
    main()