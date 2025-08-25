#!/usr/bin/env python3
"""
🎯 EJEMPLO SIMPLE DE PREDICCIÓN INDIVIDUAL
Código mínimo para usar predict_single_image()
"""

import sys
import os
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Ejemplo super simple
def predict_image(image_path: str):
    """Predicción simple de una imagen"""
    
    # Importar analizador
    from hologram_analysis import HologramAnalyzer
    
    # Crear analizador
    analyzer = HologramAnalyzer("config.yaml")
    
    # Hacer predicción
    result = analyzer.predict_single_image(image_path)
    
    # Extraer resultados básicos
    prediction = result['prediction']['class']
    confidence = result['prediction']['confidence']
    
    print(f"Imagen: {os.path.basename(image_path)}")
    print(f"Predicción: {prediction}")
    print(f"Confianza: {confidence:.1%}")
    
    return prediction, confidence

# Ejemplo de uso
if __name__ == "__main__":
    
    # Ejemplos rápidos
    examples = [
        "./data/dataset_1/Healthy/20250707_220626.png",
        "./data/dataset_1/SCD/20250321_210836.png"
    ]
    
    print("🎯 PREDICCIONES RÁPIDAS")
    print("="*40)
    
    for image_path in examples:
        if os.path.exists(image_path):
            try:
                prediction, confidence = predict_image(image_path)
                print()
            except Exception as e:
                print(f"❌ Error con {image_path}: {e}")
        else:
            print(f"❌ No encontrado: {image_path}")
    
    print("\n💡 Uso programático:")
    print("from hologram_analysis import HologramAnalyzer")
    print("analyzer = HologramAnalyzer('config.yaml')")
    print("result = analyzer.predict_single_image('imagen.png')")
    print("prediction = result['prediction']['class']")