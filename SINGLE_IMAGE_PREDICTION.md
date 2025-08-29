# 🔍 Predicción de Imagen Individual con Detección de Anomalías

Guía rápida para analizar una sola imagen con el modelo entrenado, incluyendo detección automática de anomalías.

## 🚀 Uso Rápido

### 1. Asegúrate de tener el modelo entrenado
```bash
# Si no has entrenado el modelo, ejecuta:
python run_analysis.py
```

### 2. Análisis básico de una imagen
```python
from src.hologram_analysis import HologramAnalyzer

# Crear analizador
analyzer = HologramAnalyzer("config.yaml")

# Analizar imagen
result = analyzer.predict_single_image("path/to/imagen.png")

# Ver resultado
print(f"Predicción: {result['prediction']['class']}")
print(f"Confianza: {result['prediction']['confidence']:.1%}")
```

### 3. Análisis completo con reporte y visualización
```python
# Análisis con todos los extras
result = analyzer.predict_single_image(
    "path/to/imagen.png",
    save_report=True,        # Genera reporte .txt
    save_visualization=True  # Genera gráficos .png
)
```

## 📝 Ejemplos Listos para Usar

### Ejemplo Simple
```bash
python simple_prediction_example.py
```

### Ejemplo Completo
```bash
# Con imágenes de ejemplo
python example_single_prediction.py

# Con tu propia imagen
python example_single_prediction.py path/to/tu/imagen.png
```

## 📊 Lo que Obtienes

```json
{
  "prediction": {
    "class": "SCD",           // "Healthy" o "SCD"
    "confidence": 0.905,      // Nivel de confianza
    "probabilities": {
      "Healthy": 0.095,
      "SCD": 0.905
    }
  },
  "features": {
    "total_extracted": 84,    // Características extraídas
    "total_selected": 50      // Características usadas
  },
  "analysis": {
    "top_discriminative_features": [...] // Características más importantes
  }
}
```

## 📁 Archivos Generados

Al usar `save_report=True` y `save_visualization=True`:

- **Reporte**: `./results/single_image_analysis_YYYYMMDD_HHMMSS.txt`
- **Visualización**: `./results/single_image_viz_YYYYMMDD_HHMMSS.png`

## 🎯 Casos de Uso

**Para desarrollo rápido:**
```python
prediction = analyzer.predict_single_image("imagen.png")['prediction']['class']
```

**Para análisis detallado:**
```python
result = analyzer.predict_single_image("imagen.png", save_report=True, save_visualization=True)
```

**Para integración en otros sistemas:**
```python
confidence = analyzer.predict_single_image("imagen.png")['prediction']['confidence']
if confidence > 0.8:
    # Predicción confiable
    pass
```

## ⚠️ Requisitos

1. **Modelo entrenado**: Ejecutar `python run_analysis.py` al menos una vez
2. **Imagen válida**: Formatos PNG, JPG soportados
3. **Dependencias**: Mismas que el proyecto principal

## 🔧 Solución de Problemas

**Error "Modelo no encontrado":**
```bash
python run_analysis.py  # Entrena el modelo primero
```

**Error "Imagen no encontrada":**
- Verifica la ruta del archivo
- Usa rutas absolutas si es necesario

**Error de importación:**
```bash
pip install -r requirements.txt
```

---

🎯 **¡Listo!** Con esto puedes analizar cualquier imagen individual de holograma.