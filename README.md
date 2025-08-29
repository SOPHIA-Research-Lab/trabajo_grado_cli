# 🔬 Sistema de Análisis de Hologramas para Anemia Falciforme

Sistema completo de clasificación automática de hologramas de células sanguíneas para detección de anemia falciforme, con **detección integrada de anomalías**.

## ✨ Características Principales

- 🧠 **Clasificación Automática**: Distingue células sanas vs. anemia falciforme
- 🚨 **Detección de Anomalías**: Identifica muestras fuera del dominio de entrenamiento
- 📊 **84+ Características**: LBP, GLCM, FFT, Gabor, morfológicas y estadísticas
- 🎯 **Ensemble Learning**: Combinación de múltiples algoritmos ML
- 🔍 **Análisis Individual**: Predicción y análisis detallado de imágenes únicas
- 📈 **Validación Robusta**: Cross-validation y Leave-One-Out
- 🎨 **Visualizaciones**: Gráficos interpretativos y reportes detallados

## 🚀 Inicio Rápido

### Instalación
```bash
git clone https://github.com/your-repo/trabajo_grado_v2.git
cd trabajo_grado_v2
pip install -r requirements.txt
```

### Entrenar Modelo
```bash
# Análisis completo (recomendado)
python cli.py analyze

# Análisis rápido para pruebas
python cli.py analyze --mode quick

# Análisis profundo para investigación
python cli.py analyze --mode deep
```

### Validar con Anomalías
```bash
# Validar modelo con organismos no sanguíneos
python cli.py validate --image-dir ./holograms --save-report

# Ver información del detector de anomalías  
python cli.py info --anomaly
```

### Analizar Imagen Individual
```python
from src.hologram_analysis import HologramAnalyzer

analyzer = HologramAnalyzer("config.yaml")
result = analyzer.predict_single_image("imagen.jpg")

print(f"Predicción: {result['prediction']['class']}")
print(f"Confianza: {result['prediction']['confidence']:.3f}")

# Verificar anomalías
anomaly = result['anomaly_detection']
if anomaly['is_anomaly']:
    print("🚨 ANOMALÍA DETECTADA - Muestra fuera de dominio")
else:
    print("✅ Muestra normal")
```

## 🏗️ Arquitectura del Sistema

### Componentes Principales

```
trabajo_grado_v2/
├── 🔧 cli.py                    # Interfaz de línea de comandos
├── 📊 src/
│   ├── hologram_analysis.py     # Motor principal de análisis
│   ├── anomaly_detector.py      # Detección de anomalías
│   └── dataset_unifier.py       # Unificación de datasets
├── 📁 data/                     # Datasets de entrenamiento
├── 📈 results/                  # Modelos y reportes generados
├── 🔬 holograms/               # Datos de validación
└── 📚 docs/                    # Documentación
```

### Pipeline de Procesamiento

1. **Carga de Datos** → Unificación automática de datasets
2. **Extracción de Características** → 84 features por imagen
3. **Entrenamiento** → Ensemble de 4 algoritmos ML
4. **Creación de Detector** → Automática basada en datos de entrenamiento
5. **Validación** → Cross-validation + LOOCV + detección de anomalías
6. **Reportes** → Análisis completo + visualizaciones

## 🚨 Sistema de Detección de Anomalías


### Algoritmo
- **Método**: Distancia de Mahalanobis
- **Umbral**: Percentil 95 (5% falsos positivos)
- **Features**: Mismas 84 características del modelo principal
- **Entrenamiento**: Automático con datos de células sanguíneas

### Métricas de Salida
```python
{
    'is_anomaly': True,                    # Boolean: ¿Es anomalía?
    'mahalanobis_distance': 2347.8,       # Distancia al espacio normal
    'anomaly_score': 1.0,                 # Score normalizado (0-1)  
    'threshold': 2.846,                   # Umbral de decisión
    'p_value': 0.000001,                  # Probabilidad estadística
    'closest_class': 'SCD'                # Clase más similar
}
```

## 📊 Características Extraídas

| Tipo | Cantidad | Descripción |
|------|----------|-------------|
| **LBP** | 26 | Local Binary Patterns - texturas locales |
| **GLCM** | 6 | Gray-Level Co-occurrence Matrix - texturas estadísticas |
| **FFT** | 3 | Fast Fourier Transform - frecuencias dominantes |
| **Hu Moments** | 7 | Momentos invariantes - forma |
| **Morfológicas** | 5 | Circularidad, solidez, excentricidad |
| **Gabor** | 16 | Filtros multi-orientación |
| **Estadísticas** | 5 | Media, std, asimetría, curtosis, entropía |
| **Bordes** | 3 | Densidad y características de bordes |
| **Wavelets** | 12 | Descomposición multi-escala |
| **TOTAL** | **84** | Features por imagen |

## 🎯 Resultados Típicos

### Performance del Modelo
- **Precisión**: 85-95%
- **AUC**: 0.85-0.95
- **Validación**: 5-fold CV + LOOCV
- **Características**: ~30-50 seleccionadas automáticamente

### Detección de Anomalías
- **Células sanguíneas normales**: Distancia < 3.0, score < 0.6
- **Organismos no sanguíneos**: Distancia > 100, score = 1.0
- **Tasa de detección**: ~95-100% en organismos conocidos
- **Falsos positivos**: ~5% (configurables)

## 🔧 Comandos CLI Disponibles

```bash
# 📊 ANÁLISIS
python cli.py analyze                    # Análisis completo
python cli.py analyze --mode quick      # Análisis rápido (~30s)
python cli.py analyze --mode deep       # Análisis profundo (~10min)

# 🚨 VALIDACIÓN DE ANOMALÍAS  
python cli.py validate --image-dir DIR  # Validar directorio
python cli.py validate --limit 5        # Limitar imágenes por categoría
python cli.py validate --no-anomaly     # Sin detección de anomalías

# ℹ️ INFORMACIÓN
python cli.py info                       # Información completa
python cli.py info --model             # Solo información del modelo
python cli.py info --anomaly           # Solo detector de anomalías
python cli.py info --dataset           # Solo información de datos

# 🗂️ CACHE
python cli.py cache --info             # Información del cache
python cli.py cache --clear            # Limpiar cache

# ⚙️ CONFIGURACIÓN
python cli.py config --mode deep       # Cambiar modo por defecto
python cli.py config --features 40     # Cambiar número de características
```

## 📚 Documentación

- 📄 [**SINGLE_IMAGE_PREDICTION.md**](SINGLE_IMAGE_PREDICTION.md) - Análisis de imágenes individuales
- 🚨 [**ANOMALY_DETECTION.md**](ANOMALY_DETECTION.md) - Guía completa de detección de anomalías
- 💻 [**CLI_GUIDE.md**](CLI_GUIDE.md) - Referencia completa del CLI
- 🔧 [**COMMANDS.md**](COMMANDS.md) - Lista de comandos disponibles

## 🔬 Casos de Uso

### 1. Diagnóstico Clínico
```python
# Analizar muestra de paciente
result = analyzer.predict_single_image("muestra_paciente.jpg")

if result['anomaly_detection']['is_anomaly']:
    print("⚠️ Muestra anómala - verificar preparación")
else:
    diagnosis = result['prediction']['class']
    confidence = result['prediction']['confidence']
    print(f"Diagnóstico: {diagnosis} (confianza: {confidence:.1%})")
```

### 2. Control de Calidad
```bash
# Validar lote de muestras
python cli.py validate --image-dir ./lote_muestras --save-report
```

### 3. Investigación
```python
# Analizar características discriminativas
result = analyzer.predict_single_image("muestra.jpg")
top_features = result['analysis']['top_discriminative_features']

for feature in top_features[:5]:
    print(f"{feature['name']}: {feature['discriminative_power']:.3f}")
```

## 📈 Configuración

### config.yaml (Principal)
```yaml
# Modelo
model:
  top_k_features: 35
  test_size: 0.2
  auto_optimize: true

# Detección de anomalías  
anomaly_detection:
  enabled: true
  threshold_percentile: 95.0
  algorithm: "mahalanobis"

# Ejecución
execution:
  mode: "full"        # quick, full, deep
  cache_features: true
  progress_bar: true
```

## 🛠️ Dependencias

```txt
scikit-learn>=1.0.0
scikit-image>=0.19.0  
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
PyWavelets>=1.1.0
PyYAML>=6.0
joblib>=1.1.0
tqdm>=4.64.0
```

