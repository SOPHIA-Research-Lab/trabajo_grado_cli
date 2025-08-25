# 🔬 Hologram Classifier v2.0 - Trabajo de Grado

**Versión simplificada y mejorada del sistema de clasificación de hologramas para anemia falciforme**

## 🎯 Objetivo

Sistema unificado que ejecuta **todo el análisis con un solo comando**, eliminando duplicación de código y mejorando la interpretabilidad de resultados.

## ⚡ Ejecución Rápida (Un Solo Comando)

```bash
python run_analysis.py
```

¡Eso es todo! El sistema ejecutará automáticamente:
1. ✅ **Unificación automática** del dataset (si es necesario)
2. ✅ Carga del dataset unificado
3. ✅ Extracción de características (LBP, GLCM, FFT, Hu moments)  
4. ✅ Entrenamiento del modelo optimizado
5. ✅ Validación cruzada + Leave-One-Out CV
6. ✅ Análisis de interpretabilidad
7. ✅ Generación de visualizaciones
8. ✅ Reporte final completo

## 📁 Estructura

```
hologram_classifier_v2/
├── config.yaml           # Configuración central
├── run_analysis.py       # 🎯 COMANDO PRINCIPAL
├── data/                 # 📁 DATOS LOCALES (INCLUIDOS)
│   ├── dataset_1/        # Dataset original (3000x4000)
│   ├── dataset_2/        # Dataset adicional (1190x1585)
│   └── unified/          # Dataset unificado (se crea automáticamente)
├── src/
│   ├── hologram_analysis.py  # Lógica unificada
│   └── dataset_unifier.py    # Unificador de datasets
├── results/              # Resultados generados
│   ├── analysis_report.txt
│   ├── hologram_model.pkl
│   ├── feature_importance.png
│   └── feature_distributions.png
└── README.md            # Esta guía
```

## ⚙️ Configuración

Edita `config.yaml` para personalizar:

```yaml
# Rutas de datos
data_dir: "./data"                    # Carpeta con dataset_1 y dataset_2  
dataset_path: "./data/unified"        # Dataset unificado (se crea automáticamente)
output_dir: "./results"               # Carpeta de resultados

# Modelo
model:
  target_size: [512, 512]             # Tamaño de imágenes
  test_size: 0.2                      # Split train/test
  top_k_features: 20                  # Características a seleccionar

# Validación
validation:
  use_loocv: true                     # Usar Leave-One-Out CV
  n_cv_folds: 5                       # Folds para CV estándar

# Visualización  
visualization:
  save_plots: true                    # Guardar gráficos
  show_top_features: 10               # Características a mostrar
```

## 📊 Resultados Esperados

Después de ejecutar `python run_analysis.py`:

### 📈 Performance Típico
- **Accuracy**: ~94-98%
- **AUC**: ~0.97-0.99
- **LOOCV Accuracy**: ~94%
- **Errores LOOCV**: 18-20/304 muestras

### 📄 Archivos Generados
- `analysis_report.txt`: Reporte completo con métricas
- `hologram_model.pkl`: Modelo entrenado listo para usar
- `confusion_matrix.png`: Matriz de confusión
- `feature_importance.png`: Top características importantes
- `feature_distributions.png`: Distribuciones discriminativas

### 🔍 Interpretabilidad
- ✅ Top características más importantes identificadas
- ✅ Coeficientes del modelo interpretables  
- ✅ Análisis estadístico por clase (Cohen's d)
- ✅ Visualizaciones de distribuciones

## 🛠️ Requisitos

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib seaborn pyyaml scikit-image
```

## 📋 Prerequisitos

1. **Datasets incluidos** en `./data/`
   ```
   data/
   ├── dataset_1/
   │   ├── Healthy/    # Células sanas (3000x4000 px)
   │   └── SCD/        # Células SCD (3000x4000 px)
   ├── dataset_2/
   │   ├── H-RBC/      # Células sanas (1190x1585 px)
   │   └── SCD-RBC/    # Células SCD (1190x1585 px)
   └── unified/        # Se crea automáticamente al ejecutar
       ├── Healthy/    # Combina ambos datasets
       └── SCD/        # Combina ambos datasets
   ```

2. **Python 3.7+** con las librerías mencionadas

✅ **VENTAJA**: El sistema unifica automáticamente ambos datasets al ejecutarse, sin necesidad de preparación manual.

## 🚀 Ventajas vs Versión Original

| Aspecto | Original | v2.0 Mejorada |
|---------|----------|---------------|
| **Comandos necesarios** | >10 scripts | **1 comando** |
| **Código duplicado** | 2000+ líneas | **0 líneas** |
| **Configuración** | Hardcodeada | **Archivo YAML** |
| **Preparación de datos** | Manual compleja | **Automática** |
| **Dependencias externas** | Múltiples rutas | **Auto-contenido** |
| **Interpretabilidad** | Básica | **Mejorada con visualizaciones** |
| **Mantenimiento** | Complejo | **Simple y modular** |
| **Tiempo de ejecución** | ~10 min | **~3-5 min** |

## 🐛 Solución de Problemas

### Error: "Dataset no encontrado"
```bash
# Verificar que existen los datasets base
ls ./data/
# Debe mostrar dataset_1/ y dataset_2/

# Si faltan, el sistema los creará automáticamente
# O verificar dataset unificado creado:
ls ./data/unified/
# Debe mostrar carpetas Healthy/ y SCD/
```

### Error: "Módulo no encontrado" 
```bash
# Instalar dependencias
pip install -r requirements.txt
# O manualmente:
pip install numpy pandas scikit-learn opencv-python matplotlib seaborn pyyaml scikit-image
```

### Personalizar configuración
```bash
# Editar configuración
nano config.yaml
# Cambiar rutas, parámetros, etc.
```

## 📞 Soporte

Para preguntas sobre el código o resultados:
1. Revisar `./results/analysis_report.txt` para detalles técnicos
2. Verificar configuración en `config.yaml`
3. Comprobar que existen los datasets en `./data/dataset_1/` y `./data/dataset_2/`
4. El dataset unificado se crea automáticamente en la primera ejecución

## 🎓 Uso para Trabajo de Grado

Este sistema simplificado está diseñado específicamente para trabajos académicos:

- **Fácil de ejecutar**: Un solo comando
- **Auto-contenido**: Todos los datos incluidos, sin dependencias externas
- **Unificación automática**: Combina datasets automáticamente
- **Resultados reproducibles**: Configuración centralizada
- **Reportes automáticos**: Listos para incluir en tesis
- **Visualizaciones claras**: Ideales para presentaciones
- **Código limpio**: Fácil de revisar y modificar

**¡Perfecto para demostrar dominio técnico sin complejidad innecesaria!** 🎯