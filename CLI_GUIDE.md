# 🔬 Hologram Classifier CLI v2.0 - Guía Completa

##  **Comandos Principales**

### ** Análisis (`analyze`)**

Ejecutar análisis completo de hologramas:

```bash
# Análisis básico (modo completo)
python cli.py analyze

# Análisis rápido 
python cli.py analyze --mode quick

# Análisis profundo 
python cli.py analyze --mode deep

# Análisis personalizado
python cli.py analyze --features 40 --no-cache --output ./mi_analisis

# Ver configuración antes de ejecutar
python cli.py analyze --show-config

# Modo debug con trazas completas
python cli.py analyze --debug
```

**Opciones disponibles:**
- `--mode {quick,full,deep}`: Modo de análisis
- `--features N`: Número máximo de características (ej: 30, 50)
- `--no-cache`: Deshabilitar cache de características
- `--no-optimize`: Deshabilitar optimización automática de hiperparámetros
- `--no-progress`: Deshabilitar barras de progreso
- `--output DIR`: Directorio de salida personalizado
- `--show-config`: Mostrar configuración antes del análisis
- `--debug`: Mostrar trazas completas en caso de error
