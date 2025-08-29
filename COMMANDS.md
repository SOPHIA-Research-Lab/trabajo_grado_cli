# 🔬 Hologram Classifier - Comandos Disponibles

## 📋 **COMANDOS RÁPIDOS**

### **🚀 Análisis Básico**
```bash
# Análisis completo (recomendado)
python cli.py analyze

# Análisis rápido (30 segundos)
python cli.py analyze --mode quick

# Análisis profundo (10 minutos)
python cli.py analyze --mode deep
```

### **💾 Gestión de Cache**
```bash
# Ver información del cache
python cli.py cache --info

# Limpiar cache
python cli.py cache --clear
```

### **ℹ️ Información del Sistema**
```bash
# Información completa
python cli.py info

# Solo dataset
python cli.py info --dataset

# Solo modelo entrenado
python cli.py info --model
```

### **⚙️ Configuración Rápida**
```bash
# Cambiar a modo profundo
python cli.py config --mode deep

# Cambiar número de características
python cli.py config --features 50
```

---

## 📖 **REFERENCIA COMPLETA DE COMANDOS**

### **`analyze` - Ejecutar Análisis**

```bash
python cli.py analyze [opciones]
```

**Opciones:**
- `--mode {quick,full,deep}`: Modo de análisis
- `--features N`: Número máximo de características (ej: 30, 50)
- `--no-cache`: Deshabilitar cache de características
- `--no-optimize`: Deshabilitar optimización automática
- `--no-progress`: Deshabilitar barras de progreso
- `--output DIR`: Directorio de salida personalizado
- `--show-config`: Mostrar configuración antes del análisis
- `--debug`: Mostrar trazas completas en errores

**Ejemplos:**
```bash
# Análisis básico
python cli.py analyze

# Análisis personalizado
python cli.py analyze --mode deep --features 40 --output ./resultados

# Ver configuración y ejecutar
python cli.py analyze --show-config --mode full

# Análisis sin cache (datos frescos)
python cli.py analyze --no-cache --no-optimize
```

### **`cache` - Gestión del Cache**

```bash
python cli.py cache {--info|--clear}
```

**Opciones:**
- `--info`: Mostrar información del cache
- `--clear`: Limpiar todo el cache

**Ejemplos:**
```bash
# Ver tamaño y archivos del cache
python cli.py cache --info

# Limpiar cache para empezar fresco
python cli.py cache --clear
```

### **`info` - Información del Sistema**

```bash
python cli.py info [opciones]
```

**Opciones:**
- `--dataset`: Información del dataset
- `--model`: Información del modelo entrenado
- `--config`: Configuración actual
- `--system`: Información del sistema
- *(sin opciones)*: Toda la información

**Ejemplos:**
```bash
# Información completa
python cli.py info

# Solo información del dataset
python cli.py info --dataset

# Solo configuración actual
python cli.py info --config

# Verificar modelo entrenado
python cli.py info --model
```

### **`config` - Configurar Sistema**

```bash
python cli.py config [opciones]
```

**Opciones:**
- `--mode {quick,full,deep}`: Cambiar modo de análisis
- `--features N`: Número máximo de características
- `--cache {true,false}`: Habilitar/deshabilitar cache
- `--optimize {true,false}`: Habilitar/deshabilitar optimización
- `--progress {true,false}`: Habilitar/deshabilitar barras progreso
- `--output DIR`: Directorio de salida
- *(sin opciones)*: Mostrar configuración actual

**Ejemplos:**
```bash
# Ver configuración actual
python cli.py config

# Cambiar a modo profundo
python cli.py config --mode deep

# Configurar para máximo rendimiento
python cli.py config --mode deep --features 50 --optimize true

# Deshabilitar cache
python cli.py config --cache false
```

---

## 🎯 **MODOS DE ANÁLISIS**

### **⚡ Modo Quick**
```bash
python cli.py analyze --mode quick
```
- **Tiempo**: ~30 segundos
- **Características**: ≤20
- **Validación**: 3-fold CV
- **Cache**: Reutiliza si existe
- **Ideal**: Desarrollo y pruebas

### **🚀 Modo Full (Default)**
```bash
python cli.py analyze --mode full
```
- **Tiempo**: ~5 minutos
- **Características**: ≤35
- **Validación**: 5-fold CV + LOOCV
- **Optimización**: 5 min máx
- **Ideal**: Uso general

### **🔬 Modo Deep**
```bash
python cli.py analyze --mode deep
```
- **Tiempo**: ~10 minutos
- **Características**: ≤50
- **Validación**: Completa + estabilidad
- **Optimización**: 10 min máx
- **Ideal**: Investigación

---

## 🆘 **COMANDOS DE AYUDA**

### **Ayuda General**
```bash
python cli.py --help
python cli.py -h
```

### **Ayuda por Comando**
```bash
python cli.py analyze --help
python cli.py cache --help
python cli.py info --help
python cli.py config --help
```

---

## 🔧 **COMANDOS DE DIAGNÓSTICO**

### **Verificar Sistema**
```bash
# Estado completo del sistema
python cli.py info --system

# Verificar dependencias
python cli.py info --system | grep "Dependencias"
```

### **Debug de Errores**
```bash
# Ejecutar con trazas completas
python cli.py analyze --debug

# Verificar configuración antes de error
python cli.py info --config
```

### **Limpiar y Reiniciar**
```bash
# Limpiar cache
python cli.py cache --clear

# Restablecer configuración a modo full
python cli.py config --mode full --cache true --optimize true
```

---