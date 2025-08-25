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
        elif args.system:
            self._info_system()
        else:
            # Mostrar toda la información
            self._info_system()
            self._info_dataset()
            self._info_config()
            
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
    info_parser.add_argument('--system', action='store_true', help='Información del sistema')
    
    # Comando config
    config_parser = subparsers.add_parser('config', help='Configurar el sistema')
    config_parser.add_argument('--mode', choices=['quick', 'full', 'deep'], help='Cambiar modo de análisis')
    config_parser.add_argument('--features', type=int, help='Número máximo de características')
    config_parser.add_argument('--cache', type=bool, help='Habilitar/deshabilitar cache')
    config_parser.add_argument('--optimize', type=bool, help='Habilitar/deshabilitar optimización')
    config_parser.add_argument('--progress', type=bool, help='Habilitar/deshabilitar barras de progreso')
    config_parser.add_argument('--output', help='Directorio de salida')
    
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