#!/usr/bin/env python3
"""
Dataset Unifier - Crear dataset unificado desde dataset_1 y dataset_2
Para uso interno en hologram_classifier_v2
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path

class DatasetUnifier:
    """
    Unifica dataset_1 y dataset_2 en una estructura consistente
    """
    
    def __init__(self, data_dir: str, target_size=(1190, 1585)):
        """
        Inicializar unificador
        
        Args:
            data_dir: Directorio que contiene dataset_1 y dataset_2
            target_size: Tamaño objetivo (height, width) basado en dataset_2
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.target_height, self.target_width = target_size
        
    def extract_center_patch(self, image):
        """
        Extraer parche central de la imagen
        """
        h, w = image.shape[:2]
        
        if h < self.target_height or w < self.target_width:
            # Si la imagen es más pequeña, redimensionar
            return cv2.resize(image, (self.target_width, self.target_height))
        
        # Calcular coordenadas del centro
        center_y, center_x = h // 2, w // 2
        
        # Calcular coordenadas del parche
        start_y = center_y - self.target_height // 2
        end_y = start_y + self.target_height
        start_x = center_x - self.target_width // 2
        end_x = start_x + self.target_width
        
        # Asegurar que no se salga de los límites
        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = min(h, end_y)
        end_x = min(w, end_x)
        
        patch = image[start_y:end_y, start_x:end_x]
        
        # Si el parche es menor al tamaño objetivo, redimensionar
        if patch.shape[:2] != self.target_size:
            patch = cv2.resize(patch, (self.target_width, self.target_height))
        
        return patch
    
    def create_unified_dataset(self, output_dir: str) -> dict:
        """
        Crear dataset unificado combinando dataset_1 y dataset_2
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Crear carpetas para clases unificadas
        for class_name in ['Healthy', 'SCD']:
            (output_path / class_name).mkdir(exist_ok=True)
        
        # Mapeo de clases
        class_mapping = {
            'dataset_1': {'Healthy': 'Healthy', 'SCD': 'SCD'},
            'dataset_2': {'H-RBC': 'Healthy', 'SCD-RBC': 'SCD'}
        }
        
        counters = {'Healthy': 1, 'SCD': 1}
        stats = {'dataset_1_processed': 0, 'dataset_2_copied': 0, 'total': 0}
        
        print(f"🔧 Creando dataset unificado en: {output_dir}")
        
        # Procesar Dataset 1 (extraer parches)
        print(f"\n📁 Procesando Dataset 1...")
        dataset_1_path = self.data_dir / "dataset_1"
        
        if dataset_1_path.exists():
            for original_class, unified_class in class_mapping['dataset_1'].items():
                class_path = dataset_1_path / original_class
                if not class_path.exists():
                    continue
                    
                image_files = [f for f in class_path.iterdir() if f.suffix.lower() == '.png']
                print(f"  {original_class} → {unified_class}: {len(image_files)} imágenes")
                
                for img_file in image_files:
                    # Cargar imagen
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    # Extraer parche central
                    patch = self.extract_center_patch(img)
                    
                    # Generar nombre unificado
                    new_filename = f"unified_{unified_class}_{counters[unified_class]:04d}_ds1.png"
                    output_file_path = output_path / unified_class / new_filename
                    
                    # Guardar parche
                    if cv2.imwrite(str(output_file_path), patch):
                        counters[unified_class] += 1
                        stats['dataset_1_processed'] += 1
        
        # Procesar Dataset 2 (copiar directamente)
        print(f"\n📁 Procesando Dataset 2...")
        dataset_2_path = self.data_dir / "dataset_2"
        
        if dataset_2_path.exists():
            for original_class, unified_class in class_mapping['dataset_2'].items():
                class_path = dataset_2_path / original_class
                if not class_path.exists():
                    continue
                    
                image_files = [f for f in class_path.iterdir() if f.suffix.lower() == '.png']
                print(f"  {original_class} → {unified_class}: {len(image_files)} imágenes")
                
                for img_file in image_files:
                    # Cargar y verificar imagen
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    # Verificar tamaño
                    if img.shape[:2] != self.target_size:
                        img = cv2.resize(img, (self.target_width, self.target_height))
                    
                    # Generar nombre unificado
                    new_filename = f"unified_{unified_class}_{counters[unified_class]:04d}_ds2.png"
                    output_file_path = output_path / unified_class / new_filename
                    
                    # Guardar imagen
                    if cv2.imwrite(str(output_file_path), img):
                        counters[unified_class] += 1
                        stats['dataset_2_copied'] += 1
        
        stats['total'] = stats['dataset_1_processed'] + stats['dataset_2_copied']
        stats['healthy_count'] = counters['Healthy'] - 1
        stats['scd_count'] = counters['SCD'] - 1
        
        print(f"\n✅ Dataset unificado creado:")
        print(f"  Dataset 1 (parches): {stats['dataset_1_processed']}")
        print(f"  Dataset 2 (copiados): {stats['dataset_2_copied']}")  
        print(f"  Total: {stats['total']}")
        print(f"  Healthy: {stats['healthy_count']}, SCD: {stats['scd_count']}")
        
        return stats

def create_unified_dataset_if_needed(data_dir: str) -> str:
    """
    Crear dataset unificado si no existe
    
    Returns:
        Ruta al dataset unificado
    """
    unified_path = Path(data_dir) / "unified"
    
    if unified_path.exists() and any(unified_path.iterdir()):
        print(f"✅ Dataset unificado ya existe en: {unified_path}")
        return str(unified_path)
    
    print(f"🔧 Dataset unificado no encontrado, creando...")
    unifier = DatasetUnifier(data_dir)
    unifier.create_unified_dataset(str(unified_path))
    
    return str(unified_path)