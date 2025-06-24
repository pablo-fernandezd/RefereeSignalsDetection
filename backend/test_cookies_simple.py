#!/usr/bin/env python3
"""
Test simple para verificar el uso de cookies.txt
"""

import os
from pathlib import Path

def test_cookies_file():
    """Verifica si existe el archivo cookies.txt"""
    print("Test de archivo cookies.txt")
    print("=" * 30)
    
    # Buscar cookies.txt en diferentes ubicaciones
    possible_paths = [
        Path("cookies.txt"),  # En el directorio actual (backend)
        Path("../cookies.txt"),  # Un nivel arriba
        Path("../../cookies.txt"),  # Dos niveles arriba
    ]
    
    found = False
    for path in possible_paths:
        if path.exists():
            print(f"✓ cookies.txt encontrado en: {path.absolute()}")
            print(f"  Tamaño: {path.stat().st_size} bytes")
            found = True
            break
    
    if not found:
        print("✗ cookies.txt no encontrado")
        print("\nPara usar cookies:")
        print("1. Coloca tu archivo cookies.txt en el directorio backend/")
        print("2. O en el directorio raíz del proyecto")
        print("3. El sistema lo detectará automáticamente")
    
    return found

def test_youtube_processor():
    """Test básico del YouTube processor"""
    print("\nTest del YouTube Processor")
    print("=" * 30)
    
    try:
        from youtube_processor import YouTubeProcessor
        processor = YouTubeProcessor()
        print("✓ YouTubeProcessor importado correctamente")
        
        # Verificar que puede acceder al modelo
        print(f"✓ Modelo cargado: {processor.model is not None}")
        print(f"✓ Dispositivo: {processor.DEVICE}")
        print(f"✓ Directorio base: {processor.base_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error al importar YouTubeProcessor: {e}")
        return False

def main():
    """Ejecuta todos los tests"""
    print("Test de Configuración de Cookies")
    print("=" * 40)
    
    cookies_ok = test_cookies_file()
    processor_ok = test_youtube_processor()
    
    print("\n" + "=" * 40)
    print("Resumen:")
    print(f"Cookies: {'✓ OK' if cookies_ok else '✗ No encontrado'}")
    print(f"Processor: {'✓ OK' if processor_ok else '✗ Error'}")
    
    if cookies_ok and processor_ok:
        print("\n✅ Todo listo para probar descargas!")
        print("Puedes ejecutar: python src/utils/descargador.py")
    else:
        print("\n⚠️  Algunos problemas detectados")
        if not cookies_ok:
            print("- Coloca tu cookies.txt en backend/")
        if not processor_ok:
            print("- Verifica las dependencias del proyecto")

if __name__ == "__main__":
    main() 