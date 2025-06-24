"""
Hash Utility Functions

This module provides utility functions for managing image hashes,
including hash calculation, registry management, and duplicate detection.
"""

import hashlib
from pathlib import Path
from typing import Set, Union, Optional
import threading

from config.settings import DirectoryConfig


# Thread-safe hash registry cache
_hash_cache: Optional[Set[str]] = None
_hash_cache_lock = threading.Lock()


def calculate_image_hash(image_path: Union[str, Path]) -> str:
    """
    Calculate the MD5 hash of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MD5 hash as a hexadecimal string
        
    Raises:
        OSError: If the file cannot be read
        ValueError: If the file is empty or invalid
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise OSError(f"Image file not found: {image_path}")
    
    if image_path.stat().st_size == 0:
        raise ValueError(f"Image file is empty: {image_path}")
    
    hasher = hashlib.md5()
    try:
        with open(image_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as e:
        raise OSError(f"Failed to read image file {image_path}: {e}")


def load_image_hashes() -> Set[str]:
    """
    Load all registered image hashes into a set for quick lookups.
    
    Returns:
        Set of all registered image hashes
    """
    global _hash_cache
    
    with _hash_cache_lock:
        if _hash_cache is not None:
            return _hash_cache.copy()
        
        hash_registry = DirectoryConfig.IMAGE_HASH_REGISTRY
        hashes = set()
        
        if hash_registry.exists():
            try:
                with open(hash_registry, 'r', encoding='utf-8') as f:
                    for line in f:
                        hash_value = line.strip()
                        if hash_value and len(hash_value) == 32:  # MD5 hash length
                            hashes.add(hash_value)
            except OSError:
                # If we can't read the file, start with empty set
                pass
        
        _hash_cache = hashes
        return hashes.copy()


def is_hash_registered(image_hash: str) -> bool:
    """
    Check if an image hash is already registered.
    
    Args:
        image_hash: MD5 hash to check
        
    Returns:
        True if the hash is registered, False otherwise
    """
    if not image_hash or len(image_hash) != 32:
        return False
    
    registered_hashes = load_image_hashes()
    return image_hash in registered_hashes


def register_hash(image_hash: str) -> bool:
    """
    Register a new image hash in the registry.
    
    Args:
        image_hash: MD5 hash to register
        
    Returns:
        True if registration was successful, False otherwise
    """
    if not image_hash or len(image_hash) != 32:
        return False
    
    # Check if already registered
    if is_hash_registered(image_hash):
        return True
    
    global _hash_cache
    hash_registry = DirectoryConfig.IMAGE_HASH_REGISTRY
    
    try:
        # Ensure the registry file exists
        hash_registry.parent.mkdir(parents=True, exist_ok=True)
        
        # Append the new hash
        with open(hash_registry, 'a', encoding='utf-8') as f:
            f.write(f"{image_hash}\n")
        
        # Update cache
        with _hash_cache_lock:
            if _hash_cache is not None:
                _hash_cache.add(image_hash)
        
        return True
    except OSError:
        return False


def register_image_hash(image_path: Union[str, Path]) -> bool:
    """
    Calculate and register the hash of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if registration was successful, False otherwise
    """
    try:
        image_hash = calculate_image_hash(image_path)
        return register_hash(image_hash)
    except (OSError, ValueError):
        return False


def is_duplicate_image(image_path: Union[str, Path]) -> bool:
    """
    Check if an image is a duplicate based on its hash.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if the image is a duplicate, False otherwise
    """
    try:
        image_hash = calculate_image_hash(image_path)
        return is_hash_registered(image_hash)
    except (OSError, ValueError):
        return False


def clear_hash_cache():
    """
    Clear the in-memory hash cache to force reload from file.
    This is useful when the hash registry file is modified externally.
    """
    global _hash_cache
    with _hash_cache_lock:
        _hash_cache = None


def get_hash_registry_stats() -> dict:
    """
    Get statistics about the hash registry.
    
    Returns:
        Dictionary containing registry statistics
    """
    hash_registry = DirectoryConfig.IMAGE_HASH_REGISTRY
    registered_hashes = load_image_hashes()
    
    stats = {
        'total_hashes': len(registered_hashes),
        'registry_exists': hash_registry.exists(),
        'registry_size_bytes': 0,
        'cache_loaded': _hash_cache is not None
    }
    
    if hash_registry.exists():
        try:
            stats['registry_size_bytes'] = hash_registry.stat().st_size
        except OSError:
            pass
    
    return stats


def cleanup_hash_registry() -> int:
    """
    Clean up the hash registry by removing invalid entries.
    
    Returns:
        Number of invalid entries removed
    """
    hash_registry = DirectoryConfig.IMAGE_HASH_REGISTRY
    
    if not hash_registry.exists():
        return 0
    
    valid_hashes = []
    invalid_count = 0
    
    try:
        with open(hash_registry, 'r', encoding='utf-8') as f:
            for line in f:
                hash_value = line.strip()
                if hash_value and len(hash_value) == 32:  # Valid MD5 hash
                    valid_hashes.append(hash_value)
                else:
                    invalid_count += 1
        
        # Write back only valid hashes
        if invalid_count > 0:
            with open(hash_registry, 'w', encoding='utf-8') as f:
                for hash_value in valid_hashes:
                    f.write(f"{hash_value}\n")
            
            # Clear cache to force reload
            clear_hash_cache()
        
        return invalid_count
    except OSError:
        return 0 