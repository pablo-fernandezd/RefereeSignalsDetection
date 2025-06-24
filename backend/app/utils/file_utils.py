"""
File Utilities for Referee Detection System

This module provides utilities for file operations, validation, and management.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Union
from werkzeug.datastructures import FileStorage

from config.settings import FileConfig


def allowed_file(filename: str) -> bool:
    """
    Check if a file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    file_ext = Path(filename).suffix.lower()
    allowed_extensions = FileConfig.ALLOWED_IMAGE_EXTENSIONS | FileConfig.ALLOWED_VIDEO_EXTENSIONS
    return file_ext in allowed_extensions


def save_uploaded_file(file: FileStorage, upload_dir: Union[str, Path]) -> str:
    """
    Save an uploaded file to the specified directory.
    
    Args:
        file: Uploaded file object
        upload_dir: Directory to save the file to
        
    Returns:
        Filename of the saved file
        
    Raises:
        ValueError: If file is invalid or too large
        OSError: If file cannot be saved
    """
    if not file or not file.filename:
        raise ValueError("No file provided")
    
    if not allowed_file(file.filename):
        raise ValueError(f"File type not allowed: {file.filename}")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > FileConfig.MAX_FILE_SIZE:
        raise ValueError(f"File too large: {file_size} bytes (max: {FileConfig.MAX_FILE_SIZE})")
    
    # Ensure upload directory exists
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Save file
    filename = file.filename
    file_path = upload_path / filename
    
    try:
        file.save(str(file_path))
        return filename
    except Exception as e:
        raise OSError(f"Failed to save file: {e}")


def ensure_directory_exists(directory: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def copy_file_safely(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Copy a file safely with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if copy was successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            return False
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
        return True
        
    except Exception:
        return False


def move_file_safely(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Move a file safely with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if move was successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            return False
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(src_path), str(dst_path))
        return True
        
    except Exception:
        return False


def delete_file_safely(file_path: Union[str, Path]) -> bool:
    """
    Delete a file safely with error handling.
    
    Args:
        file_path: Path to the file to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            return True
        return False
        
    except Exception:
        return False


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes, -1 if file doesn't exist
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            return path.stat().st_size
        return -1
        
    except Exception:
        return -1 