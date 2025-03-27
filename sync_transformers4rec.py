#!/usr/bin/env python3
import os
import shutil
import difflib
import filecmp
from pathlib import Path

# Source and destination directories
SOURCE_DIR = "/home/areiner/miniforge3/envs/merlin/lib/python3.10/site-packages/transformers4rec"
DEST_DIR = "/home/areiner/_/repos/thirdparty/Transformers4Rec/transformers4rec"

# Count variables for reporting
files_copied = 0
files_skipped = 0
dirs_created = 0

def compare_and_copy_file(src_file, dest_file):
    """Compare files and copy only if different"""
    global files_copied, files_skipped
    
    # Create parent directory if it doesn't exist
    if not os.path.exists(os.path.dirname(dest_file)):
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        global dirs_created
        dirs_created += 1
    
    # If destination doesn't exist, copy the file
    if not os.path.exists(dest_file):
        shutil.copy2(src_file, dest_file)
        print(f"Created: {dest_file}")
        files_copied += 1
        return
    
    # If files are different, copy the file
    if not filecmp.cmp(src_file, dest_file, shallow=False):
        shutil.copy2(src_file, dest_file)
        print(f"Updated: {dest_file}")
        files_copied += 1
    else:
        print(f"Skipped (identical): {dest_file}")
        files_skipped += 1

def sync_directories():
    """Walk through source directory and sync to destination"""
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Calculate relative path from source root
        rel_path = os.path.relpath(root, SOURCE_DIR)
        
        # Create corresponding destination directory
        dest_root = os.path.join(DEST_DIR, rel_path) if rel_path != '.' else DEST_DIR
        os.makedirs(dest_root, exist_ok=True)
        
        # Copy only .py files with differences
        for file in files:
            if file.endswith('.py') or file == 'py.typed':
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_root, file)
                compare_and_copy_file(src_file, dest_file)

if __name__ == "__main__":
    print(f"Syncing from {SOURCE_DIR} to {DEST_DIR}")
    sync_directories()
    print(f"\nSync complete:")
    print(f"- Files copied: {files_copied}")
    print(f"- Files skipped (identical): {files_skipped}")
    print(f"- Directories created: {dirs_created}") 