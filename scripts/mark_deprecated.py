#!/usr/bin/env python3
"""Utility script to mark files as deprecated.

This script adds a deprecation notice to the top of files that have been 
refactored and moved to the new structure.
"""

import os
import sys

FILES_TO_MARK = [
    "agents.py",
    "mlesolver.py",
    "papersolver.py",
    "tools.py",
    "utils.py",
    "ai_lab_repo.py",
    "app.py",
]

DEPRECATION_NOTICE = '''
# ===================================================================
# DEPRECATION NOTICE
# ===================================================================
# This file is deprecated and will be removed in a future version.
# It has been refactored and moved to the new modular structure.
# Please use the new implementation instead.
# ===================================================================

'''

def mark_file_as_deprecated(filename):
    """Add deprecation notice to a file."""
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found, skipping")
        return False
        
    with open(filename, 'r') as f:
        content = f.read()
    
    # Check if file already has deprecation notice
    if "DEPRECATION NOTICE" in content:
        print(f"File '{filename}' already marked as deprecated, skipping")
        return False
        
    # Add deprecation notice to the beginning of the file
    with open(filename, 'w') as f:
        f.write(DEPRECATION_NOTICE + content)
    
    print(f"Marked '{filename}' as deprecated")
    return True

def main():
    """Main function."""
    marked_files = 0
    for filename in FILES_TO_MARK:
        if mark_file_as_deprecated(filename):
            marked_files += 1
    
    print(f"Marked {marked_files} files as deprecated")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 