import tarfile
import os
from pathlib import Path

def extract_data():
    """Extract spam and ham datasets"""
    
    # Extract spam
    print("Extracting spam data...")
    with tarfile.open("data/raw/spam.tar.bz2", "r:bz2") as tar:
        tar.extractall("data/raw/")
    
    # Extract ham
    print("Extracting ham data...")
    with tarfile.open("data/raw/ham.tar.bz2", "r:bz2") as tar:
        tar.extractall("data/raw/")
    
    print("Extraction complete!")
    print(f"Check data/raw/ folder for extracted files")

if __name__ == "__main__":
    extract_data()