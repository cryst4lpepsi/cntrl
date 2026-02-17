#!/usr/bin/env python3
"""
Setup and Build Script for Hand Gesture Knob Controller
Creates standalone executable using PyInstaller
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed successfully!\n")


def build_executable():
    """Build standalone executable using PyInstaller"""
    print("Building standalone executable...")
    
    # PyInstaller configuration
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'mediapipe',
        'cv2',
        'rtmidi',
        'PyQt6',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GestureKnobController',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)
"""
    
    # Write spec file
    with open('gesture_controller.spec', 'w') as f:
        f.write(spec_content)
    
    # Run PyInstaller
    subprocess.run([sys.executable, "-m", "PyInstaller", "gesture_controller.spec", "--clean"])
    
    print("\nBuild complete! Executable located in 'dist' folder")


def create_directories():
    """Create necessary directories"""
    directories = ['config', 'assets', 'docs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("Directories created successfully!\n")


def run_tests():
    """Run tests if available"""
    if Path('tests').exists():
        print("Running tests...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/"])


def main():
    """Main setup function"""
    print("=" * 60)
    print("Hand Gesture Knob Controller - Setup & Build Tool")
    print("=" * 60)
    print()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "install":
            create_directories()
            install_dependencies()
            
        elif command == "build":
            build_executable()
            
        elif command == "test":
            run_tests()
            
        elif command == "all":
            create_directories()
            install_dependencies()
            run_tests()
            build_executable()
            
        else:
            print(f"Unknown command: {command}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage information"""
    print("Usage:")
    print("  python setup.py install    - Install dependencies")
    print("  python setup.py build      - Build standalone executable")
    print("  python setup.py test       - Run tests")
    print("  python setup.py all        - Do everything")
    print()


if __name__ == '__main__':
    main()