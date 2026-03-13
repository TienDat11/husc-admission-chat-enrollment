#!/usr/bin/env python3
"""
Reinstall dependencies with corrected requirements.txt
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("Reinstalling dependencies...")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Check if we're in venv
    venv_python = Path("venv/Scripts/python.exe")
    if venv_python.exists():
        pip_cmd = [str(venv_python), "-m", "pip"]
    else:
        pip_cmd = [sys.executable, "-m", "pip"]
    
    # Install requirements
    try:
        print("Installing from requirements.txt...")
        result = subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"], 
                               check=True, capture_output=True, text=True)
        print("Installation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    import os
    success = main()
    if success:
        print("Now try running run_lab.bat again")
    else:
        print("Check the errors and fix requirements.txt if needed")
