#!/usr/bin/env python3
"""
Multi-environment launcher for the face detection app.
Dynamically switches between YOLO and RetinaFace environments based on model selection.
"""
import sys
import os
import json

def setup_python_path(model_type='yolo'):
    """Add bundled dependencies to Python path based on model type"""
    # Get the directory where this launcher script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Check if we're in development mode (source files) or packaged mode
    # Development mode: script_dir is in the source python/ directory 
    # Packaged mode: script_dir is in pythondist/python/ inside app bundle
    is_development = (os.path.exists(os.path.join(script_dir, '../src')) or 
                     (os.path.basename(script_dir) == 'python' and 
                      os.path.basename(os.path.dirname(script_dir)) != 'pythondist'))
    
    print(f"Environment Detection:", file=sys.stderr)
    print(f"  Script directory: {script_dir}", file=sys.stderr)
    print(f"  Parent directory: {parent_dir}", file=sys.stderr) 
    print(f"  Development mode: {is_development}", file=sys.stderr)
    print(f"  Model type: {model_type}", file=sys.stderr)
    
    if is_development:
        # Development mode - Python environment switching handles dependencies via conda
        print(f"Development mode detected - using conda environment for {model_type} models", file=sys.stderr)
        
        # In development mode, the conda environment should already be set up correctly
        # Just add the script directory to the path
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        print(f"Added script directory to path: {script_dir}", file=sys.stderr)
    else:
        # Packaged mode - use bundled dependencies
        print(f"Packaged mode detected - using bundled dependencies for {model_type} models", file=sys.stderr)
        
        # Add the script directory to the path
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Determine the appropriate environment directory
        if model_type == 'retinaface':
            env_dir = os.path.join(parent_dir, 'retinaface-env')
        else:
            env_dir = os.path.join(parent_dir, 'yolo-env')
        
        # Check if the environment directory exists
        if os.path.exists(env_dir):
            # Check if this is a virtual environment
            venv_python = os.path.join(env_dir, 'bin', 'python')
            if os.path.exists(venv_python):
                # This is a virtual environment, set sys.executable
                sys.executable = venv_python
                print(f"Set virtual environment python: {venv_python}", file=sys.stderr)
            
            # Use virtual environment's site-packages directory by auto-detecting version
            site_packages_dir = None
            # Common layout: <env>/lib/pythonX.Y/site-packages
            lib_dir = os.path.join(env_dir, 'lib')
            if os.path.exists(lib_dir):
                python_dirs = sorted([d for d in os.listdir(lib_dir) if d.startswith('python')])
                for py_dir in python_dirs:
                    candidate = os.path.join(lib_dir, py_dir, 'site-packages')
                    if os.path.exists(candidate):
                        site_packages_dir = candidate
                        break
            # Windows layout: <env>\\Lib\\site-packages
            if site_packages_dir is None:
                win_candidate = os.path.join(env_dir, 'Lib', 'site-packages')
                if os.path.exists(win_candidate):
                    site_packages_dir = win_candidate
            
            # Add site-packages directory to the beginning of sys.path
            if site_packages_dir and os.path.exists(site_packages_dir) and site_packages_dir not in sys.path:
                sys.path.insert(0, site_packages_dir)
                print(f"Added {model_type} site-packages to path: {site_packages_dir}", file=sys.stderr)
            
            # Also add the environment directory itself for any loose modules
            if env_dir not in sys.path:
                sys.path.insert(0, env_dir)
            print(f"Added {model_type} environment to path: {env_dir}", file=sys.stderr)
            
            # Set PYTHONPATH to include site-packages
            if site_packages_dir and os.path.exists(site_packages_dir):
                sep = ';' if os.name == 'nt' else ':'
                os.environ['PYTHONPATH'] = site_packages_dir + sep + env_dir
            else:
                os.environ['PYTHONPATH'] = env_dir
            
            # Clear any existing imports to avoid conflicts
            modules_to_clear = []
            for module in list(sys.modules.keys()):
                if any(pkg in module for pkg in ['numpy', 'torch', 'tensorflow', 'cv2', 'ultralytics']):
                    modules_to_clear.append(module)
            
            for module in modules_to_clear:
                del sys.modules[module]
            
        else:
            print(f"Warning: Environment directory not found: {env_dir}", file=sys.stderr)
            print(f"Available directories in {parent_dir}:", file=sys.stderr)
            try:
                for item in os.listdir(parent_dir):
                    item_path = os.path.join(parent_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  {item}/", file=sys.stderr)
            except Exception as e:
                print(f"  Error listing directory: {e}", file=sys.stderr)
    
    # Log Python path for debugging
    print(f"Python path setup complete. Script dir: {script_dir}", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    
    # Test for critical dependencies based on model type
    print(f"Testing dependencies for {model_type} environment:", file=sys.stderr)
    if model_type == 'yolo':
        try:
            import torch
            print(f"  ✅ PyTorch {torch.__version__} available", file=sys.stderr)
        except ImportError as e:
            print(f"  ❌ PyTorch not available: {e}", file=sys.stderr)
        
        try:
            import ultralytics
            print(f"  ✅ Ultralytics {ultralytics.__version__} available", file=sys.stderr)
        except ImportError as e:
            print(f"  ❌ Ultralytics not available: {e}", file=sys.stderr)
    
    elif model_type == 'retinaface':
        # Cross-platform check: verify dependencies instead of hard-gating by platform
        try:
            # Reduce TF logs
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf  # noqa: F401
            print(f"  ✅ TensorFlow available", file=sys.stderr)
        except Exception as e:
            print(f"  ❌ TensorFlow not available for RetinaFace: {e}", file=sys.stderr)
        
        try:
            import retinaface  # noqa: F401
            print(f"  ✅ RetinaFace available", file=sys.stderr)
        except Exception as e:
            print(f"  ❌ RetinaFace not available: {e}", file=sys.stderr)
    
    # Test common dependencies
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__} available", file=sys.stderr)
    except ImportError as e:
        print(f"  ❌ OpenCV not available: {e}", file=sys.stderr)
    
    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__} available", file=sys.stderr)
    except ImportError as e:
        print(f"  ❌ NumPy not available: {e}", file=sys.stderr)

def detect_model_type():
    """Detect model type from environment or default to YOLO"""
    # Check environment variable first
    model_type = os.environ.get('MODEL_TYPE', 'yolo')
    
    # Check command line args
    if len(sys.argv) > 1 and 'retinaface' in sys.argv[1].lower():
        model_type = 'retinaface'
    
    return model_type

def main():
    """Main entry point - setup path and run subprocess API"""
    try:
        # Detect which model environment we need
        model_type = detect_model_type()
        
        # Setup Python path for the appropriate environment
        setup_python_path(model_type)
        
        # Now import and run the subprocess API
        # Import here after path is set up
        import subprocess_api
        
        # Create and run the API
        api = subprocess_api.SubprocessAPI()
        api.run()
        
    except ImportError as e:
        error_msg = {
            "type": "error",
            "message": f"Failed to import subprocess_api: {str(e)}"
        }
        print(json.dumps(error_msg))
        sys.stderr.write(f"Import error: {e}\n")
        sys.stderr.write(f"Python path: {sys.path}\n")
        sys.exit(1)
    except Exception as e:
        error_msg = {
            "type": "error", 
            "message": f"Launcher error: {str(e)}"
        }
        print(json.dumps(error_msg))
        sys.stderr.write(f"Launcher error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()