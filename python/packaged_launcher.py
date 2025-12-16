#!/usr/bin/env python3
"""
Packaged app launcher that properly sets up the bundled Python environment.
This script ensures bundled dependencies are used instead of system packages.
"""
import sys
import os
import site

def setup_bundled_environment():
    """Setup Python to use bundled dependencies only"""
    # Get the directory where this launcher script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Clear existing paths to avoid conflicts
    # Keep only the essential system paths
    essential_paths = []
    for p in sys.path[:]:
        if 'python3' in p.lower() and 'site-packages' not in p.lower():
            essential_paths.append(p)
    
    # Reset sys.path with only essential paths
    sys.path = essential_paths
    
    # Detect model type from environment
    model_type = os.environ.get('MODEL_TYPE', 'yolo').lower()
    
    # Add the appropriate bundled environment
    if model_type == 'retinaface':
        env_dir = os.path.join(parent_dir, 'retinaface-env')
    else:
        env_dir = os.path.join(parent_dir, 'yolo-env')
    
    # Add script directory first (for our Python modules)
    sys.path.insert(0, script_dir)
    
    # Add bundled environment directory
    if os.path.exists(env_dir):
        sys.path.insert(0, env_dir)
        print(f"Using bundled {model_type} environment: {env_dir}", file=sys.stderr)
        
        # Set environment variables to isolate from system Python
        os.environ['PYTHONPATH'] = env_dir
        os.environ['PYTHONNOUSERSITE'] = '1'  # Ignore user site-packages
        
        # Disable site packages to avoid conflicts
        site.ENABLE_USER_SITE = False
    else:
        print(f"Warning: Bundled environment not found: {env_dir}", file=sys.stderr)
        print(f"Falling back to system Python packages", file=sys.stderr)
    
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"sys.path: {sys.path}", file=sys.stderr)

def main():
    """Main entry point"""
    try:
        # Setup bundled environment
        setup_bundled_environment()
        
        # Import and run the subprocess API
        import subprocess_api
        
        # Create and run the API
        api = subprocess_api.SubprocessAPI()
        api.run()
        
    except ImportError as e:
        import json
        error_msg = {
            "type": "error",
            "message": f"Failed to import required module: {str(e)}"
        }
        print(json.dumps(error_msg))
        sys.stderr.write(f"Import error: {e}\n")
        sys.stderr.write(f"Python path: {sys.path}\n")
        sys.exit(1)
    except Exception as e:
        import json
        error_msg = {
            "type": "error",
            "message": f"Launcher error: {str(e)}"
        }
        print(json.dumps(error_msg))
        sys.stderr.write(f"Error: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()