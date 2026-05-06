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
        # Locate the venv's site-packages — that's where cv2/torch/etc. actually live.
        # POSIX layout: <env>/lib/pythonX.Y/site-packages
        # Windows layout: <env>\Lib\site-packages
        site_packages_dir = None
        lib_dir = os.path.join(env_dir, 'lib')
        if os.path.exists(lib_dir):
            for entry in sorted(os.listdir(lib_dir)):
                if entry.startswith('python'):
                    candidate = os.path.join(lib_dir, entry, 'site-packages')
                    if os.path.exists(candidate):
                        site_packages_dir = candidate
                        break
        if site_packages_dir is None:
            win_candidate = os.path.join(env_dir, 'Lib', 'site-packages')
            if os.path.exists(win_candidate):
                site_packages_dir = win_candidate

        # Insert site-packages first so it wins over anything else; then env_dir
        # for any loose modules sitting at the env root.
        if site_packages_dir:
            sys.path.insert(0, site_packages_dir)
            print(f"Added bundled site-packages: {site_packages_dir}", file=sys.stderr)
        sys.path.insert(0, env_dir)
        print(f"Using bundled {model_type} environment: {env_dir}", file=sys.stderr)

        # Set PYTHONPATH so subprocesses (if any) inherit the same view.
        sep = ';' if os.name == 'nt' else ':'
        os.environ['PYTHONPATH'] = (
            f"{site_packages_dir}{sep}{env_dir}" if site_packages_dir else env_dir
        )
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