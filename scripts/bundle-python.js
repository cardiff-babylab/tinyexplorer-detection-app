#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');
const { createWriteStream } = require('fs');
const tar = require('tar');

const platform = process.platform;
const pythonDistDir = path.join(__dirname, '..', 'pythondist');

console.log('Creating Python bundle...');

// --- Bundled Python 3.10 support ---
const PY_VERSION = process.env.PYTHON_310_VERSION || '3.10.14';
const bundledPythonRoot = path.join(pythonDistDir, 'base-python-3.10');

function hasExecutable(file) {
    try { fs.accessSync(file, fs.constants.X_OK); return true; } catch { return false; }
}

function findExecutableRecursively(dir, namePattern) {
    if (!fs.existsSync(dir)) return null;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            const nested = findExecutableRecursively(full, namePattern);
            if (nested) return nested;
        } else if (entry.isFile() && namePattern.test(entry.name) && hasExecutable(full)) {
            return full;
        }
    }
    return null;
}

// Cross-platform recursive copy
function copyRecursiveSync(src, dest) {
    const exists = fs.existsSync(src);
    const stats = exists && fs.statSync(src);
    const isDirectory = exists && stats.isDirectory();
    if (isDirectory) {
        fs.mkdirSync(dest, { recursive: true });
        fs.readdirSync(src).forEach(childItemName => {
            copyRecursiveSync(path.join(src, childItemName),
                            path.join(dest, childItemName));
        });
    } else {
        fs.copyFileSync(src, dest);
    }
}

async function downloadFile(url, destPath) {
    return new Promise((resolve, reject) => {
        const file = createWriteStream(destPath);
        https.get(url, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301) {
                // Handle redirect
                file.close();
                downloadFile(response.headers.location, destPath).then(resolve).catch(reject);
                return;
            }
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download: ${response.statusCode}`));
                return;
            }
            response.pipe(file);
            file.on('finish', () => {
                file.close(resolve);
            });
        }).on('error', (err) => {
            fs.unlinkSync(destPath);
            reject(err);
        });
    });
}

async function downloadWindowsPython() {
    const arch = process.arch;
    let pythonUrl;
    
    // Use python-build-standalone for portable Python with full stdlib
    if (arch === 'x64') {
        pythonUrl = 'https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-3.10.11+20230507-x86_64-pc-windows-msvc-shared-install_only.tar.gz';
    } else {
        pythonUrl = 'https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-3.10.11+20230507-i686-pc-windows-msvc-shared-install_only.tar.gz';
    }
    
    const pythonDir = path.join(pythonDistDir, 'python-windows');
    const pythonExecutable = path.join(pythonDir, 'python.exe');
    
    // Check if already downloaded
    if (fs.existsSync(pythonExecutable)) {
        try {
            execSync(`"${pythonExecutable}" --version`, { stdio: 'ignore' });
            console.log(`Windows Python already available at ${pythonExecutable}`);
            return pythonExecutable;
        } catch (e) {
            console.log('Existing Python is broken, re-downloading...');
            fs.rmSync(pythonDir, { recursive: true, force: true });
        }
    }
    
    console.log(`Downloading portable Python for Windows...`);
    const archivePath = path.join(pythonDistDir, 'python.tar.gz');
    
    try {
        await downloadFile(pythonUrl, archivePath);
        console.log('Extracting Python...');
        
        fs.mkdirSync(pythonDir, { recursive: true });
        
        // Extract tar.gz using Node.js tar package
        await tar.x({
            file: archivePath,
            cwd: pythonDir,
            strip: 1
        });
        
        // Clean up archive
        fs.unlinkSync(archivePath);
        
        // The python-build-standalone includes pip by default
        
        console.log(`Windows Python ready at ${pythonExecutable}`);
        return pythonExecutable;
    } catch (error) {
        console.error('Failed to download Windows Python:', error);
        throw error;
    }
}

async function ensureBundledPython310() {
    // 1. Direct override path
    const override = process.env.BUNDLED_PYTHON_310;
    if (override) {
        try { execSync(`"${override}" --version`, { stdio: 'ignore' }); console.log(`Using BUNDLED_PYTHON_310: ${override}`); return override; } catch { console.warn('Override invalid, continuing.'); }
    }
    
    // 2. Forced system usage
    if (process.env.USE_SYSTEM_PYTHON === '1') return 'python3.10';
    
    // 3. Windows: Download Python
    if (process.platform === 'win32') {
        try {
            return await downloadWindowsPython();
        } catch (error) {
            console.warn('Failed to download Python for Windows, trying system Python...');
            // Try system Python as fallback
            const candidates = ['python3.10', 'python3', 'python'];
            for (const cmd of candidates) {
                try {
                    const version = execSync(`${cmd} --version 2>&1`).toString();
                    if (version.includes('3.10')) {
                        console.log(`Using system Python: ${cmd}`);
                        return cmd;
                    }
                } catch (e) {
                    // Continue
                }
            }
            throw new Error('No Python 3.10 found for Windows');
        }
    }
    
    // 4. macOS/Linux: Use existing approach with Homebrew/pyenv
    if (process.platform === 'darwin') {
        let brewAvailable = false;
        try { execSync('brew --version', { stdio: 'ignore' }); brewAvailable = true; } catch (_) {}
        if (brewAvailable) {
            const brewPy = ['/opt/homebrew/opt/python@3.10/bin/python3.10','/usr/local/opt/python@3.10/bin/python3.10'];
            let found = brewPy.find(p => hasExecutable(p));
            if (!found) {
                try {
                    console.log('Homebrew python@3.10 not found, installing via brew (this may take a minute)...');
                    execSync('brew install python@3.10', { stdio: 'inherit' });
                    found = brewPy.find(p => hasExecutable(p));
                } catch (e) {
                    console.warn('brew install python@3.10 failed:', e.message);
                }
            }
            if (found) {
                console.log(`Using Homebrew python3.10 at ${found}`);
                return found;
            }
        }
    }
    
    // 5. pyenv detection (macOS/Linux)
    if (process.platform !== 'win32') {
        let pyenvAvailable = false;
        let pyenvCmd = 'pyenv';
        try { execSync('pyenv --version', { stdio: 'ignore' }); pyenvAvailable = true; } catch (_) {}
        if (!pyenvAvailable) {
            // Attempt lightweight embedded pyenv clone
            const toolsDir = path.join(pythonDistDir, 'tools');
            const embeddedPyenvRoot = path.join(toolsDir, 'pyenv');
            try {
                if (!fs.existsSync(embeddedPyenvRoot)) {
                    console.log('Installing embedded pyenv...');
                    fs.mkdirSync(toolsDir, { recursive: true });
                    execSync(`git clone --depth 1 https://github.com/pyenv/pyenv.git "${embeddedPyenvRoot}"`, { stdio: 'inherit' });
                } else {
                    console.log('Embedded pyenv already present.');
                }
                pyenvCmd = `PYENV_ROOT="${embeddedPyenvRoot}" ${embeddedPyenvRoot}/bin/pyenv`;
                execSync(`${pyenvCmd} --version`, { stdio: 'ignore', env: { ...process.env, PYENV_ROOT: embeddedPyenvRoot } });
                pyenvAvailable = true;
                console.log(`Using embedded pyenv at ${embeddedPyenvRoot}`);
            } catch (e) {
                console.warn('Embedded pyenv bootstrap failed:', e.message);
            }
        }
        if (pyenvAvailable) {
            try {
                const versionsRaw = execSync(`${pyenvCmd} versions --bare`, { env: { ...process.env, PYENV_ROOT: pyenvCmd.includes('PYENV_ROOT') ? pyenvCmd.split(' ')[1].replace(/"/g,'') : process.env.PYENV_ROOT } }).toString().split(/\n+/).map(v=>v.trim()).filter(Boolean);
                let chosen = versionsRaw.filter(v => v.startsWith('3.10.')).sort().pop();
                if (!chosen) {
                    chosen = PY_VERSION;
                    console.log(`pyenv: auto-installing ${chosen} (no 3.10.x found)...`);
                    execSync(`${pyenvCmd} install -s ${chosen}`, { stdio: 'inherit', env: { ...process.env } });
                }
                const pyPath = execSync(`PYENV_VERSION=${chosen} ${pyenvCmd} which python`, { env: { ...process.env } }).toString().trim();
                console.log(`Using pyenv Python ${chosen} at ${pyPath}`);
                return pyPath;
            } catch (e) {
                console.warn('pyenv provisioning failed:', e.message);
            }
        }
    }
    
    // 6. Fallback to PATH
    try { execSync('python3.10 --version', { stdio: 'ignore' }); return 'python3.10'; } catch(_) {}
    try { execSync('python3 --version', { stdio: 'ignore' }); return 'python3'; } catch(_) {}
    try { execSync('python --version', { stdio: 'ignore' }); return 'python'; } catch(_) {}
    
    console.error('No python3.10 found');
    process.exit(1);
}

// Helper for RetinaFace
async function findPythonForRetinaFace() {
    const forced = process.env.RETINAFACE_PYTHON || process.env.PYTHON_FOR_RETINAFACE;
    if (forced) {
        try {
            execSync(`"${forced}" --version`, { stdio: 'ignore' });
            console.log(`Using forced RetinaFace Python from env: ${forced}`);
            return forced;
        } catch (e) {
            console.warn(`Warning: forced RetinaFace Python not usable: ${forced}. Falling back to bundled 3.10.`);
        }
    }
    return await ensureBundledPython310();
}

// Clean previous build
if (fs.existsSync(pythonDistDir)) {
    fs.rmSync(pythonDistDir, { recursive: true, force: true });
}

// Create distribution directory
fs.mkdirSync(pythonDistDir, { recursive: true });

// Copy Python source files to python subdirectory
console.log('Copying Python source files...');
const pythonSubDir = path.join(pythonDistDir, 'python');
fs.mkdirSync(pythonSubDir, { recursive: true });
copyRecursiveSync(path.join(__dirname, '..', 'python'), pythonSubDir);

// Install dependencies to both yolo-env and retinaface-env directories
console.log('Installing Python dependencies to dual environments...');

// Create both environment directories
const yoloEnvDir = path.join(pythonDistDir, 'yolo-env');
const retinafaceEnvDir = path.join(pythonDistDir, 'retinaface-env');
fs.mkdirSync(yoloEnvDir, { recursive: true });
fs.mkdirSync(retinafaceEnvDir, { recursive: true });

// Use requirements from source tree
const yoloRequirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
const retinafaceRequirementsPath = path.join(__dirname, '..', 'python', 'requirements-retinaface.txt');

// Main async function
async function setupEnvironments() {
    try {
        console.log('Creating virtual environments for better package isolation...');
        
        // Create YOLO virtual environment
        console.log('Creating YOLO virtual environment...');
        const pyForYolo = await ensureBundledPython310();
        console.log(`Using ${pyForYolo} to create YOLO virtual environment...`);
        // Use --copies to ensure executables are copied, not symlinked (better for relocation)
        execSync(`"${pyForYolo}" -m venv --copies "${yoloEnvDir}"`, { stdio: 'inherit' });

        // Make venv relocatable by removing absolute path reference
        const yoloPyvenvCfg = path.join(yoloEnvDir, 'pyvenv.cfg');
        if (fs.existsSync(yoloPyvenvCfg)) {
            let cfg = fs.readFileSync(yoloPyvenvCfg, 'utf8');
            // Remove the 'home' line which contains absolute CI path - venv will use relative paths
            cfg = cfg.split('\n').filter(line => !line.trim().startsWith('home =')).join('\n');
            fs.writeFileSync(yoloPyvenvCfg, cfg, 'utf8');
            console.log('Made YOLO venv relocatable by removing absolute home path');
        }
        
        // Install YOLO packages
        console.log('Installing YOLO packages...');
        const yoloPython = process.platform === 'win32'
            ? path.join(yoloEnvDir, 'Scripts', 'python.exe')
            : path.join(yoloEnvDir, 'bin', 'python');

        // Upgrade pip first
        execSync(`"${yoloPython}" -m pip install --no-cache-dir --upgrade pip setuptools wheel`, {
            stdio: 'inherit',
            env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
        });

        // Install dependencies
        const isLinux = process.platform === 'linux';
        let installCmd = `"${yoloPython}" -m pip install --no-cache-dir --upgrade -r "${yoloRequirementsPath}"`;
        if (isLinux) {
            installCmd = `"${yoloPython}" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --upgrade -r "${yoloRequirementsPath}"`;
        }

        try {
            execSync(installCmd, { stdio: 'inherit', env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' } });
        } catch (primaryInstallError) {
            console.warn('Primary YOLO dependency install failed. Attempting fallback...');
            try {
                if (isLinux) {
                    execSync(`"${yoloPython}" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.2.2 torchvision==0.17.2`, {
                        stdio: 'inherit',
                        env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
                    });
                } else {
                    execSync(`"${yoloPython}" -m pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2`, {
                        stdio: 'inherit',
                        env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
                    });
                }
                execSync(`"${yoloPython}" -m pip install --no-cache-dir --upgrade -r "${yoloRequirementsPath}"`, {
                    stdio: 'inherit',
                    env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
                });
            } catch (fallbackError) {
                console.error('Fallback YOLO dependency install also failed.');
                throw fallbackError;
            }
        }
        
        // Create RetinaFace virtual environment
        console.log('Creating RetinaFace virtual environment...');
        try {
            const retinafacePythonSystem = await findPythonForRetinaFace();
            console.log(`Selected system Python for RetinaFace venv: ${retinafacePythonSystem}`);
            // Use --copies to ensure executables are copied, not symlinked (better for relocation)
            execSync(`"${retinafacePythonSystem}" -m venv --copies "${retinafaceEnvDir}"`, { stdio: 'inherit' });

            // Make venv relocatable by removing absolute path reference
            const retinafacePyvenvCfg = path.join(retinafaceEnvDir, 'pyvenv.cfg');
            if (fs.existsSync(retinafacePyvenvCfg)) {
                let cfg = fs.readFileSync(retinafacePyvenvCfg, 'utf8');
                // Remove the 'home' line which contains absolute CI path - venv will use relative paths
                cfg = cfg.split('\n').filter(line => !line.trim().startsWith('home =')).join('\n');
                fs.writeFileSync(retinafacePyvenvCfg, cfg, 'utf8');
                console.log('Made RetinaFace venv relocatable by removing absolute home path');
            }

            console.log('Installing RetinaFace packages...');
            const retinafacePython = process.platform === 'win32'
                ? path.join(retinafaceEnvDir, 'Scripts', 'python.exe')
                : path.join(retinafaceEnvDir, 'bin', 'python');

            // Verify Python version compatibility
            try {
                const verOut = execSync(`"${retinafacePython}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"`).toString().trim();
                const [maj, min] = verOut.split('.').map(Number);
                if (maj !== 3 || min > 11) {
                    throw new Error(`Unsupported Python ${verOut} for TensorFlow`);
                }
                console.log(`RetinaFace Python version: ${verOut}`);
            } catch (verErr) {
                console.error('RetinaFace Python version check failed:', verErr.message);
                throw verErr;
            }

            // Upgrade pip
            execSync(`"${retinafacePython}" -m pip install --no-cache-dir --upgrade pip setuptools wheel`, { 
                stdio: 'inherit', 
                env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' } 
            });

            const isDarwin = process.platform === 'darwin';
            const isArm64 = process.arch === 'arm64';
            const isX64 = process.arch === 'x64';
            let retinafaceOk = false;

            function installList(pkgs, note) {
                console.log(`Installing RetinaFace package set (${note})...`);
                const quoted = pkgs.map(p => `"${p}"`).join(' ');
                execSync(`"${retinafacePython}" -m pip install --no-cache-dir ${quoted}`, { 
                    stdio: 'inherit', 
                    env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' } 
                });
                retinafaceOk = true;
            }

            try {
                if (isDarwin && isArm64) {
                    installList([
                        'flask', 'flask-cors', 'graphene>=3.0', 'flask-graphql>=2.0',
                        'opencv-python', 'pillow', 'numpy<2.0.0',
                        'tensorflow-macos==2.15.0', 'tensorflow-metal==1.1.0', 'tf-keras==2.15.0',
                        'retina-face>=0.0.14'
                    ], 'macOS arm64 TF 2.15 + metal');
                } else if (isDarwin && isX64) {
                    installList([
                        'flask', 'flask-cors', 'graphene>=3.0', 'flask-graphql>=2.0',
                        'opencv-python', 'pillow', 'numpy<2.0.0',
                        'tensorflow==2.15.0', 'tf-keras==2.15.0',
                        'retina-face>=0.0.14'
                    ], 'macOS x64 TF 2.15 CPU');
                } else {
                    console.log('Installing RetinaFace requirements from file...');
                    execSync(`"${retinafacePython}" -m pip install --no-cache-dir --upgrade -r "${retinafaceRequirementsPath}"`, { 
                        stdio: 'inherit', 
                        env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' } 
                    });
                    retinafaceOk = true;
                }
            } catch (primaryRfError) {
                console.warn('Primary RetinaFace install failed:', primaryRfError.message);
                console.warn('Attempting fallback...');
                try {
                    installList([
                        'flask', 'flask-cors', 'graphene>=3.0', 'flask-graphql>=2.0',
                        'opencv-python', 'pillow', 'numpy<2.0.0',
                        'tensorflow==2.15.0', 'tf-keras==2.15.0',
                        'retina-face>=0.0.14'
                    ], 'fallback TF 2.15 CPU');
                } catch (fallbackRfError) {
                    console.warn('Fallback RetinaFace install failed:', fallbackRfError.message);
                    console.warn('RetinaFace env will be incomplete (optional feature).');
                }
            }

            if (retinafaceOk) {
                console.log('RetinaFace environment created successfully!');
            } else {
                console.error('RetinaFace environment incomplete');
                throw new Error('RetinaFace environment failed');
            }
        } catch (retinafaceError) {
            console.error('Failed to create RetinaFace environment');
            console.error('Error:', retinafaceError.message);
            process.exit(1);
        }
        
        // Create python-deps symlink/copy for backwards compatibility
        const depsDir = path.join(pythonDistDir, 'python-deps');
        console.log('Creating python-deps for backwards compatibility...');
        try {
            if (fs.existsSync(depsDir)) {
                fs.rmSync(depsDir, { recursive: true, force: true });
            }
        } catch (_) {}

        if (process.platform === 'win32') {
            // On Windows, copy the directory
            copyRecursiveSync(yoloEnvDir, depsDir);
        } else {
            // On Unix-like systems, create symlink
            fs.symlinkSync('yolo-env', depsDir, 'dir');
        }

        console.log('Python bundle created successfully!');
        console.log(`Bundle size: ${getDirectorySize(pythonDistDir)} MB`);
        
    } catch (error) {
        console.error('Failed to create environments:', error.message);
        process.exit(1);
    }
}

function getDirectorySize(dir) {
    try {
        function getDirSizeRecursive(dirPath) {
            let size = 0;
            const files = fs.readdirSync(dirPath);
            for (const file of files) {
                const filePath = path.join(dirPath, file);
                const stats = fs.statSync(filePath);
                if (stats.isDirectory()) {
                    size += getDirSizeRecursive(filePath);
                } else {
                    size += stats.size;
                }
            }
            return size;
        }
        const sizeInBytes = getDirSizeRecursive(dir);
        const sizeInMB = Math.round(sizeInBytes / (1024 * 1024));
        return sizeInMB.toString();
    } catch (error) {
        return 'unknown';
    }
}

// Run the setup
setupEnvironments();