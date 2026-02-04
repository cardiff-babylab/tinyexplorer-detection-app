#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');
const tar = require('tar');

const platform = process.platform;
const arch = process.arch;
const pythonDistDir = path.join(__dirname, '..', 'pythondist');

console.log('Downloading standalone Python 3.10 for macOS...');

/**
 * Fix Python symlinks in a venv to be relative to python-standalone
 * This ensures the venv remains relocatable when packaged
 */
function fixVenvPythonSymlinks(venvDir, pythonStandaloneDir) {
    if (platform === 'win32') {
        // Windows doesn't use symlinks for Python in venvs
        return;
    }
    
    const binDir = path.join(venvDir, 'bin');
    const pythonLinks = ['python', 'python3', 'python3.10'];
    
    // Calculate relative path from venv/bin to python-standalone/bin
    const relPath = path.relative(binDir, path.join(pythonStandaloneDir, 'bin', 'python3.10'));
    
    console.log(`Fixing venv symlinks in ${binDir} -> ${relPath}`);
    
    for (const link of pythonLinks) {
        const linkPath = path.join(binDir, link);
        try {
            const stat = fs.lstatSync(linkPath);
            if (stat.isSymbolicLink()) {
                const target = fs.readlinkSync(linkPath);
                // Check if it's an absolute path or points outside the bundle
                if (path.isAbsolute(target) || target.includes('..')) {
                    fs.unlinkSync(linkPath);
                    if (link === 'python3.10') {
                        // Point directly to the standalone Python
                        fs.symlinkSync(relPath, linkPath);
                    } else {
                        // python and python3 should point to python3.10
                        fs.symlinkSync('python3.10', linkPath);
                    }
                    console.log(`  Fixed ${link} -> ${link === 'python3.10' ? relPath : 'python3.10'}`);
                }
            }
        } catch (err) {
            // Link doesn't exist or other error, skip
        }
    }
}

/**
 * Downloads a file from a URL
 */
function downloadFile(url, destPath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(destPath);
        https.get(url, { 
            headers: { 'User-Agent': 'Mozilla/5.0' },
            followRedirects: true 
        }, (response) => {
            if (response.statusCode === 302 || response.statusCode === 301) {
                // Handle redirect
                downloadFile(response.headers.location, destPath).then(resolve).catch(reject);
                return;
            }
            
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download: ${response.statusCode}`));
                return;
            }
            
            const totalSize = parseInt(response.headers['content-length'], 10);
            let downloadedSize = 0;
            
            response.on('data', (chunk) => {
                downloadedSize += chunk.length;
                const progress = Math.round((downloadedSize / totalSize) * 100);
                process.stdout.write(`\rDownloading Python: ${progress}%`);
            });
            
            response.pipe(file);
            
            file.on('finish', () => {
                file.close();
                console.log('\nDownload complete');
                resolve();
            });
        }).on('error', (err) => {
            fs.unlink(destPath, () => {}); // Delete the file on error
            reject(err);
        });
    });
}

/**
 * Get the appropriate Python standalone build URL for the current platform
 */
function getPythonStandaloneUrl() {
    // Using python-build-standalone releases from indygreg
    // These are relocatable Python distributions
    const baseUrl = 'https://github.com/astral-sh/python-build-standalone/releases/download/20241016/';
    
    if (platform === 'darwin') {
        // For macOS, handle both arm64 (Apple Silicon) and x86_64 (Intel)
        // Use the environment ARCH variable or detect from process.arch
        const targetArch = process.env.ARCH || process.arch;
        
        if (targetArch === 'arm64') {
            const filename = 'cpython-3.10.15+20241016-aarch64-apple-darwin-install_only_stripped.tar.gz';
            return baseUrl + filename;
        } else {
            // Default to x86_64 for Intel Macs
            const filename = 'cpython-3.10.15+20241016-x86_64-apple-darwin-install_only_stripped.tar.gz';
            return baseUrl + filename;
        }
    } else if (platform === 'linux') {
        const filename = 'cpython-3.10.15+20241016-x86_64_v3-unknown-linux-gnu-install_only_stripped.tar.gz';
        return baseUrl + filename;
    } else if (platform === 'win32') {
        const filename = 'cpython-3.10.15+20241016-x86_64-pc-windows-msvc-shared-install_only_stripped.tar.gz';
        return baseUrl + filename;
    } else {
        throw new Error(`Unsupported platform: ${platform}`);
    }
}

async function downloadAndExtractPython() {
    const pythonStandaloneDir = path.join(pythonDistDir, 'python-standalone');
    
    // Check if already downloaded
    if (fs.existsSync(path.join(pythonStandaloneDir, 'bin', 'python3.10')) || 
        fs.existsSync(path.join(pythonStandaloneDir, 'python.exe'))) {
        console.log('Standalone Python already exists, skipping download');
        return pythonStandaloneDir;
    }
    
    // Create directory
    fs.mkdirSync(pythonStandaloneDir, { recursive: true });
    
    // Download Python
    const url = getPythonStandaloneUrl();
    const tarPath = path.join(pythonDistDir, 'python-standalone.tar.gz');
    
    console.log(`Downloading from: ${url}`);
    await downloadFile(url, tarPath);
    
    // Extract Python
    console.log('Extracting Python...');
    await tar.extract({
        file: tarPath,
        cwd: pythonStandaloneDir,
        strip: 1, // Remove the top-level directory from the archive
    });
    
    // Clean up tar file
    fs.unlinkSync(tarPath);
    
    // Make Python executable on Unix-like systems
    if (platform !== 'win32') {
        const pythonBin = path.join(pythonStandaloneDir, 'bin', 'python3.10');
        if (fs.existsSync(pythonBin)) {
            fs.chmodSync(pythonBin, '755');
        }
        // Also make python3 symlink if it doesn't exist
        const python3Link = path.join(pythonStandaloneDir, 'bin', 'python3');
        if (!fs.existsSync(python3Link)) {
            fs.symlinkSync('python3.10', python3Link);
        }
    }
    
    console.log('Standalone Python extracted successfully');
    return pythonStandaloneDir;
}

async function createVirtualEnvironments(pythonStandaloneDir) {
    const pythonExe = platform === 'win32' 
        ? path.join(pythonStandaloneDir, 'python.exe')
        : path.join(pythonStandaloneDir, 'bin', 'python3.10');
    
    if (!fs.existsSync(pythonExe)) {
        throw new Error(`Python executable not found at: ${pythonExe}`);
    }
    
    // Create both environment directories
    const yoloEnvDir = path.join(pythonDistDir, 'yolo-env');
    const retinafaceEnvDir = path.join(pythonDistDir, 'retinaface-env');
    
    // Copy Python source files to python subdirectory
    console.log('Copying Python source files...');
    const pythonSubDir = path.join(pythonDistDir, 'python');
    fs.mkdirSync(pythonSubDir, { recursive: true });
    execSync(`cp -r python/ ${pythonSubDir}/`);
    
    // Use requirements from source tree
    const yoloRequirementsPath = path.join(__dirname, '..', 'python', 'requirements.txt');
    const retinafaceRequirementsPath = path.join(__dirname, '..', 'python', 'requirements-retinaface.txt');
    
    console.log('Creating virtual environments with standalone Python...');
    
    // Create YOLO virtual environment
    console.log('Creating YOLO virtual environment...');
    execSync(`"${pythonExe}" -m venv "${yoloEnvDir}"`, { stdio: 'inherit' });
    
    // Fix symlinks to be relative for relocatability
    fixVenvPythonSymlinks(yoloEnvDir, pythonStandaloneDir);
    // Install YOLO packages
    const yoloPython = platform === 'win32'
        ? path.join(yoloEnvDir, 'Scripts', 'python.exe')
        : path.join(yoloEnvDir, 'bin', 'python');
    
    execSync(`"${yoloPython}" -m pip install --no-cache-dir --upgrade pip setuptools wheel`, {
        stdio: 'inherit',
        env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
    });
    
    console.log('Installing YOLO dependencies...');
    execSync(`"${yoloPython}" -m pip install --no-cache-dir --upgrade -r "${yoloRequirementsPath}"`, {
        stdio: 'inherit',
        env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
    });
    
    // Create RetinaFace virtual environment
    console.log('Creating RetinaFace virtual environment...');
    execSync(`"${pythonExe}" -m venv "${retinafaceEnvDir}"`, { stdio: 'inherit' });
    
    // Fix symlinks to be relative for relocatability
    fixVenvPythonSymlinks(retinafaceEnvDir, pythonStandaloneDir);
    
    // Install RetinaFace packages
    const retinafacePython = platform === 'win32'
        ? path.join(retinafaceEnvDir, 'Scripts', 'python.exe')
        : path.join(retinafaceEnvDir, 'bin', 'python');
    
    execSync(`"${retinafacePython}" -m pip install --no-cache-dir --upgrade pip setuptools wheel`, {
        stdio: 'inherit',
        env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
    });
    
    console.log('Installing RetinaFace dependencies...');

    // Install platform-appropriate TensorFlow and dependencies
    if (platform === 'darwin') {
        const isArm64 = arch === 'arm64' || process.env.ARCH === 'arm64';
        if (isArm64) {
            // Apple Silicon: install TF stack and pin numpy in ONE transaction, then install retina-face without deps
            execSync(`"${retinafacePython}" -m pip install --no-cache-dir --upgrade ` +
                     `flask flask-cors "graphene>=3.0" "flask-graphql>=2.0" ` +
                     `opencv-python pillow "numpy<2.0.0" "gdown>=3.10.1" ` +
                     `tensorflow-macos==2.15.0 tensorflow-metal==1.1.0 tf-keras==2.15.0`, {
                stdio: 'inherit',
                env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
            });
            // Prevent retina-face from upgrading tensorflow/numpy by skipping its deps
            execSync(`"${retinafacePython}" -m pip install --no-cache-dir --no-deps "retina-face>=0.0.14"`, {
                stdio: 'inherit',
                env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
            });
        } else {
            // Intel macOS: pin tensorflow CPU 2.15 (requires AVX). If target Macs lack AVX, RetinaFace will not run.
            const packages = [
                'flask', 'flask-cors',
                '"graphene>=3.0"', '"flask-graphql>=2.0"',
                'opencv-python', 'pillow', '"numpy<2.0.0"', '"gdown>=3.10.1"',
                'tensorflow==2.15.0', 'tf-keras==2.15.0',
                '"retina-face>=0.0.14"'
            ];
            for (const pkg of packages) {
                execSync(`"${retinafacePython}" -m pip install --no-cache-dir ${pkg}`, {
                    stdio: 'inherit',
                    env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
                });
            }
        }
    } else {
        // Other platforms - use requirements file (pins TF accordingly)
        execSync(`"${retinafacePython}" -m pip install --no-cache-dir --upgrade -r "${retinafaceRequirementsPath}"`, {
            stdio: 'inherit',
            env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1' }
        });
    }
    
    // Create python-deps symlink for backwards compatibility
    const depsDir = path.join(pythonDistDir, 'python-deps');
    console.log('Creating python-deps symlink for backwards compatibility...');
    try {
        execSync(`rm -f "${depsDir}"`);
    } catch (_) {}
    execSync(`ln -sf yolo-env "${depsDir}"`);
    
    console.log('Virtual environments created successfully with standalone Python!');
}

async function main() {
    try {
        // Clean previous build
        if (fs.existsSync(pythonDistDir)) {
            console.log('Cleaning previous build...');
            execSync(`rm -rf ${pythonDistDir}`);
        }
        
        // Create distribution directory
        fs.mkdirSync(pythonDistDir, { recursive: true });
        
        // Download and extract standalone Python
        const pythonStandaloneDir = await downloadAndExtractPython();
        
        // Create virtual environments using standalone Python
        await createVirtualEnvironments(pythonStandaloneDir);
        
        // Get bundle size
        const getDirectorySize = (dir) => {
            try {
                const result = execSync(`du -sm ${dir}`).toString().trim();
                return result.split('\t')[0];
            } catch (error) {
                return 'unknown';
            }
        };
        
        console.log(`Bundle size: ${getDirectorySize(pythonDistDir)} MB`);
        console.log('Python bundle with standalone interpreter created successfully!');
        
    } catch (error) {
        console.error('Failed to create Python bundle:', error);
        process.exit(1);
    }
}

// Check if tar module is available
try {
    require('tar');
} catch (e) {
    console.log('Installing tar module...');
    execSync('npm install tar', { stdio: 'inherit' });
}

main();