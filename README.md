# Remote Viewing Experiment

## Ignored Files and Directories

The following files and directories are excluded from version control:

### Dataset and Features
- `data/` - CIFAR-100 dataset directory
- `cifar_features/` - Extracted CNN and semantic features
- `*.pt` - PyTorch model/feature files
- `*.npy` - NumPy array files
- `selection_results.json` - Results from image selection
- `metadata.json` - Feature extraction metadata

### Python and Environment
- `__pycache__/` - Python bytecode cache
- `*.py[cod]` - Python compiled files
- `*.so` - C extensions
- `build/`, `dist/`, `*.egg-info/` - Build directories
- `venv/`, `ENV/`, `env/`, `.env/` - Virtual environments

### IDE and Editor
- `.idea/` - PyCharm settings
- `.vscode/` - VS Code settings
- `*.swp`, `*.swo` - Vim swap files
- `.project`, `.pydevproject` - Eclipse files

### OS-specific
- `.DS_Store` - macOS metadata
- `Thumbs.db` - Windows thumbnail cache

## Getting Started

1. Clone the repository
2. Create a virtual environment
3. Install requirements
4. Run feature extraction (will download CIFAR-100)
5. Run analysis

Note: The CIFAR-100 dataset will be automatically downloaded when running the code for the first time. Generated features and results will be stored locally but not tracked by git. 