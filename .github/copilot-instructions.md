# Intermediate Quantitative Economics with Python - Lecture Materials

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Build the Repository
Execute these commands in order to set up a complete working environment:

**NEVER CANCEL: Each step has specific timing expectations - wait for completion.**

```bash
# 1. Set up conda environment (takes ~3 minutes)
conda env create -f environment.yml
# NEVER CANCEL: Wait 3-5 minutes for completion

# 2. Activate environment (required for all subsequent commands)
source /usr/share/miniconda/etc/profile.d/conda.sh
conda activate quantecon

# 3. Install PyTorch with CUDA support (takes ~3 minutes)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# NEVER CANCEL: Wait 3-5 minutes for completion

# 4. Install Pyro (takes ~10 seconds)
pip install pyro-ppl

# 5. Install JAX with CUDA support (takes ~30 seconds)
pip install --upgrade "jax[cuda12-local]==0.6.2"

# 6. Install NumPyro (takes ~5 seconds)
pip install numpyro

# 7. Test JAX installation (takes ~5 seconds)
python scripts/test-jax-install.py
```

### Build Commands

**CRITICAL TIMING WARNING: NEVER CANCEL BUILD COMMANDS - They take 45+ minutes to complete.**

```bash
# HTML build (primary build target) - takes 45-60 minutes
jb build lectures --path-output ./ -W --keep-going
# NEVER CANCEL: Set timeout to 90+ minutes. Executes 80+ notebooks sequentially.

# PDF build via LaTeX - takes 30-45 minutes  
jb build lectures --builder pdflatex --path-output ./ -n -W --keep-going
# NEVER CANCEL: Set timeout to 75+ minutes.

# Jupyter notebook build - takes 30-45 minutes
jb build lectures --path-output ./ --builder=custom --custom-builder=jupyter -n -W --keep-going
# NEVER CANCEL: Set timeout to 75+ minutes.
```

### Environment and Dependency Details

- **Python**: 3.12 with Anaconda 2024.10
- **Primary Framework**: Jupyter Book 1.0.3 with MyST markdown
- **Scientific Computing**: JAX 0.6.2, PyTorch (nightly), NumPyro, SciPy
- **Content Format**: MyST markdown files in `/lectures/` directory
- **Build System**: Sphinx-based via Jupyter Book
- **Execution**: Notebooks are executed during build and cached

### Validation Scenarios

**Always run these validation steps after making changes:**

1. **Environment Test**: `python scripts/test-jax-install.py` - verifies JAX/scientific stack
2. **Build Test**: Start HTML build and verify first few notebooks execute successfully
3. **Content Verification**: Check `_build/html/` directory contains expected output files
4. **Notebook Validation**: Verify generated notebooks in `_build/jupyter/` are executable

### Known Limitations and Workarounds

- **Network Access**: Intersphinx inventory warnings for external sites (intro.quantecon.org, python-advanced.quantecon.org) are expected in sandboxed environments - build continues normally
- **GPU Support**: JAX runs in CPU mode in most environments - this is expected and functional
- **Build Cache**: Uses `_build/.jupyter_cache` to avoid re-executing unchanged notebooks
- **Memory Usage**: Large notebooks may require substantial RAM during execution phase

## Project Architecture

### Key Directories
```
lectures/               # MyST markdown lecture files (80+ files)
├── _config.yml        # Jupyter Book configuration
├── _toc.yml          # Table of contents structure  
├── _static/          # Static assets (images, CSS, etc.)
└── *.md              # Individual lecture files

_build/               # Build outputs (created during build)
├── html/            # HTML website output
├── latex/           # PDF build intermediate files  
├── jupyter/         # Generated notebook files
└── .jupyter_cache/  # Execution cache

scripts/              # Utility scripts
└── test-jax-install.py  # JAX installation validator

.github/              # CI/CD workflows
└── workflows/        # GitHub Actions definitions
```

### Content Structure
- **80+ lecture files** covering intermediate quantitative economics
- **MyST markdown format** with embedded Python code blocks
- **Executable notebooks** - code is run during build process
- **Multiple output formats**: HTML website, PDF via LaTeX, downloadable notebooks

### Build Targets
1. **HTML**: Main website at `_build/html/` - primary deliverable
2. **PDF**: Single PDF document via LaTeX at `_build/latex/`
3. **Notebooks**: Individual .ipynb files at `_build/jupyter/`

## Development Workflow

### Making Changes
1. **Always activate environment first**: `conda activate quantecon`
2. **Edit lecture files**: Modify `.md` files in `/lectures/` directory
3. **Test changes**: Run quick build test on subset if possible
4. **Full validation**: Complete HTML build to verify all notebooks execute
5. **Check outputs**: Verify `_build/html/` contains expected results

### Common Tasks

**View repository structure:**
```bash
ls -la /home/runner/work/lecture-python.myst/lecture-python.myst/
# Output: .git .github .gitignore README.md _notebook_repo environment.yml lectures scripts
```

**Check lecture content:**
```bash
ls lectures/ | head -10
# Shows: intro.md, various economics topic files (.md format)
```

**Monitor build progress:**
- Build shows progress as "reading sources... [X%] filename"
- Each notebook execution time varies: 5-120 seconds per file
- Total build time: 45-60 minutes for full HTML build

**Environment verification:**
```bash
conda list | grep -E "(jax|torch|jupyter-book)"
# Should show: jax 0.6.2, torch 2.9.0.dev, jupyter-book 1.0.3
```

## Troubleshooting

### Common Issues
- **"cuInit(0) failed"**: Expected JAX warning in CPU-only environments - build continues normally
- **Intersphinx warnings**: Network inventory fetch failures are expected - build continues normally  
- **Debugger warnings**: "frozen modules" warnings during notebook execution are normal
- **Long execution times**: Some notebooks (like ar1_bayes.md) take 60+ seconds - this is normal

### Performance Notes
- **First build**: Takes longest due to fresh notebook execution
- **Subsequent builds**: Faster due to caching system
- **Cache location**: `_build/.jupyter_cache` stores execution results
- **Cache invalidation**: Changes to notebook content triggers re-execution

## CI/CD Integration

The repository uses GitHub Actions with:
- **Cache workflow**: Weekly rebuild of execution cache
- **CI workflow**: Pull request validation builds
- **Publish workflow**: Production deployment on tags

**Local builds should match CI behavior** - use same commands and expect similar timing.