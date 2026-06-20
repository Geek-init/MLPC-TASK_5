# Classification Tutorial

Welcome to the classification tutorial! This tutorial will guide you through the basics for conducting classification experiments with `scikit-learn`.

The tutorial is structured into different parts

0. `00_environment_setup.md`: This document, which contains a few tips about setting up the environment and making sure that everything is setup correctly.
1. `01_classification_metrics.ipynb`: Before going into the nitty-gritty details about classification tasks, we are going to explore metrics used to evaluate the performance of a classification system. If you
2. `02_classifiers_tutorial.ipynb`: This is the main contents of the session and we explore approaching the task as a classification.

## Environment Setup

For this tutorial, we will use the same conda environment that we used in the previous assignment (`mlpc2026`). We will use the `scikit-learn` package, so you need to make sure to have it already installed. For convenience, follow the instructions that correspond to your case.

### You already created the conda environment `mlpc2026`

If you are unsure whether you already installed `scikit-learn`, try the following one-liner in your terminal:

**Mac/Linux** — open a terminal and run:
```bash
conda activate mlpc2026; python -c "exec(\"try:\\n import sklearn\\n print('scikit-learn installed, version:', sklearn.__version__)\\nexcept ImportError:\\n print('scikit-learn is NOT installed')\")"
```

**Windows** — open the **Anaconda Prompt** and run:
```bat
conda activate mlpc2026 && python -c "exec(\"try:\n import sklearn\n print('scikit-learn installed, version:', sklearn.__version__)\nexcept ImportError:\n print('scikit-learn is NOT installed')\")"
```

#### If you have already installed `scikit-learn`

Continue with the rest of this tutorial!

#### If you have not installed `scikit-learn`

```bash
# Activate the environment
conda activate mlpc2026

# Install scikit-learn and ipywidgets
pip install scikit-learn ipywidgets
```

---

### You have not created the conda environment `mlpc2026`

Before we start, create a dedicated conda environment. If you have not done so yet, download and install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

#### Mac/Linux

Open a terminal, restart it after installation, and run:

```bash
# 1. Create a new conda environment 'mlpc2026' with Python 3.12
conda create -y -n mlpc2026 -c conda-forge python=3.12 numpy pandas matplotlib jupyterlab librosa scikit-learn ipywidgets

# 2. Activate the environment
conda activate mlpc2026

# 3. Launch JupyterLab
python -m jupyterlab
```

#### Windows

Open the **Anaconda Prompt** (installed with Miniconda) and run:

```bat
:: 1. Create a new conda environment 'mlpc2026' with Python 3.12
conda create -y -n mlpc2026 -c conda-forge python=3.12 numpy pandas matplotlib jupyterlab librosa scikit-learn ipywidgets

:: 2. Activate the environment
conda activate mlpc2026

:: 3. Launch JupyterLab
python -m jupyterlab
```

> **Note for Windows users:** use the **Anaconda Prompt** (not PowerShell or cmd) to ensure conda commands are available.

**Tips:**
- Always use a dedicated environment per project to avoid dependency conflicts.

---
## Final Check: verify that all required packages are installed and display their versions.

Activate your environment and run the following code in a Python console.

```
import importlib

packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "librosa": "librosa",
    "sklearn": "scikit-learn",
    "jupyterlab": "jupyterlab",
    "ipywidgets": "ipywidgets",
}

all_ok = True
for module, display_name in packages.items():
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {display_name} {version}")
    except ImportError:
        print(f"✗ {display_name} NOT installed")
        all_ok = False

print()
print("All packages OK!" if all_ok else "Some packages are missing — check the instructions above.")
```