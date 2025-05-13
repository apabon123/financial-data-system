"""
Dependency checker for DB Inspector.

This module checks for required dependencies and provides useful error messages
when dependencies are missing.
"""

import sys
import importlib
import subprocess
import logging
from typing import Dict, List, Tuple, Optional, Set

# Setup logging
logger = logging.getLogger(__name__)

# Define required dependencies with friendly names and installation instructions
REQUIRED_DEPENDENCIES = {
    "pandas": {
        "friendly_name": "Pandas",
        "min_version": "2.0.0",
        "install_cmd": "pip install pandas>=2.0.0",
        "required_for": "All functionality"
    },
    "duckdb": {
        "friendly_name": "DuckDB",
        "min_version": "1.2.1",
        "install_cmd": "pip install duckdb>=1.2.1",
        "required_for": "Database operations"
    },
    "yaml": {
        "friendly_name": "PyYAML",
        "min_version": "6.0",
        "install_cmd": "pip install pyyaml>=6.0.1",
        "required_for": "Configuration management"
    },
    "rich": {
        "friendly_name": "Rich",
        "min_version": "13.0.0",
        "install_cmd": "pip install rich>=13.7.0",
        "required_for": "CLI user interface"
    },
    "numpy": {
        "friendly_name": "NumPy",
        "min_version": "1.24.0",
        "install_cmd": "pip install numpy>=1.24.0",
        "required_for": "Data analysis"
    },
    "matplotlib": {
        "friendly_name": "Matplotlib",
        "min_version": "3.7.0",
        "install_cmd": "pip install matplotlib>=3.7.0",
        "required_for": "Data visualization"
    },
    "seaborn": {
        "friendly_name": "Seaborn",
        "min_version": "0.12.0",
        "install_cmd": "pip install seaborn>=0.12.0",
        "required_for": "Enhanced data visualization"
    },
    "networkx": {
        "friendly_name": "NetworkX",
        "min_version": "3.0",
        "install_cmd": "pip install networkx>=3.0",
        "required_for": "Schema visualization"
    },
    "prompt_toolkit": {
        "friendly_name": "Prompt Toolkit",
        "min_version": "3.0.33",
        "install_cmd": "pip install prompt_toolkit>=3.0.33",
        "required_for": "Interactive SQL execution"
    },
    "pygments": {
        "friendly_name": "Pygments",
        "min_version": "2.15.0",
        "install_cmd": "pip install pygments>=2.15.0",
        "required_for": "Syntax highlighting"
    }
}

# Optional dependencies
OPTIONAL_DEPENDENCIES = {
    "tabulate": {
        "friendly_name": "Tabulate",
        "min_version": "0.9.0",
        "install_cmd": "pip install tabulate>=0.9.0",
        "required_for": "Enhanced table formatting"
    }
}

def check_module(module_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a module is available.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        Tuple of (is_available, version)
    """
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        if not version and module_name == "yaml":
            # PyYAML doesn't expose version as __version__
            version = getattr(module, "version", None)
            if isinstance(version, tuple):
                version = ".".join(map(str, version))
        return True, version
    except ImportError:
        return False, None

def check_dependencies(required_only: bool = True) -> Tuple[bool, List[Dict[str, str]]]:
    """
    Check for all required dependencies.
    
    Args:
        required_only: Whether to check only required dependencies (not optional ones)
        
    Returns:
        Tuple of (all_satisfied, missing_dependencies)
    """
    missing_deps = []
    all_satisfied = True
    
    # Check required dependencies
    for module_name, info in REQUIRED_DEPENDENCIES.items():
        is_available, version = check_module(module_name)
        
        if not is_available:
            missing_deps.append({
                "name": info["friendly_name"],
                "import_name": module_name,
                "required_for": info["required_for"],
                "install_cmd": info["install_cmd"]
            })
            all_satisfied = False
            logger.warning(f"Required dependency {info['friendly_name']} ({module_name}) is missing")
    
    # Check optional dependencies
    if not required_only:
        for module_name, info in OPTIONAL_DEPENDENCIES.items():
            is_available, version = check_module(module_name)
            
            if not is_available:
                missing_deps.append({
                    "name": info["friendly_name"],
                    "import_name": module_name,
                    "required_for": info["required_for"],
                    "install_cmd": info["install_cmd"],
                    "optional": True
                })
                logger.info(f"Optional dependency {info['friendly_name']} ({module_name}) is missing")
    
    return all_satisfied, missing_deps

def print_dependency_report() -> bool:
    """
    Print a report of missing dependencies with installation instructions.
    
    Returns:
        True if all required dependencies are satisfied, False otherwise
    """
    all_satisfied, missing_deps = check_dependencies(required_only=False)
    
    if not missing_deps:
        print("\nAll dependencies satisfied.")
        return True
    
    # Separate required and optional dependencies
    required_missing = [dep for dep in missing_deps if not dep.get("optional", False)]
    optional_missing = [dep for dep in missing_deps if dep.get("optional", False)]
    
    if required_missing:
        print("\n⛔ Missing Required Dependencies:")
        print("-" * 50)
        for dep in required_missing:
            print(f"• {dep['name']} ({dep['import_name']})")
            print(f"  Required for: {dep['required_for']}")
            print(f"  Install with: {dep['install_cmd']}")
            print()
    
    if optional_missing:
        print("\n⚠️ Missing Optional Dependencies:")
        print("-" * 50)
        for dep in optional_missing:
            print(f"• {dep['name']} ({dep['import_name']})")
            print(f"  Used for: {dep['required_for']}")
            print(f"  Install with: {dep['install_cmd']}")
            print()
    
    if required_missing:
        print("\n🔧 Install all missing required dependencies with:")
        print(f"pip install {' '.join([dep['import_name'] for dep in required_missing])}")
        print("\nOr install all dependencies with:")
        print("pip install -r requirements.txt")
    
    return all_satisfied

def install_dependencies(dependencies: List[str], upgrade: bool = False) -> bool:
    """
    Attempt to install missing dependencies.
    
    Args:
        dependencies: List of dependency names to install
        upgrade: Whether to upgrade existing packages
        
    Returns:
        True if installation was successful, False otherwise
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(dependencies)
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Installation successful!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error installing dependencies: {e}")
        return False