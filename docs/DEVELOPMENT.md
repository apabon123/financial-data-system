# Development Guide

This document provides comprehensive guidelines for developing and contributing to the Financial Data Management System.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Dependencies](#dependencies)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Issue Reporting](#issue-reporting)
7. [Code Review Process](#code-review-process)

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up your development environment
4. Create a new branch for your changes

## Development Workflow

### Branch Naming Convention

Use the following naming convention for branches:

- `feature/your-feature-name`: For new features
- `bugfix/issue-description`: For bug fixes
- `docs/what-you-documented`: For documentation changes
- `refactor/what-you-refactored`: For code refactoring

### Coding Standards

- Follow PEP 8 style guidelines for Python code
- Use descriptive variable and function names
- Include docstrings for all functions, classes, and modules following Google Style
- Add type hints to function signatures
- Keep functions focused on a single responsibility
- Maximum line length: 100 characters

Example function with proper documentation:

```python
def parse_query(query: str) -> Dict[str, Any]:
    """Parse a natural language query to extract parameters.
    
    Args:
        query: A natural language query string.
        
    Returns:
        Dictionary containing extracted parameters.
        
    Raises:
        ValueError: If the query cannot be parsed.
    """
    # Function implementation...
```

### Commit Message Guidelines

Follow the conventional commits specification:

- `feat: add new feature`
- `fix: resolve bug issue`
- `docs: update documentation`
- `refactor: improve code structure`
- `test: add tests for feature`
- `chore: update build scripts`

### Pull Request Process

1. Create a pull request from your feature branch to the main repository
2. Fill out the PR template with all required information
3. Ensure the CI pipeline passes on your PR
4. Request a review from at least one maintainer
5. Address any feedback or requested changes
6. Once approved, a maintainer will merge your PR

## Dependencies

### Required Dependencies

The following dependencies are required for basic functionality:

- `pandas`: Data manipulation and analysis
- `duckdb`: Database interface
- `numpy`: Numerical computing
- `pyyaml`: Configuration file parsing
- `rich`: Terminal UI formatting

### Optional Dependencies

The following dependencies provide enhanced functionality:

- **Visualization**
  - `matplotlib`: Data plotting and visualization
  - `seaborn`: Statistical data visualization

- **Schema Visualization**
  - `networkx`: Network/graph modeling and analysis

- **Interactive SQL**
  - `prompt_toolkit`: Interactive command line interfaces
  - `pygments`: Syntax highlighting

- **Table Formatting**
  - `tabulate`: Pretty-print tabular data

### Installation

#### Installing All Dependencies

To install all dependencies at once:

```bash
pip install -r requirements.txt
```

#### Installing Specific Groups

To install only core dependencies:

```bash
pip install pandas>=2.0.0 duckdb>=1.2.1 numpy>=1.24.0 pyyaml>=6.0.1 rich>=13.7.0
```

To install visualization dependencies:

```bash
pip install matplotlib>=3.7.0 seaborn>=0.12.0
```

### Checking Dependencies

#### Windows

Run the dependency checking script:

```batch
check_dependencies.bat
```

#### Linux/Mac

Run the dependency checking script:

```bash
./check_dependencies.sh
```

### Fallback Mode

If dependencies are missing, you can run DB Inspector in fallback mode which uses only the standard library:

#### Windows

```batch
DB_inspect_enhanced.bat --fallback
```

#### Linux/Mac

```bash
./db_inspect_enhanced --fallback
```

Fallback mode provides:
- Basic database information
- Table listing and schema viewing
- Simple data browsing
- Limited SQL query capabilities

### Troubleshooting Dependencies

If you encounter dependency errors:

1. Run the dependency checker to identify missing packages
2. Install missing dependencies using pip
3. If installation fails, check if your Python environment has write permissions
4. Try using a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
   pip install -r requirements.txt
   ```
5. If all else fails, use fallback mode with the `--fallback` flag

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a PR
- Maintain or improve test coverage
- Run tests using:
  ```bash
  pytest tests/
  ```

## Documentation

- Update documentation for any features you add or change
- Document any breaking changes clearly
- Add examples for new functionality
- Keep code comments up to date

### Adding a New Agent

When creating a new agent, follow these steps:

1. Use the agent template in `templates/agent_template.py`
2. Implement the required methods based on the agent's responsibility
3. Follow the single-file agent pattern
4. Add comprehensive tests in the appropriate test directory
5. Document the agent in [AGENTS.md](AGENTS.md)
6. Add usage examples to [EXAMPLES.md](EXAMPLES.md)

## Issue Reporting

When reporting an issue, please include:

1. A clear description of the problem
2. Steps to reproduce the issue
3. Expected vs. actual behavior
4. System information (Python version, OS, etc.)
5. Any relevant logs or error messages

### Feature Requests

For feature requests, please:

1. Clearly describe the feature and its benefits
2. Explain how it fits into the project's goals
3. If possible, provide a rough implementation plan

## Code Review Process

All code will be reviewed by at least one maintainer. Reviews focus on:

1. Code quality and adherence to project style
2. Test coverage and quality
3. Documentation quality
4. Performance considerations
5. Security implications

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License. 