# Contributing to Financial Data Management System

Thank you for your interest in contributing to the Financial Data Management System! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We want to maintain a welcoming and inclusive environment for all contributors.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up your development environment as described in [SETUP.md](SETUP.md)
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

### Testing Requirements

- Write tests for all new functionality
- Ensure all tests pass before submitting a PR
- Maintain or improve test coverage
- See the Testing section in [SETUP.md](SETUP.md) for details

Run tests using:
```bash
pytest tests/
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

## Adding a New Agent

When creating a new agent, follow these steps:

1. Use the agent template in `templates/agent_template.py`
2. Implement the required methods based on the agent's responsibility
3. Follow the single-file agent pattern
4. Add comprehensive tests in the appropriate test directory
5. Document the agent in [AGENTS.md](AGENTS.md)
6. Add usage examples to [EXAMPLES.md](EXAMPLES.md)

## Documentation

- Update documentation for any features you add or change
- Document any breaking changes clearly
- Add examples for new functionality
- Keep code comments up to date

## Reporting Issues

When reporting an issue, please include:

1. A clear description of the problem
2. Steps to reproduce the issue
3. Expected vs. actual behavior
4. System information (Python version, OS, etc.)
5. Any relevant logs or error messages

## Feature Requests

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