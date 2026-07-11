# Contributing to CrowdWisdomTrading

Thank you for your interest! We welcome contributions that improve the
research platform's correctness, performance, or usability.

## Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/crowdwisdom_quant.git
   cd crowdwisdom_quant
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```
4. Install development dependencies:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes. Keep them focused and well-tested.
3. Run the test suite:
   ```bash
   python -m pytest tests/ -v
   ```
4. Run linting:
   ```bash
   python -m ruff check crowdwisdom_quant/
   ```
5. Commit your changes using conventional commit messages:
   ```
   feat: add strategy-specific models
   fix: correct timezone offset in macro event parsing
   docs: update walk-forward validation section
   ```

## Pull Request Process

1. Update `CHANGELOG.md` with your changes.
2. Ensure all tests pass and coverage does not decrease.
3. Open a PR against the `main` branch.
4. Include a clear description of what your PR does and why.
5. Link any related issues.

## Code Style

- **Python**: 3.12+ with type hints everywhere.
- **Formatting**: Follow PEP 8. We use `ruff` for linting.
- **Docstrings**: NumPy-style docstrings for all public APIs.
- **Imports**: Group standard library, third-party, then local.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **No dead code**: Remove commented-out code and unused imports.

## Testing

- Aim for 90%+ code coverage.
- Every bug fix should include a regression test.
- Every new feature should include unit tests for edge cases.
- Tests must not depend on external APIs (mock them).

## Questions?

Open a [GitHub Discussion](https://github.com/your-org/crowdwisdom_quant/discussions).
