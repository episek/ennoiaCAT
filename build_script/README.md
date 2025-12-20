# Build Scripts - Code Quality Tools

This directory contains automated code quality analysis tools for the EnnoiaCAT project.

## Setup

Install the required tools:

```bash
pip install -r build_script/requirements.txt
```

## Available Scripts

### 1. Linting (`run_lint.py`)

Checks code style and quality using flake8 and pylint.

```bash
python build_script/run_lint.py
```

**What it checks:**
- PEP 8 style compliance
- Code formatting issues
- Common programming errors
- Code smells and anti-patterns

**Configuration:**
- Max line length: 120 characters
- Disabled warnings: Some naming conventions and complexity warnings

---

### 2. Static Analysis (`run_static_analysis.py`)

Performs type checking and security analysis.

```bash
python build_script/run_static_analysis.py
```

**What it checks:**
- Type hints and type safety (mypy)
- Security vulnerabilities (bandit)
- Common security issues like:
  - SQL injection risks
  - Shell injection
  - Hardcoded passwords
  - Insecure cryptography

**Output:**
- Console output for quick review
- `build_script/bandit_report.json` for detailed security analysis

---

### 3. Complexity Analysis (`run_complexity_analysis.py`)

Measures code complexity metrics.

```bash
python build_script/run_complexity_analysis.py
```

**What it measures:**
- **Cyclomatic Complexity (CC):** Code flow complexity
  - A (1-5): Simple, easy to test
  - B (6-10): Well structured
  - C (11-20): Slightly complex
  - D (21-30): More complex, needs refactoring
  - E (31-40): Very complex, difficult to maintain
  - F (41+): Extremely complex, high risk

- **Maintainability Index (MI):** Overall maintainability
  - A (100-20): Very high maintainability
  - B (19-10): Medium maintainability
  - C (9-0): Extremely low, needs immediate attention

- **Halstead Metrics:** Program vocabulary and difficulty
- **Raw Metrics:** Lines of code, comments, blank lines

**Output:**
- Console output with all metrics
- `build_script/cc_report.json` for cyclomatic complexity
- `build_script/mi_report.json` for maintainability index

---

### 4. Run All Checks (`run_all_checks.py`)

Runs all quality checks in sequence.

```bash
python build_script/run_all_checks.py
```

This is the recommended way to run a complete quality analysis before commits or releases.

---

## Continuous Integration

You can integrate these scripts into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Code Quality Checks
  run: |
    pip install -r build_script/requirements.txt
    python build_script/run_all_checks.py
```

## Interpreting Results

### Critical Issues (Must Fix)
- Flake8 errors (syntax, undefined names)
- Bandit high-severity security issues
- Functions with CC > 20

### Warnings (Should Review)
- Pylint warnings
- MyPy type issues
- Functions with CC 11-20
- Files with MI < 20

### Informational
- Code style suggestions
- Low-severity bandit issues
- Halstead and raw metrics

## Configuration Files

You can create configuration files in the project root for customization:

- `.flake8` - Flake8 configuration
- `.pylintrc` - Pylint configuration
- `mypy.ini` - MyPy configuration
- `.bandit` - Bandit configuration

## Best Practices

1. **Run checks before committing** - Catch issues early
2. **Fix critical issues immediately** - Don't commit code with errors
3. **Review warnings regularly** - They often indicate real problems
4. **Monitor complexity trends** - Keep functions simple (CC < 10)
5. **Maintain high MI scores** - Aim for MI > 20 on all files

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed: `pip install -r build_script/requirements.txt`
2. Check that you're running from the project root
3. Verify Python version compatibility (Python 3.7+)
4. Check file permissions if scripts won't execute

## Future Enhancements

- Unit test integration
- Code coverage reporting
- Automated formatting with black
- Pre-commit hooks
- HTML report generation
