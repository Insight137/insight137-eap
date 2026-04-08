# Contributing to Insight137 EAP

Thank you for your interest in contributing to the Entropy Attunement Protocol library.

## How to Contribute

### Reporting Issues
- Open an issue at [github.com/Insight137/insight137-eap/issues](https://github.com/Insight137/insight137-eap/issues)
- Include: Python version, NumPy version, minimal reproducible example
- For numerical discrepancies, include expected vs actual values

### Suggesting Extensions
We welcome proposals for:
- New Psi dimension formulations (with peer-reviewed grounding)
- Additional validation datasets
- Performance optimizations
- Documentation improvements
- New use case examples

### Submitting Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the full test suite: `pytest tests/ -v`
5. Verify implementation integrity: `python insight137_eap.py`
6. Submit a pull request with a clear description

## Code Standards

### Requirements for all contributions:
- Every public function must have a docstring with Args, Returns, Raises, and Reference
- Type hints on all function signatures
- Input validation for all public entry points (reject NaN, Inf, wrong types)
- No magic numbers — use named constants
- All changes must pass the existing 155-test suite with zero failures
- `verify_huang_paper()` must still return 7/7 PASS after any change

### Mathematical correctness:
- Any new computation must cite a peer-reviewed source
- Include validation against published results where possible
- Property-based tests (using hypothesis) for mathematical invariants
- Never sacrifice correctness for performance

## Testing

```bash
# Run all tests
pytest tests/ -v

# Adversarial tests only
pytest tests/test_adversarial.py -v

# Property-based tests
pytest tests/test_property_based.py -v

# Performance benchmarks
pytest tests/test_performance.py -v

# Built-in verification
python insight137_eap.py
```

## License

By contributing, you agree that your contributions will be licensed under CC BY-NC-ND 4.0, consistent with the project license.

## Questions?

Open an issue or email roger@insight137.com.
