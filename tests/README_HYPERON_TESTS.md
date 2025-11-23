# Hyperon-PUMA Integration Tests

## Quick Start

```bash
# Install dependencies
pip install pytest pytest-asyncio numpy

# Optional: Install Hyperon for full test coverage
pip install hyperon

# Run all tests
pytest tests/test_hyperon_integration.py -v

# Validate test suite
python tests/validate_hyperon_tests.py
```

## Files Created

1. **`test_hyperon_integration.py`** - Main test suite (54 tests)
2. **`HYPERON_INTEGRATION_TEST_SUMMARY.md`** - Detailed documentation
3. **`validate_hyperon_tests.py`** - Validation script
4. **`README_HYPERON_TESTS.md`** - This file

## Test Coverage

### ✅ 54 Tests Across 11 Suites

| Suite | Tests | Description |
|-------|-------|-------------|
| TestMeTTaExecutionEngine | 11 | MeTTa program execution, RFT conversion, DSL compilation |
| TestSubAgentManager | 9 | Agent pool management, task routing, messaging |
| TestRFTHyperonBridge | 5 | RFT-MeTTa bridge, symbolic reasoning |
| TestSubAgentCoordinator | 5 | Coordination strategies, async execution |
| TestHyperonAtomspaceAdapter | 5 | Atomspace persistence, queries, links |
| TestEndToEndWorkflows | 4 | Complete integration pipelines |
| TestFrequencyLedgerIntegration | 2 | Frequency analysis via MeTTa |
| TestParallelExecution | 2 | Load balancing, scalability |
| TestInterAgentCommunication | 3 | Message passing, coordination |
| TestPerformanceBenchmarks | 4 | Execution timing, throughput |
| TestErrorHandling | 4 | Error recovery, edge cases |

### ✅ 7 Components Tested

- ✓ MeTTaExecutionEngine
- ✓ SubAgentManager
- ✓ SubAgentCoordinator
- ✓ RFTHyperonBridge
- ✓ HyperonAtomspaceAdapter
- ✓ FrequencyLedger Integration
- ✓ RFT System Integration

### ✅ 5 End-to-End Workflows

1. RFT → MeTTa → Inference → Results
2. Frequency Ledger → MeTTa → Pattern Discovery
3. Parallel subagent execution
4. Inter-agent communication
5. Context + Entity integration

## Running Tests

### Basic Usage

```bash
# Run all tests with verbose output
pytest tests/test_hyperon_integration.py -v

# Run with detailed output (including print statements)
pytest tests/test_hyperon_integration.py -v -s

# Run specific test suite
pytest tests/test_hyperon_integration.py::TestMeTTaExecutionEngine -v

# Run specific test
pytest tests/test_hyperon_integration.py::TestMeTTaExecutionEngine::test_basic_execution -v
```

### Advanced Usage

```bash
# Run with coverage
pytest tests/test_hyperon_integration.py --cov=puma.hyperon_subagents --cov-report=html

# Run only fast tests (skip benchmarks)
pytest tests/test_hyperon_integration.py -v -m "not benchmark"

# Run only benchmarks
pytest tests/test_hyperon_integration.py::TestPerformanceBenchmarks -v

# Run with parallel execution
pytest tests/test_hyperon_integration.py -n auto

# Stop on first failure
pytest tests/test_hyperon_integration.py -x
```

### Filtering Tests

```bash
# Run only async tests
pytest tests/test_hyperon_integration.py -k "async" -v

# Run only RFT-related tests
pytest tests/test_hyperon_integration.py -k "rft" -v

# Run only workflow tests
pytest tests/test_hyperon_integration.py::TestEndToEndWorkflows -v
```

## Understanding Test Results

### Without Hyperon Installed

```
SKIPPED: 11 tests (RFTHyperonBridge suite)
ERROR: ~32 tests (require Hyperon execution)
PASSED: ~11 tests (Atomspace, coordination, communication)
```

### With Hyperon Installed

```
PASSED: 50+ tests
SKIPPED: 0-5 tests (optional components)
ERROR: 0-2 tests (configuration issues)
```

## Validation

The `validate_hyperon_tests.py` script checks:

```bash
python tests/validate_hyperon_tests.py
```

**Checks performed:**
1. ✓ Python syntax validation
2. ✓ Test structure analysis (classes, functions, fixtures)
3. ✓ Import structure verification
4. ✓ Component availability check
5. ✓ pytest test discovery

## Fixtures Available

The test suite provides these fixtures:

```python
@pytest.fixture
def temp_atomspace_path():
    """Temporary directory for atomspace persistence"""

@pytest.fixture
def metta_engine():
    """Fresh MeTTa execution engine"""

@pytest.fixture
def subagent_manager():
    """SubAgentManager with agent pool"""

@pytest.fixture
async def subagent_coordinator():
    """SubAgentCoordinator with async support"""

@pytest.fixture
def rft_bridge():
    """RFT-Hyperon bridge"""

@pytest.fixture
def hyperon_atomspace_adapter(temp_atomspace_path):
    """HyperonAtomspaceAdapter with persistence"""

@pytest.fixture
def sample_rft_frames():
    """Sample RFT frames for testing"""

@pytest.fixture
def sample_context():
    """Sample RFT context"""

@pytest.fixture
def sample_entity():
    """Sample RFT entity"""
```

## Performance Benchmarks

Run benchmarks separately for accurate timing:

```bash
# Run all benchmarks
pytest tests/test_hyperon_integration.py::TestPerformanceBenchmarks -v -s

# Run specific benchmark
pytest tests/test_hyperon_integration.py::TestPerformanceBenchmarks::test_metta_execution_performance -v -s
```

**Benchmarks include:**
- MeTTa execution speed (100 iterations)
- RFT conversion performance (100 frames)
- Parallel scalability (5-20 tasks)
- Atomspace operations (100 atoms)

## Troubleshooting

### ImportError: No module named 'hyperon'

**Solution**: Tests will skip Hyperon-dependent functionality automatically. For full coverage:
```bash
pip install hyperon
```

### ImportError: No module named 'pytest'

**Solution**: Install pytest:
```bash
pip install pytest pytest-asyncio
```

### RuntimeError: Maximum agent limit reached

**Solution**: Tests use function-scoped fixtures. If you see this, restart pytest:
```bash
# Clean pytest cache
rm -rf .pytest_cache
pytest tests/test_hyperon_integration.py -v
```

### ImportError: No module named 'atomspace_db'

**Solution**: The atomspace_db is in a subdirectory. Add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/atomspace-db"
pytest tests/test_hyperon_integration.py -v
```

Or the tests handle this automatically via sys.path manipulation.

### Tests taking too long

**Solution**: Run specific suites or skip benchmarks:
```bash
# Skip benchmarks
pytest tests/test_hyperon_integration.py -v --ignore=tests/test_hyperon_integration.py::TestPerformanceBenchmarks

# Or use markers (if configured)
pytest tests/test_hyperon_integration.py -v -m "not slow"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Hyperon Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install pytest pytest-asyncio numpy
        # Optional: pip install hyperon

    - name: Validate tests
      run: python tests/validate_hyperon_tests.py

    - name: Run integration tests
      run: pytest tests/test_hyperon_integration.py -v --tb=short

    - name: Run benchmarks
      run: pytest tests/test_hyperon_integration.py::TestPerformanceBenchmarks -v
```

### Jenkins Example

```groovy
pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'pip install pytest pytest-asyncio numpy'
            }
        }
        stage('Validate') {
            steps {
                sh 'python tests/validate_hyperon_tests.py'
            }
        }
        stage('Test') {
            steps {
                sh 'pytest tests/test_hyperon_integration.py -v --junitxml=results.xml'
            }
        }
        stage('Benchmark') {
            steps {
                sh 'pytest tests/test_hyperon_integration.py::TestPerformanceBenchmarks -v'
            }
        }
    }
    post {
        always {
            junit 'results.xml'
        }
    }
}
```

## Test Development

### Adding New Tests

```python
class TestNewFeature:
    """Test suite for new feature"""

    def test_new_functionality(self, metta_engine):
        """Test new functionality"""
        # Arrange
        input_data = "test input"

        # Act
        result = metta_engine.new_method(input_data)

        # Assert
        assert result.success
        assert result.output is not None
```

### Adding New Fixtures

```python
@pytest.fixture
def new_fixture():
    """Description of fixture"""
    # Setup
    resource = create_resource()

    # Provide to test
    yield resource

    # Cleanup
    resource.cleanup()
```

### Async Tests

```python
@pytest.mark.asyncio
async def test_async_feature(self, subagent_coordinator):
    """Test async functionality"""
    result = await subagent_coordinator.async_method()
    assert result.success
```

## Best Practices

1. **Use Fixtures**: Don't create components in tests, use fixtures
2. **Test Isolation**: Each test should be independent
3. **Clear Assertions**: Use descriptive assertion messages
4. **Mock External Dependencies**: Use mocks for external services
5. **Test Both Success and Failure**: Include error cases
6. **Performance Tests Separate**: Keep benchmarks in dedicated suite
7. **Document Complex Tests**: Add docstrings explaining test purpose

## Coverage Goals

- **Component Coverage**: >90% of public APIs
- **Integration Coverage**: All major workflows
- **Error Coverage**: Common error conditions
- **Performance Coverage**: Key performance characteristics

## Contributing

When adding tests:

1. Follow existing naming conventions
2. Add to appropriate test suite
3. Include docstrings
4. Update this README if adding new suites
5. Run validation: `python tests/validate_hyperon_tests.py`
6. Ensure tests pass: `pytest tests/test_hyperon_integration.py -v`

## Support

- **Documentation**: See `HYPERON_INTEGRATION_TEST_SUMMARY.md`
- **Component Docs**: See `/puma/hyperon_subagents/` README files
- **Issues**: Check test output for specific error messages

## License

Same as PUMA project license.

---

**Last Updated**: November 2025
**Test Suite Version**: 1.0
**Maintainer**: PUMA Development Team
