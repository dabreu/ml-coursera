def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark a test as integration test"
    )

    config.addinivalue_line(
        "markers", "integration_slow: mark the test as integration and slow"
    )
