def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark a test as integration test"
    )
