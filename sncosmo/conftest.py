import pytest


collect_ignore = ["tests/test_download_builtins.py"]


def pytest_addoption(parser):
    parser.addoption("--no-downloads", action="store_true",
                     default=False,
                     help="skip tests that might download data.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-downloads"):
        skip_test = pytest.mark.skip(
            reason="--no-downloads set"
        )
        for item in items:
            if "might_download" in item.keywords:
                item.add_marker(skip_test)
