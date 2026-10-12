"""Tests are offline by default; paid API calls require pytest --live."""

import os
import sys

import pytest

if "--live" not in sys.argv:
    os.environ["DEVMA_OFFLINE"] = "1"
    os.environ["DEEPSEEK_API_KEY"] = ""


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Allow real paid model API calls",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        for item in items:
            if "live" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="real API tests require --live")
                )


@pytest.fixture(scope="session", autouse=True)
def isolated_demo_database(tmp_path_factory):
    from data.seed_orders import seed_orders
    from devma.tools import order_query

    path = str(tmp_path_factory.mktemp("orders") / "orders.db")
    seed_orders(path)
    original = order_query.ORDER_DB_PATH
    order_query.ORDER_DB_PATH = path
    yield
    order_query.ORDER_DB_PATH = original
