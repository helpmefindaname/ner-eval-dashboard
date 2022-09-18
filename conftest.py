import json
from pathlib import Path
from typing import Any, Callable, Dict, no_type_check

import pytest


@pytest.fixture(scope="session", autouse=True)
@no_type_check
def pydantic_construct_validation() -> None:
    from pydantic import BaseModel

    @classmethod
    def create_model_with_validation(cls, *args, **kwargs) -> BaseModel:
        return cls(*args, **kwargs)

    BaseModel.construct = create_model_with_validation


@pytest.fixture
def testdata() -> Callable[[Any], Path]:
    base_path = Path(__file__).parent / "tests" / "data"
    return base_path.joinpath


@pytest.fixture
def test_snapshots(gen_snapshots: bool) -> Callable[[Dict[str, Any], Path], None]:
    def inner(output: Dict[str, Any], output_path: Path) -> None:

        if gen_snapshots:
            with output_path.open("w+", encoding="utf-8") as f:
                json.dump(output, f, indent=4, sort_keys=True, ensure_ascii=False)
        else:
            with output_path.open("r", encoding="utf-8") as f:
                expected = json.load(f)
            assert output == expected

    return inner


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--gen-snapshots", action="store_true", help="recreates outputs instead of testing")
    parser.addoption(
        "--run-debug-tests",
        action="store_true",
        help="run tests that are only used for debugging",
    )


def pytest_generate_tests(metafunc: Any) -> None:
    option_value = metafunc.config.option.gen_snapshots
    if "gen_snapshots" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("gen_snapshots", [option_value])


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if not config.getoption("--run-debug-tests"):
        skip_integration = pytest.mark.skip(reason="need --run-debug-tests option to run")
        for item in items:
            if "debug" in item.keywords:
                item.add_marker(skip_integration)
