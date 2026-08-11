from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "trace_to_micro"
LOWER_LAYER_PACKAGES = (
    "data_model",
    "utils",
    "replay",
    "analysis",
    "evaluation",
    "clean",
    "runtime",
)


def test_root_package_contains_only_public_entry_modules() -> None:
    modules = sorted(path.name for path in SOURCE_ROOT.glob("*.py"))

    assert modules == ["__init__.py", "cli.py", "config.py"]


def test_lower_layers_do_not_import_runner_or_cli() -> None:
    forbidden = ("trace_to_micro.runner", "trace_to_micro.cli")

    violations = []
    for package in LOWER_LAYER_PACKAGES:
        for path in (SOURCE_ROOT / package).rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            for dependency in forbidden:
                if dependency in content:
                    violations.append(
                        f"{path.relative_to(SOURCE_ROOT)} -> {dependency}"
                    )

    assert violations == []
