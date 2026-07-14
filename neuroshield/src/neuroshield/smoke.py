"""Environment smoke test: confirms the software MVP dependencies import and prints versions."""

import importlib


PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "neurokit2",
    "matplotlib",
    "seaborn",
    "joblib",
    "pydantic",
    "fastapi",
    "uvicorn",
    "serial",
]


def main() -> None:
    print("NeuroShield software environment smoke test")
    for name in PACKAGES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"  {name:<12} {version}")
    print("software env ok")


if __name__ == "__main__":
    main()
