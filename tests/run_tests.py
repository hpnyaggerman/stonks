"""Dependency-free test runner: discovers and runs every ``test_*`` function in the
``test_*`` modules beside it, prints a per-test verdict, and exits non-zero on any
failure. Usable directly (``python tests/run_tests.py``) where pytest is unavailable.
"""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def main():
    modules = sorted(f[:-3] for f in os.listdir(HERE)
                     if f.startswith("test_") and f.endswith(".py"))
    passed, failed = 0, 0
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
        for name in tests:
            try:
                getattr(mod, name)()
                passed += 1
                print(f"  PASS  {mod_name}.{name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {mod_name}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
