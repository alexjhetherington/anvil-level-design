"""Timer-based async test executor for Blender.

Replaces unittest.TextTestRunner.run() with a timer-driven executor that
yields control to Blender's event loop between steps. This allows generator
test methods to simulate real modal operators via event_simulate.

Generator protocol:
- If a test method is a generator (contains yield), the executor advances
  it one step per timer tick.
- Regular (non-generator) methods run synchronously in a single step.
- Yielded values control delay: yield 0.1 waits 100ms, bare yield uses 50ms.
"""

import sys
import traceback
import unittest
import inspect

import bpy


DEFAULT_TICK = 0.05  # 50ms between generator steps


class AsyncTestResult:
    """Collects test results and prints in unittest verbose format."""

    def __init__(self, verbosity):
        self.verbosity = verbosity
        self.successes = []
        self.failures = []   # list of (test, traceback_str)
        self.errors = []     # list of (test, traceback_str)
        self.tests_run = 0

    def addSuccess(self, test):
        self.tests_run += 1
        self.successes.append(test)
        if self.verbosity >= 2:
            print(f"{test} ... ok", flush=True)

    def addFailure(self, test, tb_str):
        self.tests_run += 1
        self.failures.append((test, tb_str))
        if self.verbosity >= 2:
            print(f"{test} ... FAIL", flush=True)

    def addError(self, test, tb_str):
        self.tests_run += 1
        self.errors.append((test, tb_str))
        if self.verbosity >= 2:
            print(f"{test} ... ERROR", flush=True)

    def wasSuccessful(self):
        return len(self.failures) == 0 and len(self.errors) == 0

    def print_summary(self):
        print()
        # Print failure/error details
        if self.failures:
            print("=" * 70)
            for test, tb_str in self.failures:
                print(f"FAIL: {test}")
                print("-" * 70)
                print(tb_str)
        if self.errors:
            print("=" * 70)
            for test, tb_str in self.errors:
                print(f"ERROR: {test}")
                print("-" * 70)
                print(tb_str)

        print("-" * 70)
        print(f"Ran {self.tests_run} test(s)")
        if self.wasSuccessful():
            print("OK")
        else:
            parts = []
            if self.failures:
                parts.append(f"failures={len(self.failures)}")
            if self.errors:
                parts.append(f"errors={len(self.errors)}")
            print(f"FAILED ({', '.join(parts)})")
        sys.stdout.flush()


def _flatten_suite(suite):
    """Recursively flatten a TestSuite into a list of TestCase instances."""
    tests = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            tests.extend(_flatten_suite(item))
        else:
            tests.append(item)
    return tests


def _is_generator_func(method):
    """Check if a method is a generator function."""
    return inspect.isgeneratorfunction(method)


def run_suite_async(suite, verbosity, on_complete):
    """Run a test suite asynchronously using Blender timers.

    Args:
        suite: unittest.TestSuite to run
        verbosity: verbosity level (0, 1, or 2)
        on_complete: callback(result) when all tests finish
    """
    tests = _flatten_suite(suite)
    result = AsyncTestResult(verbosity)

    state = {
        "index": 0,
        "phase": "setup_class",  # setup_class -> setup -> run -> teardown -> next
        "generator": None,
        "current_class": None,
    }

    def _run_setup_class(test):
        """Run setUpClass if we're entering a new test class."""
        test_class = type(test)
        if test_class != state["current_class"]:
            state["current_class"] = test_class
            if hasattr(test_class, "setUpClass"):
                test_class.setUpClass()

    def _run_setup(test):
        """Run setUp for the current test."""
        test.setUp()

    def _run_teardown(test):
        """Run tearDown for the current test."""
        test.tearDown()

    def tick():
        """Timer callback — processes one step per tick."""
        if state["index"] >= len(tests):
            # All tests done
            result.print_summary()
            on_complete(result)
            return None  # Unregister timer

        test = tests[state["index"]]
        phase = state["phase"]

        try:
            if phase == "setup_class":
                _run_setup_class(test)
                state["phase"] = "setup"
                return DEFAULT_TICK

            elif phase == "setup":
                _run_setup(test)
                # Determine if test method is a generator
                method_name = test._testMethodName
                method = getattr(test, method_name)
                if _is_generator_func(method):
                    state["generator"] = method()
                    state["phase"] = "run"
                else:
                    # Synchronous test — run in one step
                    method()
                    result.addSuccess(test)
                    state["phase"] = "teardown"
                return DEFAULT_TICK

            elif phase == "run":
                gen = state["generator"]
                try:
                    delay = next(gen)
                    if delay is None:
                        delay = DEFAULT_TICK
                    return delay
                except StopIteration:
                    # Generator finished successfully
                    state["generator"] = None
                    result.addSuccess(test)
                    state["phase"] = "teardown"
                    return DEFAULT_TICK

            elif phase == "teardown":
                _run_teardown(test)
                state["index"] += 1
                state["phase"] = "setup_class"
                return DEFAULT_TICK

        except AssertionError:
            tb_str = traceback.format_exc()
            result.addFailure(test, tb_str)
            if state["generator"]:
                state["generator"].close()
                state["generator"] = None
            # Still run teardown
            try:
                _run_teardown(test)
            except Exception:
                pass
            state["index"] += 1
            state["phase"] = "setup_class"
            return DEFAULT_TICK

        except Exception:
            tb_str = traceback.format_exc()
            result.addError(test, tb_str)
            if state["generator"]:
                state["generator"].close()
                state["generator"] = None
            try:
                _run_teardown(test)
            except Exception:
                pass
            state["index"] += 1
            state["phase"] = "setup_class"
            return DEFAULT_TICK

        return DEFAULT_TICK

    bpy.app.timers.register(tick, first_interval=DEFAULT_TICK)
