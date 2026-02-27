@echo off
REM Run Anvil Level Design tests via Blender in GUI mode.
REM
REM Usage:
REM   run_tests.bat                                          Run all tests
REM   run_tests.bat test_smoke                               Run one module
REM   run_tests.bat test_smoke.SmokeTest                     Run one class
REM   run_tests.bat test_smoke.SmokeTest.test_passes         Run one test
REM   run_tests.bat --save                                   Run all, save .blend outputs
REM   run_tests.bat --save test_smoke                        Run one module, save outputs
REM
REM Set BLENDER_EXE environment variable to override the Blender path, e.g.:
REM   set BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 4.4\blender.exe

if "%BLENDER_EXE%"=="" (
    set "BLENDER_EXE=blender"
)

pushd "%~dp0"

if "%~1"=="" (
    "%BLENDER_EXE%" --enable-event-simulate --python tests/run_tests.py
) else (
    "%BLENDER_EXE%" --enable-event-simulate --python tests/run_tests.py -- %*
)

popd
