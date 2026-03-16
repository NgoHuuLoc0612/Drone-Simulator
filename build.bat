@echo off
setlocal enabledelayedexpansion
title Drone Simulator v2 -- Parallel Build (SIMD + ThreadPool + NUMA)

echo ================================================================
echo   Drone Simulator v2 -- Parallel Build
echo   SIMD: AVX2+FMA auto-detected at compile time
echo   Threading: NUMA-aware work-stealing ThreadPool
echo   Swarm: M drones stepped in parallel
echo ================================================================
echo.

for %%I in ("%~dp0.") do set "ROOT=%%~sI"
set "BUILD_DIR=%ROOT%\build_win"

echo [INFO]  Project root : %~dp0
echo [INFO]  Short root   : %ROOT%
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    pause & exit /b 1
)
for /f "delims=" %%v in ('python --version 2^>^&1') do echo [INFO]  %%v

echo.
echo [1/4] Installing Python dependencies ...
python -m pip install --upgrade pip --quiet
python -m pip install ursina rasterio numpy scipy scikit-image pybind11 --quiet
if errorlevel 1 echo [WARN]  Some pip packages failed -- continuing.

echo.
echo [2/4] Locating pybind11 cmake directory ...
python -c "import pybind11,os; open('__pb11path__.tmp','w').write(pybind11.get_cmake_dir())"
if errorlevel 1 (
    echo [ERROR] pybind11 not found.
    pause & exit /b 1
)
set /p PYBIND11_LONG=<__pb11path__.tmp
del __pb11path__.tmp >nul 2>&1
for %%I in ("%PYBIND11_LONG%") do set "PYBIND11_DIR=%%~sI"
echo [INFO]  pybind11: %PYBIND11_DIR%

echo.
echo [3/4] Detecting build toolchain ...
set "GEN="
set "EXTRA="

where ninja >nul 2>&1
if not errorlevel 1 (set "GEN=Ninja" & echo [INFO] Using: Ninja & goto :do_cmake)

if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set "GEN=Visual Studio 17 2022" & set "EXTRA=-A x64" & echo [INFO] Using VS 2022 Community & goto :do_cmake)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe" (
    set "GEN=Visual Studio 17 2022" & set "EXTRA=-A x64" & echo [INFO] Using VS 2022 Professional & goto :do_cmake)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe" (
    set "GEN=Visual Studio 17 2022" & set "EXTRA=-A x64" & echo [INFO] Using VS 2022 Enterprise & goto :do_cmake)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set "GEN=Visual Studio 16 2019" & set "EXTRA=-A x64" & echo [INFO] Using VS 2019 & goto :do_cmake)

where mingw32-make >nul 2>&1
if not errorlevel 1 (set "GEN=MinGW Makefiles" & echo [INFO] Using MinGW & goto :do_cmake)

where cl >nul 2>&1
if not errorlevel 1 (set "GEN=NMake Makefiles" & echo [INFO] Using NMake & goto :do_cmake)

echo [WARN]  No toolchain detected -- CMake will auto-detect.

:do_cmake
mkdir "%BUILD_DIR%" 2>nul
echo [INFO]  Running CMake configure ...
echo.

if defined GEN (
    cmake -S "%ROOT%" -B "%BUILD_DIR%" ^
        -G "%GEN%" %EXTRA% ^
        -DCMAKE_BUILD_TYPE=Release ^
        -Dpybind11_DIR="%PYBIND11_DIR%"
) else (
    cmake -S "%ROOT%" -B "%BUILD_DIR%" ^
        -DCMAKE_BUILD_TYPE=Release ^
        -Dpybind11_DIR="%PYBIND11_DIR%"
)

if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    pause & exit /b 1
)

echo.
echo [4/4] Compiling C++ extension (AVX2+ThreadPool) ...
echo.
cmake --build "%BUILD_DIR%" --config Release

if errorlevel 1 (
    echo [ERROR] Compilation failed. Check output above.
    pause & exit /b 1
)

echo.
echo [INFO]  Searching for drone_core*.pyd ...
set "FOUND=0"
for /r "%BUILD_DIR%" %%F in (drone_core*.pyd) do (
    copy /Y "%%F" "%~dp0" >nul
    echo [INFO]  Installed: %%~nxF
    set "FOUND=1"
)
if exist "%BUILD_DIR%\Release" (
    for /r "%BUILD_DIR%\Release" %%F in (*.pyd) do (
        copy /Y "%%F" "%~dp0" >nul
        echo [INFO]  Installed: %%~nxF
        set "FOUND=1"
    )
)
if "%FOUND%"=="0" (
    echo [ERROR] No .pyd file produced.
    pause & exit /b 1
)

echo.
echo ================================================================
echo   Build complete!
echo.
echo   Quick start:
echo     python main.py
echo.
echo   Multi-drone swarm (new):
echo     python main.py --swarm 16
echo.
echo   Batch turbulence mode:
echo     python main.py --turb-batch parallel   (ThreadPool)
echo     python main.py --turb-batch simd       (AVX2 Dryden)
echo.
echo   Other options:
echo     --turb 1.2      turbulence intensity
echo     --frame hex     hexarotor
echo     --frame octo    octorotor
echo     --lod 3         terrain LOD stride
echo     --seed 999      noise seed
echo ================================================================
pause
