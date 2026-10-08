@echo off
chcp 65001 >nul
echo =====================================================================
echo    KHOI CHAY AD MODEL VISUALIZER DASHBOARD (100%% O D:)
echo =====================================================================
echo.
echo [0/2] Kiem tra va giai phong port cu (8000 & 5173)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo     - Dong tien trinh PID %%a tren port 8000...
    taskkill /f /t /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do (
    echo     - Dong tien trinh PID %%a tren port 5173...
    taskkill /f /t /pid %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo [1/2] Khoi dong Backend FastAPI (Port 8000)...
start "AD Visualizer Backend" cmd /k "chcp 65001 >nul && set PYTHONUTF8=1 && cd /d d:\AD\Visualize && .venv\Scripts\python.exe -X utf8 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Khoi dong Frontend Vite (Port 5173)...
start "AD Visualizer Frontend" cmd /k "cd /d d:\AD\Visualize\frontend && npm run dev"

echo.
echo Dang cho backend va frontend san sang (3 giay)...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5173

echo.
echo =====================================================================
echo    Dashboard da duoc khoi chay thanh cong tai:
echo    http://127.0.0.1:5173
echo =====================================================================
echo Luu y: Giu 2 cua so Backend va Frontend mo de dashboard hoat dong.
echo Nhan phim bat ky de dong cua so trinh khoi chay nay.
pause

