@echo off
echo =====================================================================
echo    KHOI CHAY AD MODEL VISUALIZER DASHBOARD (100%% O D:)
echo =====================================================================
echo [1/2] Khoi dong Backend FastAPI (Port 8000)...
start "AD Visualizer Backend" cmd /k "cd /d d:\AD\Visualize && .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Khoi dong Frontend Vite (Port 5173)...
start "AD Visualizer Frontend" cmd /k "cd /d d:\AD\Visualize\frontend && npm run dev"

echo.
echo Dang mo trinh duyet toi http://127.0.0.1:5173 ...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5173

echo.
echo Dashboard da duoc khoi chay thanh cong!
echo Luu y: Giu 2 cua so Backend va Frontend mo de dashboard hoat dong.
echo Nhan phim bat ky de dong cua so trinh khoi chay nay.
pause
