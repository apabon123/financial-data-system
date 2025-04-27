@echo off
echo Running task setup script from tasks directory...
powershell -Command "Start-Process cmd -ArgumentList '/c tasks\setup_user_tasks.bat' -Verb RunAs"
echo.
echo If a UAC prompt appeared, please accept it to continue with task setup.
echo Once complete, you will have two scheduled tasks: VIXUpdate1 (3:50 PM CST) and VIXUpdate2 (7:00 PM CST).
pause 