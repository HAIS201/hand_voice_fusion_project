@echo off
set ROOT=%~dp0
mkdir "%ROOT%src" 2>nul
mkdir "%ROOT%work" 2>nul
mkdir "%ROOT%work\labels" 2>nul
mkdir "%ROOT%work\features\gesture" 2>nul
mkdir "%ROOT%work\features\voice" 2>nul
mkdir "%ROOT%work\features\fusion\gesture" 2>nul
mkdir "%ROOT%work\features\fusion\voice" 2>nul
echo [OK]
pause

