@echo off
REM AI Interface LLM Wrapper Script for Windows

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Run the AI interface with the provided arguments
python "%SCRIPT_DIR%..\src\ai\ai_interface_llm.py" %* 