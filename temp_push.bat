@echo off
cd C:\Code-Workspace\insight137-eap
"C:\PROGRA~1\Git\bin\git.exe" init
"C:\PROGRA~1\Git\bin\git.exe" add .
"C:\PROGRA~1\Git\bin\git.exe" commit -m "insight137-eap-v2.0.0-initial-release"
"C:\PROGRA~1\Git\bin\git.exe" branch -M main
"C:\PROGRA~1\Git\bin\git.exe" remote add origin https://github.com/Insight137/insight137-eap.git
"C:\PROGRA~1\Git\bin\git.exe" push -u origin main --force
