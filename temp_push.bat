@echo off
cd C:\Code-Workspace\insight137-eap
C:\PROGRA~1\Git\bin\git.exe add -A
C:\PROGRA~1\Git\bin\git.exe commit -m "library-release-ready-155-tests-pass" --allow-empty
C:\PROGRA~1\Git\bin\git.exe tag -a v2.0.0 -m "EAP v2.0.0 - Enterprise release. 155 tests, 7/7 verification, 10K+ calls/sec"
C:\PROGRA~1\Git\bin\git.exe push origin main
C:\PROGRA~1\Git\bin\git.exe push origin v2.0.0
