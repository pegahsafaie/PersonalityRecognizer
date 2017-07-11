@echo off
rem WINDOWS LAUNCH SCRIPT

rem ENVIRONMENT VARIABLES TO MODIFY

set JDK_PATH="C:\Program Files\Java\jdk1.6.0_01"
set WEKA="H:\apps\weka-3-4\weka.jar"

rem ----------------------------------

set COMMONS_CLI="lib\commons-cli-1.0.jar"
set JMRC="lib\jmrc.jar"

set LIBS=%WEKA%;%COMMONS_CLI%;%JMRC%;%CD%;bin\

%JDK_PATH%\bin\java -Xmx512m -classpath %LIBS% recognizer.PersonalityRecognizer %1 %2 %3 %4 %5 %6 %7 %8 %9