@echo off
rem WINDOWS COMPILATION SCRIPT

rem ENVIRONMENT VARIABLES TO MODIFY

set JDK_PATH="C:\Program Files\Java\jdk1.6.0_01"
set WEKA="H:\apps\weka-3-4\weka.jar"

rem ----------------------------------

set COMMONS_CLI="lib\commons-cli-1.0.jar"

set LIBS=%WEKA%;%COMMONS_CLI%;%CD%;bin\

%JDK_PATH%\bin\javac -classpath %LIBS% src\recognizer\PersonalityRecognizer.java src\recognizer\Utils.java -d bin\