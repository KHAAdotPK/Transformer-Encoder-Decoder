@REM ECHO OFF
@rem
@rem https://stackoverflow.com/questions/3155492/how-do-i-specify-the-platform-for-msbuild
@rem /p is short for /property
@rem msbuild lib\libpng\libpng.csproj /p:Configuration=Debug /p:Platform=x64
@msbuild project.xml /p:CSVPreprocessorDefinitions=yes
