@echo off
setlocal enabledelayedexpansion

set verbose_option=
set epochs=

:parse_args
if "%1"=="verbose" (
    set verbose_option=verbose
) else (
    if "%1" neq "" (
        set /A epochs=%1
    )
)

if "%2" neq "" (
    shift
    goto :parse_args
)

@ .\encoder-decoder.exe corpus i ./data/chat/INPUT.txt t ./data/chat/TARGET.txt e %epochs% bs_line %verbose_option%


 
 