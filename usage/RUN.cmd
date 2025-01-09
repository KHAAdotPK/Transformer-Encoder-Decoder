@echo off
setlocal enabledelayedexpansion

set verbose_option=
set w1_filename_option="./data/weights/w1p.dat"
set epochs_option=1

:start_parsing_args

if "%1"=="verbose" (
    set verbose_option=verbose
    shift
    goto :start_parsing_args
) else if "%1"=="e" (
    if "%2" neq "" (    
        set epochs_option=%2        
        shift
    ) 
    shift
    goto :start_parsing_args
) else if "%1"=="w1" (
    if "%2" neq "" (
        set w1_filename_option="%2"
        shift
    )
    shift
    goto :start_parsing_args
) 

@ .\encoder-decoder.exe corpus i ./data/chat/INPUT.txt t ./data/chat/TARGET.txt e %epochs_option% bs_line w1 %w1_filename_option% %verbose_option%

:eof


 
 