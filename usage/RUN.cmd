:: -----------------------------------------------------------------
:: This script parses command-line arguments for an encoder-decoder
:: program. It supports the following options:
::  - "verbose": Enables verbose output.
::  - "e [number]": Sets the number of epochs (default is 1).
::  - "w1 [filename]": Specifies the path to the weights file.
::  - "build [verbose]": "build" command is used to compile program 
::     adding "verbose" option to the build command enables
::     conditional preprocessing.              
::
:: The script loops through arguments using SHIFT and handles each 
:: option accordingly.
:: Delayed expansion is enabled to support dynamic variable updates
:: if needed in future modifications.
:: -----------------------------------------------------------------

@echo off
setlocal enabledelayedexpansion

set build_verbose_option="CSVPreprocessorDefinitions=no"
set build_verbose_option_for_position_encoding="BuildPositionEncodingVerbose=no"
set build_verbose_option_for_target_encoding="BuildTargetEncodingVerbose=no"
set build_verbose_option_for_encoder_input="BuildEncoderInputVerbose=no"
set build_verbose_option_for_encoder_output="BuildEncoderOutputVerbose=no"
set build_verbose_option_for_input_sequence="BuildInputSequenceVerbose=no"
set build_verbose_option_for_input_sequence_target_encoding="BuildInputSequenceTargetEncodingVerbose=no"
set build_verbose_option_for_encoder_input_output="BuildEncoderInputOutputVerbose=no" 
set build_verbose_option_for_position_encoding_encoder_input="BuildPositionEncodingEncoderInputVerbose=no"
set build_verbose_option_for_decoder_input="BuildDecoderInputVerbose=no"
set temporary_stress_test_backward_in_forward_propogation="TemporaryStressTestBackwardInForwardPropogation=no"
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
) else if "%1"=="build" (
    if "%2" neq "" ( 
        if "%2"=="verbose" (            
            set build_verbose_option="CSVPreprocessorDefinitions=yes"            
        ) else if "%2"=="verbose_pe" (	
            set build_verbose_option_for_position_encoding="BuildPositionEncodingVerbose=yes"
	    ) else if "%2"=="verbose_te" (
	        set build_verbose_option_for_target_encoding="BuildTargetEncodingVerbose=yes"							
        ) else if "%2"=="stress_test_backward_in_forward_pass" (
            set temporary_stress_test_backward_in_forward_propogation="TemporaryStressTestBackwardInForwardPropogation=yes"
        ) else if "%2" == "verbose_ei" (
            set build_verbose_option_for_encoder_input="BuildEncoderInputVerbose=yes"
        ) else if "%2" == "verbose_eo" (
            set build_verbose_option_for_encoder_output="BuildEncoderOutputVerbose=yes"
        ) else if "%2" == "verbose_is" (
            set build_verbose_option_for_input_sequence="BuildInputSequenceVerbose=yes"
        ) else if "%2" == "verbose_ei_eo" (
            set build_verbose_option_for_encoder_input_output="BuildEncoderInputOutputVerbose=yes"
        ) else if "%2" == "verbose_is_te" (
            set build_verbose_option_for_input_sequence_target_encoding="BuildInputSequenceTargetEncodingVerbose=yes"            
        ) else if "%2" == "verbose_pe_ei" (
            set build_verbose_option_for_position_encoding_encoder_input="BuildPositionEncodingEncoderInputVerbose=yes"
        ) else if "%2" == "verbose_di" (
            set build_verbose_option_for_decoder_input="BuildDecoderInputVerbose=yes"            
        ) else (
            echo Unknown build option: %2
            exit /b 1
        )
        shift
    )
    shift
    goto :build
) 

goto :run

:build
@REM ECHO OFF
@rem
@rem https://stackoverflow.com/questions/3155492/how-do-i-specify-the-platform-for-msbuild
@rem /p is short for /property
@rem msbuild lib\libpng\libpng.csproj /p:Configuration=Debug /p:Platform=x64
@rem msbuild project.xml /p:CSVPreprocessorDefinitions=yes
@ msbuild project.xml /p:%build_verbose_option_for_position_encoding% /p:%build_verbose_option% /p:%temporary_stress_test_backward_in_forward_propogation% /p:%build_verbose_option_for_target_encoding% /p:%build_verbose_option_for_encoder_input% /p:%build_verbose_option_for_encoder_output% /p:%build_verbose_option_for_input_sequence% /p:%build_verbose_option_for_encoder_input_output% /p:%build_verbose_option_for_position_encoding_encoder_input% /p:%build_verbose_option_for_decoder_input% /p:%build_verbose_option_for_input_sequence_target_encoding%
goto :eof

:run
@ .\encoder-decoder.exe corpus i ./data/chat/INPUT.txt t ./data/chat/TARGET.txt e %epochs_option% bs_line w1 %w1_filename_option% %verbose_option%

:eof


 
 