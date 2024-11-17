/*
    src/main.hh
    Q@khaa.pk
 */

#include <iostream>

#include "../../Implementation/lib/argsv-cpp/lib/parser/parser.hh"
#include "../../Implementation/lib/corpus/corpus.hh"
#include "../../Implementation/lib/sundry/cooked_read_new.hh"

#include "../../Implementation/ML/NLP/transformers/encoder-decoder/model.hh"

#ifndef TRANSFORMERS_CODER_ENCODER_MODULE_MAIN_HH
#define TRANSFORMERS_CODER_ENCODER_MODULE_MAIN_HH

/*
    This allows you to conditionally include or exclude code based on whether macro is used as it is or prefixed/preceded by bang.
    However, it's important to note that using a "bang" (!) before a macro name is not a standard practice in C or C++ and may not behave as expected.
 */
#define MAKE_IT_VERBOSE_MAIN_HH true

/*
    Note: The delimiter used to separate the elements in the COMMAND macro can be customized.
    The first definition uses commas (,) as delimiters, while the second definition uses whitespace. 
    If you wish to change the delimiter or adjust its size, you can modify the corresponding settings in the file...
    lib/csv/parser.h or in your CMakeLists.txt.
    Alternatively, you can undefine and redefine the delimiter after including the lib/argsv-cpp/lib/parser/parser.hh 
    file according to your specific requirements.

    Please note that the macros mentioned below are by default or originally defined in the file lib/csv/parser.h
    #define GRAMMAR_END_OF_TOKEN_MARKER ","
    #define GRAMMAR_END_OF_TOKEN_MARKER_SIZE 1
    #define GRAMMAR_END_OF_LINE_MARKER "\n"
    #define GRAMMAR_END_OF_LINE_MARKER_SIZE 1

    The following two macros are defined in file  lib\argsv-cpp\lib\parser\parser.hh
    #define HELP_STR_START    "("
    #define HELP_STR_END      ")"
 */
/*
    To change the default parsing behaviour of the CSV parser
        
    Snippet from CMakeLists.txt file
    # Add a definition for the GRAMMAR_END_OF_TOKEN_MARKER macro
    #add_definitions(-DGRAMMAR_END_OF_TOKEN_MARKER=" ")
    #add_definitions(-DGRAMMAR_END_OF_TOKEN_MARKER_SIZE=1)

    Snippet from CMakeLists.txt file
    # Add a definition for the GRAMMAR_END_OF_TOKEN_MARKER macro for the replika target
    #target_compile_definitions(replika PRIVATE GRAMMAR_END_OF_TOKEN_MARKER=" ")
    #target_compile_definitions(replika PRIVATE GRAMMAR_END_OF_TOKEN_MARKER_SIZE=1)
 */
/*
    To change the default parsing behaviour of the CSV parser

    Snippet from the msbuild project file(named here project.xml)
    <ItemDefinitionGroup>
        <ClCompile>
            <PreprocessorDefinitions Condition="'$(CSVPreprocessorDefinitions)'=='yes'">CSV_EXAMPLE_APPLICATION;CSV_NOT_ALLOW_EMPTY_TOKENS;GRAMMAR_END_OF_TOKEN_MARKER=" "</PreprocessorDefinitions>
        </ClCompile>
    </ItemDefinitionGroup>  

    and then youe compile...
    @msbuild project.xml /p:CSVPreprocessorDefinitions=yes
 */

#ifdef GRAMMAR_END_OF_TOKEN_MARKER
#undef GRAMMAR_END_OF_TOKEN_MARKER
#endif
#define GRAMMAR_END_OF_TOKEN_MARKER ' '

#ifdef GRAMMAR_END_OF_LINE_MARKER
#undef GRAMMAR_END_OF_LINE_MARKER
#endif
#define GRAMMAR_END_OF_LINE_MARKER '\n'

#define COMMAND "h -h help --help ? /? (Displays help screen)\n\
v -v version --version /v (Displays version number)\n\
e epoch --epoch /e (Its a hyperparameter, sets epoch or number of times the training loop would run)\n\
corpus --corpus (Path to the file which has the training data)\n\
dmodel --dmodel (Its a hperparameter, the dimension of the model)\n\
verbose --verbose (Display of output, verbosly)\n\
lr --lr (Its a hperparameter, sets learning rate)\n\
sfc --sfc (Scaling factor constant)\n\
bs batchsize --batchsize (Its a hyperparameter, sets batch size)\n\
bs_line (Set batch size hyperparameter to line)\n\
bs_paragraph bs_para (Sets batch size hyperparameter to paragraph)\n"

#define CORPUS_COMMAND "i I -i -I input --input (Path to file which contains input sequences)\n\
t T -t -T target --target (Path to file which contains target sequences)\n"

#endif