/*
    src/main.cpp
    Q@khaa.pk
 */

/*
       THE GOAL DURING TRAINING IS FOR YOUR MODEL TO LEARN TO PREDICT THE TARGET SEQUENCE GIVEN THE INPUT SEQUENCE
    ------------------------------------------------------------------------------------------------------------------   
    The model's objective is to generate target sequences that closely match the true target sequences in the dataset.
 */

#include "main.hh"

using cc_tokenizer::allocator;

int main(int argc, char* argv[])
{
    ARG arg_bs, arg_bs_line, arg_bs_para, arg_corpus, arg_dmodel, arg_epoch, arg_help, arg_verbose;

    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
            
    if (argc < 2)
    {        
        HELP(argsv_parser, arg_help, "help");                
        HELP_DUMP(argsv_parser, arg_help); 

        return 0;                    
    }    
    
    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    FIND_ARG(argv, argc, argsv_parser, "verbose", arg_verbose);

    FIND_ARG(argv, argc, argsv_parser, "bs", arg_bs);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_bs);
    FIND_ARG(argv, argc, argsv_parser, "bs_line", arg_bs_line);
    FIND_ARG(argv, argc, argsv_parser, "bs_paragraph", arg_bs_para);
    
    FIND_ARG(argv, argc, argsv_parser, "corpus", arg_corpus);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_corpus);
    FIND_ARG(argv, argc, argsv_parser, "--dmodel", arg_dmodel);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_dmodel);
    FIND_ARG(argv, argc, argsv_parser, "epoch", arg_epoch);
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_epoch);    
        
    cc_tokenizer::String<char> input_sequence_data;
    cc_tokenizer::String<char> target_sequence_data;

    try 
    {
        if (arg_corpus.i && arg_corpus.argc)
        {
            ARG arg_input, arg_target;

            cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(CORPUS_COMMAND));

            FIND_ARG((argv + arg_corpus.i), (arg_corpus.argc + 1), argsv_parser, "input", arg_input);
            FIND_ARG_BLOCK((argv + arg_corpus.i), arg_corpus.argc + 1, argsv_parser, arg_input);
            FIND_ARG((argv + arg_corpus.i) , (arg_corpus.argc + 1), argsv_parser, "target", arg_target);
            FIND_ARG_BLOCK((argv + arg_corpus.i), (arg_corpus.argc + 1), argsv_parser, arg_target);

            if (arg_input.argc)
            {
                input_sequence_data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + arg_input.i + 1]);
            }

            if (arg_target.argc)
            {
                target_sequence_data = cc_tokenizer::cooked_read<char>(argv[arg_corpus.i + arg_target.i + 1]);
            }
        }
    }
    catch (ala_exception &e)
    {
        std::cerr <<"main(): "<<e.what()<<std::endl;
        return 0;  
    }

    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> input_csv_parser(input_sequence_data); 
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> target_csv_parser(target_sequence_data); 

    class Corpus input_sequence_vocab;
    class Corpus target_sequence_vocab;

    // instance of parser and the size of corpus in number of lines
    try
    {       
        input_sequence_vocab = Corpus(input_csv_parser/*, 13*/);
        target_sequence_vocab = Corpus(target_csv_parser/*, 13*/);
    }
    catch (ala_exception& e)
    {
        std::cerr << e.what() << std::endl;                
        return 0;
    }

    Collective<float> decoderInput;
    Collective<float> divisionTerm;
    Collective<float> encoderInput;
    Collective<float> inputSequence;
    Collective<float> position;
    Collective<float> positionEncoding;
    Collective<float> targetSequence;

    /*
        d_model
        ---------
        In the original "Attention Is All You Need" paper that introduced the Transformer architecture, the hyperparameter is referred to as "d_model" (short for "dimension of the model").
        Commonly used values for d_model range from 128 to 1024, with 256 being a frequently chosen value.
        Each word is embedded into a vector of size "dimensionsOfTheModelHyperparameter". This embedding dimension determines how much information is captured in each vector representation of a word.
        Higher dimensions can potentially allow for more expressive representations, but they also increase the computational requirements of the model.
        Smaller dimensions might lead to more compact models but might struggle to capture complex patterns in the data.
     */
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModelHyperparameter;
    if (arg_dmodel.argc)
    {
        dimensionsOfTheModelHyperparameter = std::atoi(argv[arg_dmodel.i + 1]);   
    }
    else
    {
        dimensionsOfTheModelHyperparameter = DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER;
    }
       
    try {
        if (arg_epoch.argc)
        {
            if (arg_bs_line.i) // Batch size is line
            {                
                TRAINING_LOOP_LINE_BATCH_SIZE(input_csv_parser, target_csv_parser, encoderInput, decoderInput, dimensionsOfTheModelHyperparameter, std::atoi(argv[arg_epoch.i + 1]), input_sequence_vocab, target_sequence_vocab, position, divisionTerm, positionEncoding, inputSequence, targetSequence, float, arg_verbose.i ? MAKE_IT_VERBOSE_MAIN_HH : !MAKE_IT_VERBOSE_MAIN_HH);
            }
            else if (arg_bs_para.i) // Batch size is para
            {
            }
            else if (arg_bs.argc) // Batch size is as on command line
            {
            }            
        }
        else
        {
            if (arg_bs_line.i) // Batch size is line
            {             
                TRAINING_LOOP_LINE_BATCH_SIZE(input_csv_parser, target_csv_parser, encoderInput, decoderInput, dimensionsOfTheModelHyperparameter, DEFAULT_EPOCH_HYPERPARAMETER, input_sequence_vocab, target_sequence_vocab, position, divisionTerm, positionEncoding, inputSequence, targetSequence, float, arg_verbose.i ? MAKE_IT_VERBOSE_MAIN_HH : !MAKE_IT_VERBOSE_MAIN_HH);
            }
            else if (arg_bs_para.i) // Batch size is para
            {
            }
            else if (arg_bs.argc) // Batch size is as on command line
            {
            }            
        }
    }
    catch (ala_exception& e)
    {       
       std::cerr << e.what() << '\n'; 
    }
        
    return 0;
}