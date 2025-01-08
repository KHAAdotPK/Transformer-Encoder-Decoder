/*
    lib/NLP/transformers/encoder-decoder/model.hh
    Q@khaa.pk
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH

/*          
    @brief Computes sine and cosine values for a templated type and fills even and odd indices of the position encoding array.
            
    This code block involves the following steps: 
    1. Computes the sine values for the product of two templated types (p * dt) using Numcy::sin<t>.
    2. Creates a Collective instance named 'product' to store the sine values.
    Note: Using a separate instance is necessary to avoid potential issues related to the order of evaluation in complex expressions.
    In some cases, direct usage of Numcy::sin<t>((p * dt)) within the FILL macros might lead to unintended behavior due to the way macros handle expressions.
    3. Fills even indices of the position encoding array (@pe) with the computed sine values using the FILL_EVEN_INDICES_OF_POSITION_ENCODING macro.
    4. Fills odd indices of the position encoding array (@pe) with the same 'product' Collective instance, effectively storing the sine values in even indices and cosine values in odd indices using the FILL_ODD_INDICES_OF_POSITION_ENCODING macro.

    @param pe Position encoding array to be filled with sine and cosine values.
    @param p Templated type representing a mathematical value.
    @param dt Templated type representing another mathematical value.
 */
/*
    @brief Fills even indices of the position encoding array with values obtained from the sine function.

    This macro is designed to operate on a position encoding array (@pe) and a Collective instance (@s). The Collective instance is expected to represent the result of applying the sine function to a product of two templated types, denoted by Numcy::sin<t>((p * dt)).

    The loop iterates through each element of the Collective instance and assigns the corresponding sine function result to the even indices of the position encoding array.

    @param pe Position encoding array to be filled with values.
    @param s Collective instance representing Numcy::sin<t>((p * dt)).
*/
#define FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, s) {\
try\
{\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < s.getShape().getN(); i+=2)\
    {\
        /* Assign the sine function result to even indices of the position encoding array.*/\
        pe[i + 0] = s[i];\
    }\
}\
catch (ala_exception& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("FILL_EVEN_INDICES_OF_POSITION_ENCODING() Error: ") + e.what());\
}\
}

/*
    @brief Fills odd indices of the position encoding array with values from a Collective instance.

    This macro is designed to operate on a position encoding array (@pe) and a Collective instance (@c). The Collective instance is expected to contain values that need to be assigned to the odd indices of the position encoding array.

    The loop iterates through each element of the Collective instance and assigns the corresponding value to the odd indices of the position encoding array.

    @param pe Position encoding array to be filled with values.
    @param s Collective instance containing values to be assigned to odd indices.
*/
#define FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, s) {\
try\
{\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i < s.getShape().getN(); i+=2)\
    {\
        /* Assign the sine function result to even indices of the position encoding array.*/\
        pe[i + 0] = s[i];\
    }\
}\
catch (ala_exception& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("FILL_EVEN_INDICES_OF_POSITION_ENCODING() Error: ") + e.what());\
}\
}

/*
 * ---------------------------------------------------------
 * | BUILD INPUT SEQUENCE WHEN BATCH SIZE IS SET TO A LINE |
 * ---------------------------------------------------------    
 */
/* Temporary Solution to Address Compile-Time Error ("narrow conversion") */
/*
 * If you are confident that the 'int_type(int)' value can be safely accommodated within 'size_t' without loss of data,
 * you can use a 'static_cast' to perform the conversion. However, exercise caution when using this approach.
 */
/* TODO: Eliminate the Need for the Following "Narrow Conversion" */
/*
 * The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type',
 * whereas the type of 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'.
 * Converting a signed integer to its unsigned equivalent is considered a "narrow conversion,"
 * which may lead to unexpected behavior. It is advisable to avoid such conversions whenever possible.

 * In future iterations of the codebase, consider revising the design of the parser and related entities to ensure
 * that values of similar semantics share consistent data types. This will enhance code safety and maintainability.
 */
/*
    @is, input sequence
    @v, vocabulary, it is input vocabulary
    @icp, input csv parser
    @t, type
 */
#define BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, v, icp, t) {\
t *ptr = NULL;\
try\
{\
    ptr = cc_tokenizer::allocator<t>().allocate(icp.get_total_number_of_tokens());\
}\
catch (std::bad_alloc& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE() Error: ") + cc_tokenizer::String<char>(e.what()));\
}\
catch (std::length_error& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE() Error: ") + cc_tokenizer::String<char>(e.what()));\
}\
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < icp.get_total_number_of_tokens(); i++)\
{\
    ptr[i] = v(icp.get_token_by_number(i + 1));\
}\
/* TODO: Eliminate the Need for Narrow Conversion */\
/* The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type', */\
/* while 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'. */\
/* Converting a signed to unsigned is a narrow conversion; it's recommended to avoid such conversions. */\
/* In future iterations, enhance code consistency by ensuring similar semantics share consistent data types.*/\
is = Collective<t>{ptr, DIMENSIONS{static_cast<cc_tokenizer::string_character_traits<char>::size_type>(icp.get_total_number_of_tokens()), 1, NULL, NULL}};\
}\

/*
 * ----------------------------------------------------------
 * | BUILD TARGET SEQUENCE WHEN BATCH SIZE IS SET TO A LINE |
 * ----------------------------------------------------------    
 */
/* Temporary Solution to Address Compile-Time Error ("narrow conversion") */

/* If you are confident that the 'int_type' value can be safely accommodated within 'size_t' without loss of data,
 * you can use a 'static_cast' to perform the conversion. However, exercise caution when using this approach.

/* TODO: Eliminate the Need for the Following "Narrow Conversion" */

/* 
 * The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type',
 * whereas the type of 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'.
 * Converting a signed integer to its unsigned equivalent is considered a "narrow conversion,"
 * which may lead to unexpected behavior. It is advisable to avoid such conversions whenever possible.

 * In future iterations of the codebase, consider revising the design of the parser and related entities to ensure
 * that values of similar semantics share consistent data types. This will enhance code safety and maintainability.
 */
/*
    @ts, target sequence
    @v, vocabulary, it is target vocabulary
    @tcp, target csv parser
    @t, type
 */
#define BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, v, tcp, t) {\
t *ptr = NULL;\
try\
{\
    ptr = cc_tokenizer::allocator<t>().allocate(tcp.get_total_number_of_tokens());\
}\
catch (std::bad_alloc& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE() Error: ") + cc_tokenizer::String<char>(e.what()));\
}\
catch (std::length_error& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE() Error: ") + cc_tokenizer::String<char>(e.what()));\
}\
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < tcp.get_total_number_of_tokens(); i++)\
{\
    ptr[i] = v(tcp.get_token_by_number(i + 1));\
}\
/* TODO: Eliminate the Need for Narrow Conversion */\
/* The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type', */\
/* while 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'. */\
/* Converting a signed to unsigned is a narrow conversion; it's recommended to avoid such conversions. */\
/* In future iterations, enhance code consistency by ensuring similar semantics share consistent data types.*/\
ts = Collective<t>{ptr, DIMENSIONS{static_cast<cc_tokenizer::string_character_traits<char>::size_type>(tcp.get_total_number_of_tokens()), 1, NULL, NULL}};\
}\

/*
    @p, position an instance of Collective composite
    @is, input sequence
    @dt, division term
    @dm, dimensions of the model(d_model)
    @pe, position encoding
    @t, type
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t) {\
    try\
    {\
        p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)is.getShape().getNumberOfColumns(), (t)1.0, DIMENSIONS{1, is.getShape().getNumberOfColumns(), NULL, NULL}),  DIMENSIONS{1, is.getShape().getNumberOfColumns(), NULL, NULL}};\
        dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, 1, NULL, NULL}), dm), DIMENSIONS{dm, 1, NULL, NULL}};\
        dt = dt * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm));\
        pe = Numcy::zeros<t>(DIMENSIONS{dm, is.getShape().getNumberOfColumns(), NULL, NULL});\
        /* Please read the comments */\
        Collective<t> product = Numcy::sin<t>(p * dt);\
        FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe,  product);\
        FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, product);\
    }\
    catch (ala_exception& e)\
    {\
        throw ala_exception(cc_tokenizer::String<char>("BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE() -> ") + cc_tokenizer::String<char>(e.what()));\
    }\
}\


/*
    @p, position an instance of Collective composite
    @is, input sequence
    @dt, division term
    @dm, dimensions of the model(d_model)
    @pe, position encoding
    @t, type
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE_OLD(p, is, dt, dm, pe, t) {\
    try\
    {\
        t* ptr = Numcy::arange<t, t>((t)0.0, (t)is.getShape().getNumberOfColumns()/*[NUMCY_DIMENSIONS_SHAPE_COLUMNS]*/, (t)1.0, DIMENSIONS{1, is.getShape().getNumberOfColumns()/*[NUMCY_DIMENSIONS_SHAPE_COLUMNS]*/, NULL, NULL});\
        p = Collective<t>{ptr,  DIMENSIONS{1, is.getShape().getNumberOfColumns()/*[NUMCY_DIMENSIONS_SHAPE_COLUMNS]*/, NULL, NULL}};\
        /*std::cout<< ">>>> " << p.getShape().getDimensionsOfArray()[p.getShape().getDimensionsOfArray().getN() - 1] << std::endl;*/\
        dt = Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm/2, 1, NULL, NULL}), dm/2);\
        /*std::cout<< ">>>>> " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
        /*dt = Numcy::dot(p, dt);*/\
        dt = dt * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm));\
        pe = Numcy::zeros<t>(DIMENSIONS{dm, is.getShape().getNumberOfColumns()/*[NUMCY_DIMENSIONS_SHAPE_COLUMNS]*/, NULL, NULL});\
        /* Please read the comments */\
        Collective<t> product = Numcy::sin<t>(p * dt);\
        FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, /*Numcy::sin<t>((p * dt))*/ product);\
        FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, /*Numcy::cos<t>((p * dt))*/ product);\
    }\
    catch (ala_exception& e)\
    {\
        std::cout<< e.what() << std::endl;\
    }\
}\

/*
    @icp, input csv parser
    @tcp, target csv parser
    @ei, encoder input
    @di, decoder input
    @dm, dimensions of the model(d_model)
    @es, epochs, 
    @iv, input sequence vocabulary
    @tv, target sequence vocabulary
    @p, position
    @dt, division term
    @pe, position encoding
    @is, input sequence
    @ts, target sequence
    @t, type
    @v, be verbose when true
 */
/*#define TRAINING_LOOP_LINE_BATCH_SIZE(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v)*/\
#define TRAINING_LOOP_LINE_BATCH_SIZE(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v)\
{\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < es; i++)\
    {\
        if (v == true)\
        {\
            std::cout << "Epoch " << (i + 1) <<", batch size set to a single line and total number of lines in input vocabulary is "<<iv.get_number_of_lines()<< " and total number of lines in target vocabulary is "<<tv.get_number_of_lines()<<std::endl;\
        }\
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.get_number_of_lines(); j++)\
        {\
            icp.get_line_by_number(j + 1);\
            tcp.get_line_by_number(j + 1);\
            if (v == true)\
            {\
                std::cout << "Status of Forward Pass " << (j + 1) << ", input tokens# "<< icp.get_total_number_of_tokens() << ", target tokens# "<< tcp.get_total_number_of_tokens() << std::endl;\
            }\
            try\
            {\
                BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, t);\
                std::cout<< is.getShape().getNumberOfColumns() << " - " << is.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);\
                BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t);\
                std::cout<< pe.getShape().getNumberOfColumns() << " - " << pe.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                /* Encoder Input */\
                ei = Numcy::concatenate(pe, is);\
                BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t);\
                std::cout<< ei.getShape().getNumberOfColumns() << " - " << ei.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                Encoder<t> encoder(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);\
                encoder.forward(ei);\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< "TRAINING_LOOP_LINE_BATCH_SIZE() -> " << e.what() << std::endl;\
            }\
        }\
    }\
}\

/*
    @icp, input csv parser
    @tcp, target csv parser
    @ei, encoder input
    @di, decoder input
    @dm, dimensions of the model(d_model)
    @es, epochs, 
    @iv, input sequence vocabulary
    @tv, target sequence vocabulary
    @p, position
    @dt, division term
    @pe, position encoding
    @is, input sequence
    @ts, target sequence
    @t, type
    @v, be verbose when true
 */
/*#define TRAINING_LOOP_LINE_BATCH_SIZE(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v)*/\
#define TRAINING_LOOP_LINE_BATCH_SIZE_OLD(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v)\
{\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < es; i++)\
    {\
        if (v == true)\
        {\
            std::cout << "Epoch " << (i + 1) <<", batch size set to a single line and total number of lines in input vocabulary is "<<iv.get_number_of_lines()<< " and total number of lines in target vocabulary is "<<tv.get_number_of_lines()<<std::endl;\
        }\
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.get_number_of_lines(); j++)\
        {\
            icp.get_line_by_number(j + 1);\
            tcp.get_line_by_number(j + 1);\
            if (v == true)\
            {\
                std::cout << "Status of Forward Pass " << (j + 1) << ", input tokens# "<< icp.get_total_number_of_tokens() << ", target tokens# "<< tcp.get_total_number_of_tokens() << std::endl;\
            }\
            BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, t);\
            BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);\
            BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t);\
            try\
            {\
                /* Encoder Input */\
                ei = Numcy::concatenate(pe, is);\
                /* Decoder Input */\
                di = Numcy::concatenate(pe, ts);\
                /*Masks*/\
                /* The srcMask composite is used as masking matrix for the self-attention mechanism in the Transformer model.*/\
                /* This mask is applied to the attention scores during the self-attention computation to prevent attending to future positions in the sequence. */ \
                if (v == true)\
                {\
                    std::cout<< is.getShape().getN() << std::endl;\
                    Collective<t> foo = Numcy::ones(DIMENSIONS{is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL});\
                    for (int i = 0; i < foo.getShape().getN(); i++)\
                    {\
                        std::cout<<foo[i] << " ";\
                    }\
                    std::cout<< std::endl;\
                }\
                Collective<t> srcMask = Numcy::triu<t>(Numcy::ones(DIMENSIONS{is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}), 1);\
                if (v == true)\
                {\
                    std::cout<< di.getShape().getN() << std::endl;\
                    Collective<t> bar = Numcy::ones(DIMENSIONS{di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL});\
                    for (int i = 0; i < bar.getShape().getN(); i++)\
                    {\
                        std::cout<<bar[i] << " ";\
                    }\
                    std::cout<< std::endl;\
                }\
                Collective<t> tgtMask = Numcy::triu<t>(Numcy::ones(DIMENSIONS{di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}), 1);\
                Encoder<t> encoder(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);\
                encoder.forward(ei);\
            }\
            catch (ala_exception& e)\
            {\
                std::cout<< e.what() << std::endl;\
            }\
        }\
    }\
}\

#endif