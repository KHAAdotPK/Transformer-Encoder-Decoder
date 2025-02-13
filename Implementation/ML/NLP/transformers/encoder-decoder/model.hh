/*
    ML/NLP/transformers/encoder-decoder/model.hh
    Q@khaa.pk
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH

/*
    The softmax function is a mathematical transformation that converts a vector of real numbers 
    into a probability distribution. This ensures:
      - All output values lie between 0 and 1.
      - The sum of all output values is 1.

    In the context of CBOW (Continuous Bag of Words):
      - The input to softmax is the **predicted output scores** (logits) from the hidden layer.
      - These scores are obtained after taking the average of word embeddings from the context words 
        and multiplying them with the weight matrix W2.
      - The softmax function then converts these scores into probabilities, representing the likelihood 
        of each word in the vocabulary being the correct target word.

    Parameters:
      - a: Collective<T> 
        A vector of real-valued numbers representing the unnormalized logits (raw scores) 
        from the output layer of the CBOW model.

      - verbose: bool (optional, default = false)
        If true, prints intermediate steps (useful for debugging).

    Returns:
      - Collective<T>
        A probability distribution over the vocabulary, where each value represents the probability 
        of the corresponding word being the correct target word.

    The computation follows these steps:
      1. Subtract the maximum value from all elements for numerical stability.
      2. Apply the exponential function.
      3. Normalize by dividing each element by the sum of all exponentiated values.

    This ensures that the output probabilities do not suffer from floating-point precision issues 
    and remain numerically stable.
*/
template <typename T>
Collective<T> softmax(Collective<T>& a, bool verbose = false) throw (ala_exception)
{
    Collective<T> m; // max
    Collective<T> a_m; // a minus m 
    Collective<T> e_a_m; // exp over a_m
    Collective<T> s_e_a_m; // sum of e_a_m
    Collective<T> e_a_minus_max_divided_by_e_a_minus_max_sum;    

    try
    {
        m = Numcy::max(a); // Max value for numerical stability
        a_m = Numcy::subtract(a, m); // a - max(a)
        e_a_m = Numcy::exp(a_m); // exp(a - max(a))  
        s_e_a_m = Numcy::sum(e_a_m); // sum(exp(a - max(a)))
        /*
            m is max
            a_m, a minus m
            e_a_m, exp over a_m
            s_e_a_m, sum of e_a_m
         */
        /*
            Normalization step:
            Each element is divided by the sum of all exponentiated values 
            to ensure that the sum of the output probabilities is exactly 1.
         */
        e_a_minus_max_divided_by_e_a_minus_max_sum = Numcy::divide(e_a_m, s_e_a_m);     
    }
    catch(ala_exception& e)
    {        
        throw ala_exception(cc_tokenizer::String<char>("softmax() -> ") + cc_tokenizer::String<char>(e.what()));
    }
    
    return e_a_minus_max_divided_by_e_a_minus_max_sum;
}

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
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < s.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i+=2)\
    {\
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < s.getShape().getNumberOfColumns(); j++)\
        {\
            pe[i*pe.getShape().getNumberOfColumns() + j] = s[i*s.getShape().getNumberOfColumns() + j];\
        }\
    }\
}\
catch (ala_exception& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("FILL_EVEN_INDICES_OF_POSITION_ENCODING() -> ") + e.what());\
}\
}\

#define FILL_EVEN_INDICES_OF_POSITION_ENCODING_OLD(pe, s) {\
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
}\

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
    for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i < s.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i+=2)\
    {\
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < s.getShape().getNumberOfColumns(); j++)\
        {\
            /* Assign the sine function result to odd indices of the position encoding array.*/\
            pe[i*pe.getShape().getNumberOfColumns() + j] = s[i*s.getShape().getNumberOfColumns() + j];\
        }\
    }\
}\
catch (ala_exception& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("FILL_ODD_INDICES_OF_POSITION_ENCODING() -> ") + e.what());\
}\
}\

#define FILL_ODD_INDICES_OF_POSITION_ENCODING_OLD(pe, s) {\
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
}\

/*
 * ---------------------------------------------------------
 * | BUILD INPUT SEQUENCE WHEN BATCH SIZE IS SET TO A LINE |
 * ---------------------------------------------------------    
 */
/*
   Pre-trained Embeddings: For the input sequence, you use pre-trained embeddings (e.g., from Skip-gram or CBOW) to represent each token as a dense vector. 
   This is beneficial because:
   - It captures semantic relationships between words.
   - It provides a meaningful initialization for the model, especially if your dataset is small
   Why This Approach?
   - Consistency: The same word always maps to the same index and embedding, ensuring consistent representation.
   - Efficiency: Reduces memory usage by storing only one embedding per unique word.
   - Generalization: Helps the model learn patterns based on word identity rather than treating each instance as unique.
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
 *   
 * In future iterations of the codebase, consider revising the design of the parser and related entities to ensure
 * that values of similar semantics share consistent data types. This will enhance code safety and maintainability.
 */
/*
 * Builds an input sequence for a batch of tokens from a line of text.
 * 
 * This macro allocates memory and processes tokens to create an input sequence
 * using pre-trained word embeddings.
 * 
 * Parameters:
 * @is    - An output parameter of type Collective<t> that will store the final input sequence.
 * @v     - Vocabulary object that maps tokens to indices
 * @icp   - Input CSV parser object representing the input corpus, which provides token-related information.
 * @mntpl - Each input sequence is padded to ensure uniform length across variable-length sequences per line. 
 *          The value of maximum number of tokens/sequences per line (mntpl) determines the size of all input sequences. 
 *          If an input line has fewer tokens, padding is added to match the required length.
 * @t     - The data type of embeddings (e.g., float, double).
 * @w1    - Matrix of pre-trained word embeddings where each row represents a word vector.
 * 
 * Implementation:
 * 1. Allocates memory for all tokens * embedding dimension in the current line
 * 2. For each token in the line:
 *    - Looks up the token's index in the vocabulary
 *    - If found, copies the corresponding word embedding from w1
 *    - Each word embedding is a row vector from the w1 matrix
 * 
 * Error Handling:
 * - Handles memory allocation failures (bad_alloc)
 * - Handles length errors
 * - Handles custom ala_exceptions
 * - All errors are propagated with additional context
 *  
 * Note: The Vocabulary object uses internal indexing that starts at INDEX_ORIGINATES_AT_VALUE.
 *       In contrast, word embeddings use zero-based indexing (starting at 0).
 */
#define BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, v, icp, mntpl, t, w1) {\
t *ptr = NULL;\
try\
{\
    ptr = cc_tokenizer::allocator<t>().allocate(/*icp.get_total_number_of_tokens()*/ mntpl*w1.getShape().getNumberOfColumns());\
    memset(ptr, DEFAULT_PADDING_WORD_VECTOR_VALUE, mntpl*w1.getShape().getNumberOfColumns());\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < icp.get_total_number_of_tokens(); i++)\
    {\
        /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */\
        cc_tokenizer::string_character_traits<char>::size_type index = v(icp.get_token_by_number(i + 1), icp.get_current_line_number(), i + 1);\
        if (index != INDEX_NOT_FOUND_AT_VALUE)\
        {\
            /* we, Word Embedding */\
            Collective<t> we = w1.slice((index - INDEX_ORIGINATES_AT_VALUE)*w1.getShape().getNumberOfColumns(), w1.getShape().getNumberOfColumns());\
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < we.getShape().getN(); j++)\
            {\
                ptr[i*we.getShape().getN() + j] = we[j];\
            }\
        }\
    }\
}\
catch (std::bad_alloc& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE() Error: ") + cc_tokenizer::String<char>(e.what()));\
}\
catch (std::length_error& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE() Error: ") + cc_tokenizer::String<char>(e.what()));\
}\
catch (ala_exception& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE() -> ") + cc_tokenizer::String<char>(e.what()));\
}\
\
/* TODO: Eliminate the Need for Narrow Conversion */\
/* The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type', */\
/* while 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'. */\
/* Converting a signed to unsigned is a narrow conversion; it's recommended to avoid such conversions. */\
/* In future iterations, enhance code consistency by ensuring similar semantics share consistent data types.*/\
is = Collective<t>{ptr, DIMENSIONS{w1.getShape().getNumberOfColumns(), /*static_cast<cc_tokenizer::string_character_traits<char>::size_type>(icp.get_total_number_of_tokens())*/ mntpl, NULL, NULL}};\
}\

/*
 * ----------------------------------------------------------
 * | BUILD TARGET SEQUENCE WHEN BATCH SIZE IS SET TO A LINE |
 * ----------------------------------------------------------    
 */
/*
    Target Sequence:
    Token Indices: For the target sequence, you typically use token indices (integers) from the vocabulary instead of pre-trained embeddings.
    Here's why:
    - Task-Specific Learning: The target sequence is usually used for tasks like machine translation,
      text generation, or sequence prediction. The model learns to predict the next token (or sequence of tokens)
      based on the input sequence and its own internal representations.
    - Embedding Layer in Decoder: The decoder has its own embedding layer, 
      which learns to map token indices to dense vectors during training.
      This embedding layer is specific to the target vocabulary and is optimized for the task at hand. 
    - Output Layer: The decoder's output layer predicts the probability distribution over the target vocabulary.
      This is done using a softmax function, and the model is trained to minimize the cross-entropy loss between
      the predicted and actual token indices.   
 */
/* Temporary Solution to Address Compile-Time Error ("narrow conversion") */
/* ---------------------------------------------------------------------- */ 
/* If you are confident that the 'int_type' value can be safely accommodated within 'size_t' without loss of data,
 * you can use a 'static_cast' to perform the conversion. However, exercise caution when using this approach.
/* TODO: Eliminate the Need for the Following "Narrow Conversion" */
/* 
 * The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type',
 * whereas the type of 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'.
 * Converting a signed integer to its unsigned equivalent is considered a "narrow conversion,"
 * which may lead to unexpected behavior. It is advisable to avoid such conversions whenever possible.
 *
 * In future iterations of the codebase, consider revising the design of the parser and related entities to ensure
 * that values of similar semantics share consistent data types. This will enhance code safety and maintainability.
 */
/*
    @ts, An output parameter of type `Collective<t>` that will store the final target sequence.
    @v, A callable object that maps tokens to their corresponding vocabulary indices.
    @tcp, An object representing the target corpus, which provides token-related information.
    @t, The data type for storing token indices
 */
#define BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, v, tcp, t)\
{\
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
        /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */\
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
 *  ------------------------------------------------------------------------------------------------
 * | IMPORTANT NOTE: NON-DETERMINISTIC BEHAVIOR IN BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE MACRO |
 *  ------------------------------------------------------------------------------------------------
 *
 * The macro `BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE` may not produce the same output values for 
 * the same input values due to the following reason/s:
 * 1. **Floating-Point Precision**: 
 *    - The use of floating-point arithmetic (e.g., `Numcy::sin`, `Numcy::exp`, and multiplication) 
 *      can introduce small numerical errors, leading to slightly different results even for identical 
 *      inputs.
 * 2. **External Dependencies**:
 *    - The macro relies on external functions like `Numcy::arange`, `Numcy::exp`, and `Numcy::sin`, 
 *      whose implementations might not be deterministic or could depend on external state.
 */ 
/*
 *  ---------------------------------------------------------
 * | BUILD POSITION ENCODING WHEN BATCH SIZE IS A SINGLE LINE |
 *  ---------------------------------------------------------    
 */
/**
 * @brief Constructs position encoding for a batch of input sequences.
 *
 * This macro generates position encoding vectors that will be used in 
 * transformer-based models to retain positional information.
 *
 * @param p  An output parameter of type `Collective<t>` representing position indices.
 * @param is An input tensor representing the input sequence batch.
 * @param dt Division Term, an output parameter of type `Collective<t>` representing the scaling term.
 * @param dm The model's embedding dimension.
 * @param pe An output parameter of type `Collective<t>` that stores the final position encodings.
 * @param t The data type for computations (e.g., float, double).
 *
 * Functionality:
 * - Computes position indices (`p`) using `Numcy::arange()`, representing sequence positions.
 * - Computes the scaling term (`dt`) using an exponential function with a predefined scaling factor.
 * - Initializes the position encoding tensor (`pe`) with zeros.
 * - Applies sine functions to compute position encoding values.
 * - Fills even and odd indices separately using helper macros.
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t) {\
try\
{\
    /* Generate position indices: range from 0 to input sequence length(exclusive), length is the number of tokens in the line */\
    p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), (t)1.0, DIMENSIONS{1, /*is.getShape().getNumberOfColumns()*/ is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}),  DIMENSIONS{1, is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};\
        std::cout<< "p = " << p.getShape().getNumberOfColumns() << " - " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    /* Compute scaling term dt using an exponential function */\
    dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, 1, NULL, NULL}), dm), DIMENSIONS{dm, 1, NULL, NULL}};\
    /* Scale dt by a predefined scaling factor */ \
    dt = dt * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm));\
    /* Initialize position encoding tensor with zeros */\
    pe = Numcy::zeros<t>(DIMENSIONS{dm, is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL});\
        /* Please read the comments */\
        std::cout<< "dt = " << dt.getShape().getNumberOfColumns() << " - " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    /* Compute sine-transformed position encodings */\
    Collective<t> product = Numcy::sin<t>(p * dt);\
        std::cout<< "product = " << product.getShape().getNumberOfColumns() << " - " << product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    /* Fill even and odd indices of position encoding */ \
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
        t* ptr = Numcy::arange<t, t>((t)0.0, (t)is.getShape().getDimensionsOfArray().getInnerMostArrays(), (t)1.0, DIMENSIONS{1, is.getShape().getDimensionsOfArray().getInnerMostArrays(), NULL, NULL});\
        p = Collective<t>{ptr,  DIMENSIONS{1, is.getShape().getNumberOfColumns()/*[NUMCY_DIMENSIONS_SHAPE_COLUMNS]*/, NULL, NULL}};\
        /*std::cout<< ">>>> " << p.getShape().getDimensionsOfArray()[p.getShape().getDimensionsOfArray().getN() - 1] << std::endl;*/\
        dt = Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm/2, 1, NULL, NULL}), dm/2);\
        /*std::cout<< ">>>>> " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/\
        /*dt = Numcy::dot(p, dt);*/\
        dt = dt * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm));\
        pe = Numcy::zeros<t>(DIMENSIONS{dm, is.getShape().getDimensionsOfArray().getInnerMostArrays(), NULL, NULL});\
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
    @w1, vector of trained word embeddings, used as an input sequence
 */
/*#define TRAINING_LOOP_LINE_BATCH_SIZE(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v)*/\
#define TRAINING_LOOP_LINE_BATCH_SIZE(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v, w1)\
{\
    /* maximum number of tokens per line */\
    cc_tokenizer::string_character_traits<char>::size_type mntpl = icp.max_sequence_length();\
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
                BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, mntpl, t, w1);\
                std::cout<< "is = " << is.getShape().getNumberOfColumns() << " - " << is.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);\
                std::cout<< "ts = " << ts.getShape().getNumberOfColumns() << " - " << ts.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t);\
                std::cout<< "pe = " << pe.getShape().getNumberOfColumns() << " - " << pe.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                /* Encoder Input */\
                /*  pe = nx64, is = nx16 */\
                ei = Numcy::concatenate<t>(pe, is);\
                std::cout<< "ei = " << ei.getShape().getNumberOfColumns() << " - " << ei.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                /*Encoder<t> encoder(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);*/\
                Encoder<t> encoder(ei.getShape().getNumberOfColumns(), DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);\
                encoder.forward(ei);\
                std::cout<<" --------------------------------------------> " << std::endl;\
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
            /*BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, t);*/\
            /*BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);*/\
            /*BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t);*/\
            try\
            {\
                /* Encoder Input */\
                /*ei = Numcy::concatenate(pe, is);*/\
                /* Decoder Input */\
                /*di = Numcy::concatenate(pe, ts);*/\
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
                Collective<t> srcMask = Numcy::triu(Numcy::ones(DIMENSIONS{is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}), 1);\
                if (v == true)\
                {\
                    std::cout<< di.getShape().getN() << std::endl;\
                    Collective<t> bar = Numcy::ones<t>(DIMENSIONS{di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL});\
                    for (int i = 0; i < bar.getShape().getN(); i++)\
                    {\
                        std::cout<<bar[i] << " ";\
                    }\
                    std::cout<< std::endl;\
                }\
                Collective<t> tgtMask = Numcy::triu<t>(Numcy::ones(DIMENSIONS{di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), di.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}), 1);\
                Encoder<t> encoder<t>(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);\
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