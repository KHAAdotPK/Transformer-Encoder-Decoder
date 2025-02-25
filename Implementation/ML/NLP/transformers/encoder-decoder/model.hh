/*
    ML/NLP/transformers/encoder-decoder/model.hh
    Q@khaa.pk
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HH

/*
    I am using post-padding, where positions are initially generated based on the longest sequence in a batch.
    However, some sequences in the batch may be shorter, meaning they will have padding at the end.

    I want to set the positions of these padded tokens to zero, ensuring that only valid tokens have nonzero positions.

    At the same time, I want to avoid using zero as a valid position value for any real token. 
    This means the first valid position should start from a nonzero value (e.g., 0.1 or 1) to clearly distinguish real
    tokens from padding.
 */
/*
    Position encoding typically starts from 0 (as seen in transformers and many NLP models),
    using 0.1 instead might lead to subtle differences in how positions are represented later in the pipeline.
    
    This is intentional, make sure it aligns with overall mathematical formulation and be ready to change it back to 0.0f.

    The issue is that some sequences in the batch will be shorter than the maximum sequence length, 
    After generating positions, we use a masking mechanism to identify padded positions and manually set them to 0.
    Start at 1 instead of 0, this ensures that valid tokens always get a nonzero position value (avoiding ambiguity with padding).

    Using post-padding, the issue is that some sequences in the batch will be shorter than the maximum sequence length, and you want to make sure:
    - Padding positions get zeroed out so they don’t contribute to the model.
    - Valid positions start from a nonzero value (which is why you started at 0.1 instead of 0.0).
    you're ensuring that no valid token position is assigned 0, which could otherwise be confused with padding.
 */
#define POSITIONAL_ENCODING_START_VALUE 1.0f

enum BatchType
{
    SINGLE_LINE,
    PARAGRAPH,
    PAGE
};

template <typename t = double>
class Model 
{
    public:

        /*
         * --------------------------------------------
         * | BUILD INPUT SEQUENCE FOR ANY BATCH SIZE |
         * --------------------------------------------
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
         * @mask  - Padding tokens should not receive valid position encodings because they do not contribute to the model’s 
         *          understanding of sequence structure(padding tokens are added to make all input sequences 
         *          uniform in length). 
         *          Since positional encodings influence attention weights, allowing padding tokens to have meaningful encodings
         *          might lead to misleading attention patterns.
         *          You need a mask that differentiates real tokens from padding tokens. The mask should have:
         *          Value (DEFAULT_VALID_WORD_VECTOR_MASK_VALUE) for real tokens (to keep their positional encoding).
         *          Value (DEFAULT_PADDING_WORD_VECTOR_VALUE) for padding tokens (to zero out their positional encoding).
         * @t     - The data type of embeddings (e.g., float, double).
         * @w1    - Matrix of pre-trained word embeddings where each row represents a word vector.
         * @redundancy - Optional parameter that allows for multiple occurrences of the same token in the vocabulary.
         * @v    - Optional parameter that enables verbose output for debugging purposes.
         * 
         * Implementation:
         * 1. Allocates memory for all tokens * embedding dimension in the current line
         * 2. For each token in the line:
         *    - Looks up the token's index in the vocabulary
         *    - If found, copies the corresponding word embedding from w1
         *    - Each word embedding is a row vector from the w1 matrix
         * 3. Mask Setting in this Macro:
         *    - Memory is allocated for the mask (`ptr_mask`), with all values initially set to `DEFAULT_PADDING_WORD_VECTOR_VALUE`.
         *    - Inside the loop, when a valid token is found in the vocabulary, its corresponding mask index is set to `DEFAULT_VALID_WORD_VECTOR_MASK` (typically 1).
         *    - Padding tokens remain with their initial value (`DEFAULT_PADDING_WORD_VECTOR_VALUE`, typically 0), ensuring they are ignored in position encoding calculations.
         *    - Finally, the mask is wrapped in a `Collective<t>` object for use in downstream processing.
         *    (PLEASE NOTE:-  Implementation can ignore trailing padding, example: If all sequences are mntpl=10, but only the first 7 positions contain valid tokens, you can use sequence_length=7 instead of a mask.) 
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
        void buildInputSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& icp, CORPUS& iv, Collective<t>& is, Collective<t>& mask, Collective<t>& W1, bool redundancy = ALLOW_REDUNDANCY, bool v = false) throw (ala_exception)
        {                                                                                                                                            
            try
            {                           
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < icp.get_total_number_of_tokens(); i++)
                {
                    /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */
                    cc_tokenizer::string_character_traits<char>::size_type index = iv(icp.get_token_by_number(i + 1), icp.get_current_line_number(), i + 1, redundancy);

                    /* If this condition is false, we are no longer strictly using post-padding; instead, padding tokens may appear */
                    /* between valid tokens, leading to mixed padding. */
                    /* TODO: Investigate whether the following statement can ever evaluate to be false, because under that circumstances */
                    /* mixed padding might occur. */
                    if (index != INDEX_NOT_FOUND_AT_VALUE)
                    {
                        /* Marking real tokens with a valid mask value */
                        mask[i] = DEFAULT_VALID_WORD_VECTOR_MASK_VALUE;
            
                        /* we, Word Embedding */
                        Collective<t> we = W1.slice((index - INDEX_ORIGINATES_AT_VALUE)*W1.getShape().getNumberOfColumns(), W1.getShape().getNumberOfColumns());
                        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < we.getShape().getN(); j++)
                        {
                            is[i*we.getShape().getN() + j] = we[j];
                        }
                    }
                    else
                    {
                        /* Handling Vocabulary Lookup Failure: */
                        /* ----------------------------------- */
                        /* If the token is not found in the vocabulary (`index == INDEX_NOT_FOUND_AT_VALUE`), we must halt processing immediately and raise an exception. */
                        /* It prevents mixed-padding: If we continue processing after encountering an unknown token, padding tokens may be inserted between valid tokens instead of at the end, violating  the post-padding strategy. */
                        throw ala_exception("Model::buildInputSequence() Error: Encountered a token that is not present in the vocabulary. This should never happen if the inputs are within the expected range. Potential cause: Vocabulary is incomplete or incorrectly loaded.");
                    }                
                }
#ifdef MAKE_THIS_MODEL_VERBOSE                     
#endif                
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildInputSequence() -> ") + cc_tokenizer::String<char>(e.what()));   
            }
        }

        /*
            * @param pe An output parameter of type `Collective<t>` that stores the final position encodings.
         */
        /**
         * @brief Constructs position encoding for a batch of input sequences.
         *
         * This macro generates position encoding vectors that will be used in 
         * transformer-based models to retain positional information.
         *
         * @param p  An output parameter of type `Collective<t>` representing position indices.
         * @param pe An output parameter of type `Collective<t>` that stores the final position encodings.
         * @param dt Division Term, an output parameter of type `Collective<t>` representing the scaling term.
         * @param dm The model's embedding dimension.
         * @param is An input tensor representing the input sequence batch.
         * @param mask a mask that differentiates real tokens from padding tokens. 
         *        Padding tokens should not receive valid position encodings because they do not contribute to the model’s 
         *        understanding of sequence structure(padding tokens are added to make all input sequences 
         *        uniform in length).  
         * @param mntpl Each input sequence is padded to ensure uniform length across variable-length sequences per line.
                  The value of maximum number of tokens/sequences per line (mntpl) determines the size of all input sequences. 
         *        If an input line has fewer tokens, padding is added to match the required length. 
         *
         * Functionality:
         * - Computes position indices (`p`) using `Numcy::arange()`, representing sequence positions.
         * - Computes the scaling term (`dt`) using an exponential function with a predefined scaling factor.
         * - Initializes the position encoding tensor (`pe`) with zeros.
         * - Applies sine functions to compute position encoding values.
         * - Fills even and odd indices separately using helper macros.
         */
        /*
            m    n
            p = mntpl x 1
            mask = 1 x mntpl
            n    p           
            m x p                 
            p * mask   
         */
        void buildPositionEncoding(Collective<t>& p, Collective<t>& pe, Collective<t>& dt, cc_tokenizer::string_character_traits<char>::size_type dm, Collective<t>& is, Collective<t>& mask, cc_tokenizer::string_character_traits<char>::size_type mntpl) throw (ala_exception)
        {
            /*
                Getting ready for placement new.
                Explicitly destroy old object (optional)
             */
            p.~Collective();            
            //dt.~Collective();
            //pe.~Collective();
            
            try
            {   /*
                    Generate position indices: range from POSITIONAL_ENCODING_START_VALUE(inclusive) to input sequence-length(exclusive), sequence-length is the number of tokens in a line.
                    Placement new with Copy Construction
                 */
                new (&p) Collective<t>{Numcy::arange<t, t>((t)POSITIONAL_ENCODING_START_VALUE, (t)mntpl + (t)POSITIONAL_ENCODING_START_VALUE, (t)1.0, DIMENSIONS{1, mntpl, NULL, NULL}), DIMENSIONS{1, mntpl, NULL, NULL}};
                /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < mntpl; i++)
                {
                    p[i] = (t)POSITIONAL_ENCODING_START_VALUE + i;
                }
                p = Collective<t>{Numcy::arange<t, t>((t)POSITIONAL_ENCODING_START_VALUE, (t)mntpl + (t)POSITIONAL_ENCODING_START_VALUE, (t)1.0, DIMENSIONS{1, mntpl, NULL, NULL}),  DIMENSIONS{1, mntpl, NULL, NULL}};*/
                /*std::cout<< "p, Columns: " << p.getShape().getNumberOfColumns() << ", Rows: " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
                /*
                 * Perform element-wise multiplication between the position matrix `p` and the mask matrix `mask`.
                 *
                 * Why this is done:
                 * - The position matrix `p` contains positional encodings for each token in the sequence.
                 * - The mask matrix `mask` indicates which tokens are valid (1) and which are padding (0).
                 * - By multiplying `p` and `mask` element-wise, we ensure that positional encodings for invalid tokens
                 *   (padding) are zeroed out, while valid tokens retain their positional values.
                 *
                 * How this works:
                 * - Both `p` and `mask` have the same number of elements, but they may have different shapes.
                 * - The `[]` operator is overloaded to access elements linearly, regardless of the shapes of `p` and `mask`.
                 * - The loop iterates over each element of `p` and `mask`, multiplying them together and storing the result in `p`.
                 *
                 * Example:
                 * - If `p` is [1, 2, 3] and `mask` is [1, 1, 0], the result will be [1, 2, 0].
                 * - This ensures that the positional encoding for the third token (padding) is zeroed out.
                 *
                 * Note:
                 * - This is NOT broadcasting. Broadcasting would automatically expand the smaller array to match the shape
                 *   of the larger array, but here we are explicitly iterating over the elements and performing the
                 *   multiplication manually.
                 */                
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < mntpl; i++)
                {
                    p[i] = p[i] * mask[i];
                }

                p = p * mask;
                
                /*std::cout<< "p, Columns: " << p.getShape().getNumberOfColumns() << ", Rows: " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
                p = Numcy::transpose<t>(p);
                /* 
                    Compute scaling term dt using an exponential function. 
                    Placement new with Copy Construction does not work here...

                    new (&dt) Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)POSITIONAL_ENCODING_START_VALUE, (t)dm  + (t)POSITIONAL_ENCODING_START_VALUE, (t)2.0, DIMENSIONS{dm, mntpl, NULL, NULL}), dm), DIMENSIONS{dm, mntpl, NULL, NULL}};
                 */
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
                {
                    t value = (t)POSITIONAL_ENCODING_START_VALUE;

                    for (t j = 0; j < dm; j++)
                    {                        
                        dt[i*dt.getShape().getNumberOfColumns() + j] = std::exp(value);

                        value = value + (t)2;    
                    }
                }                
                /* Scale dt by a predefined scaling factor */
                dt = dt * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm));
                /* Compute sine-transformed position encodings */                
                Collective<t> sin_transformed_product = Numcy::sin<t>(p * dt);
                /* Fill even and odd indices separately */
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                
                std::cout<< "sin_transformed_product, Columns: " << sin_transformed_product.getShape().getNumberOfColumns() << ", Rows: " << sin_transformed_product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
#endif                
                /* Initialize position encoding tensor with zeros */
                /*
                    Placement new Requires a Constructor Call.
                    I can't directly use...
                    new (&pe) Numcy::zeros<t>(DIMENSIONS{dm, is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL});
                 */
                /*t* ptr = cc_tokenizer::allocator<t>().allocate(is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*dm);*/
                /*memset(ptr, 0, sizeof(t)*is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*dm);*/
                /*new (&pe) Collective<t>{ptr, DIMENSIONS{dm, is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};*/
                //pe = Collective<t>{ptr, DIMENSIONS{dm, is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL}};
                //FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, sin_transformed_product);
                /* Fill even and odd indices separately */
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < pe.getShape().getN(); i+=2)
                {
                    pe[i] = sin_transformed_product[i];
                }
                for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i < pe.getShape().getN(); i+=2)
                {
                    pe[i] = sin_transformed_product[i];
                }
            }
            catch (std::bad_alloc& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding() Error: ") + cc_tokenizer::String<char>(e.what()));
            }
            catch (std::length_error& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding() Error: ") + cc_tokenizer::String<char>(e.what()));
            }            
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding() -> ") + cc_tokenizer::String<char>(e.what()));
            }
        }

        /*
         * --------------------------------------------
         * | BUILD TARGET SEQUENCE FOR ANY BATCH SIZE |
         * --------------------------------------------
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
            @tcp, An object representing the target corpus, which provides token-related information.
            @tv, A callable object that maps tokens to their corresponding vocabulary indices.
            @ts, An output parameter of type `Collective<t>` that will store the final target sequence.            
            @v, Display output, verbosly.
         */        
        void buildTragetSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& tcp, CORPUS& tv, Collective<t>& ts, bool verbose = false) throw (ala_exception)
        {
            // Getting ready for placement new.
            ts.~Collective();

            t *ptr = NULL;
            try
            {
                ptr = cc_tokenizer::allocator<t>().allocate(tcp.get_total_number_of_tokens());\
            }
            catch (std::bad_alloc& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildTragetSquence() Error: ") + cc_tokenizer::String<char>(e.what()));\
            }
            catch (std::length_error& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildTargetSequence() Error: ") + cc_tokenizer::String<char>(e.what()));\
            }
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < tcp.get_total_number_of_tokens(); i++)
            {
                /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */
                cc_tokenizer::string_character_traits<char>::size_type index = tv(tcp.get_token_by_number(i + 1));

                if (index != INDEX_NOT_FOUND_AT_VALUE)
                {                    
                    ptr[i] = tv(tcp.get_token_by_number(i + 1));
                }
                else
                {
                    /*
                        Handling Vocabulary Lookup Failure: 
                        -----------------------------------
                        If the token is not found in the vocabulary (`index == INDEX_NOT_FOUND_AT_VALUE`), we must halt processing immediately and raise an exception.                        
                     */                         
                    throw ala_exception("Model::buildTargetSequence() Error: Encountered a token that is not present in the vocabulary. This should never happen if the inputs are within the expected range. Potential cause: Vocabulary is incomplete or incorrectly loaded."); 
                }
            }

            /* 
                TODO: Eliminate the Need for Narrow Conversion 
                The return type of 'get_total_number_of_tokens()' is 'cc_tokenizer::string_character_traits<char>::int_type',
                while 'DIMENSIONS::columns' is 'cc_tokenizer::string_character_traits<char>::size_type'. 
                Converting a signed to unsigned is a narrow conversion; it's recommended to avoid such conversions. 
                In future iterations, enhance code consistency by ensuring similar semantics share consistent data types.
             */        
            new (&ts) Collective<t>{ptr, DIMENSIONS{static_cast<cc_tokenizer::string_character_traits<char>::size_type>(tcp.get_total_number_of_tokens()), 1, NULL, NULL}};
        }
    
        /*
            @es, epochs
            @iv, input sequence vocabulary
            @tv, target sequence vocabulary
            @icp, input csv parser
            @tcp, target csv parser
            @is, input sequence
            @ts, target sequence
            @p, position
            @pe, position encoding
            @dm, dimensions of the model(d_model)
            @dt, division term
            @ei, encoder input
            @di, decoder input
            @w1, vector of trained word embeddings, used as an input sequence            
            @batch, batch type 
            @v, be verbose when true                                                                                                                                                           
         */
        void startTraining(cc_tokenizer::string_character_traits<char>::size_type es, CORPUS& iv, CORPUS& tv, cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& icp, cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& tcp, Collective<t>& is, Collective<t>& ts, Collective<t>& p, Collective<t>& pe, cc_tokenizer::string_character_traits<char>::size_type dm, Collective<t>& dt, Collective<t>& ei, Collective<t>& di, Collective<t>& W1, BatchType batch = SINGLE_LINE, bool v = false) throw (ala_exception)
        {
            switch (batch)
            {
                case SINGLE_LINE:
                {
                    /* maximum number of tokens per line(number of tokens in the largest line of input text) */
                    cc_tokenizer::string_character_traits<char>::size_type mntpl = icp.max_sequence_length();

                    t* ptr = NULL;
                    Collective<t> mask;

                    try
                    {   
                        /* 
                             Post-Padding instead of Pre-Padding or Mixed-Padding 
                            ------------------------------------------------------
                            Here, using post-padding by allocating memory for a fixed-length sequence (mntpl) and filling it with real tokens first, 
                            followed by padding. This means that all padding appears at the end, not in the middle or mixed with real tokens. 
                         */

                        /*ptr = cc_tokenizer::allocator<t>().allocate(mntpl); 

                        memset(ptr, 0, sizeof(t)*mntpl);
                        p = Collective<t>{ptr, DIMENSIONS{1, mntpl, NULL, NULL}};*/

                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl*W1.getShape().getNumberOfColumns());

                        memset (ptr, 0, sizeof(t)*(mntpl*W1.getShape().getNumberOfColumns()));
                        is = Collective<t>{ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), mntpl, NULL, NULL}};

                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl);

                        memset(ptr, 0, sizeof(t)*mntpl);
                        mask = Collective<t>{ptr, DIMENSIONS{mntpl, 1, NULL, NULL}};

                        /* Allocate enough memory for scaling term dt and reset it to zeros */
                        ptr = cc_tokenizer::allocator<t>().allocate(dm*mntpl);

                        memset(ptr, 0, sizeof(t)*dm*mntpl);
                        dt = Collective<t>{ptr, DIMENSIONS{dm, mntpl, NULL, NULL}};

                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl*dm); 

                        memset(ptr, 0, sizeof(t)*mntpl*dm);
                        pe = Collective<t>{ptr, DIMENSIONS{dm, mntpl, NULL, NULL}};
                        
                        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < es; i++)
                        {
                            if (v)
                            {                        
                                std::cout<< "Epoch " << (i + 1) << ", batch size set to a single line and total number of lines in input vocabulary is " << icp.get_total_number_of_lines()<< " and total number of lines in target vocabulary is " << tcp.get_total_number_of_lines() << std::endl;
                            }

                            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.get_number_of_lines(); j++)
                            {
                                icp.get_line_by_number(j + 1);
                                tcp.get_line_by_number(j + 1);

                                if (v)
                                {
                                    std::cout << "Status of Forward Pass " << (j + 1) << ", input tokens# "<< icp.get_total_number_of_tokens() << ", target tokens# "<< tcp.get_total_number_of_tokens() << std::endl;
                                }

                                buildInputSequence(icp, iv, is, mask, W1, !ALLOW_REDUNDANCY);
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                                
                                std::cout<< "Number of tokens in this line: " << icp.get_total_number_of_tokens() << std::endl; 
                                std::cout<< "::: DEBUG DATA -: Model::buildInputSequence() :- :::"  << std::endl;
                                std::cout<< "is(Input Sequence), Columns: " << is.getShape().getNumberOfColumns() << ", Rows: " << is.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;                                
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < is.getShape().getN(); k++)
                                {
                                    std::cout<< is[(k/is.getShape().getNumberOfColumns())*is.getShape().getNumberOfColumns() + (k%is.getShape().getNumberOfColumns())] << " ";

                                    if ((k + 1)%is.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
                                std::cout<< "mask, Columns: " << mask.getShape().getNumberOfColumns() << ", Rows: " << mask.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < mask.getShape().getN(); k++)
                                {
                                    std::cout<< mask[(k/mask.getShape().getNumberOfColumns())*mask.getShape().getNumberOfColumns() + (k%mask.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%mask.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
#endif                                
                                buildTragetSequence(tcp, tv, ts, v);
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                                
                                std::cout<< "::: DEBUG DATA -: Model::buildTargetSequence() :- :::"  << std::endl;
#endif                                
                                buildPositionEncoding(p, pe, dt, dm, is, mask, mntpl);
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                                
                                std::cout<< "::: DEBUG DATA -: (Model::buildPositionEncoding()) for Position Encoding) :- :::"  << std::endl;                                                                                                                                                                
                                std::cout<< "Transposed(p * mask), Columns: " << p.getShape().getNumberOfColumns() << ", Rows: " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;                                                                
                                for (int k = 0; k < p.getShape().getN(); k++)
                                {
                                    std::cout<< p[(k/p.getShape().getNumberOfColumns())*p.getShape().getNumberOfColumns() + (k%p.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%p.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
                                std::cout<< "dt * SCALING_FACTOR, Columns: " << dt.getShape().getNumberOfColumns() << ", Rows: " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < dt.getShape().getN(); k++)
                                {
                                    std::cout<< dt[(k/dt.getShape().getNumberOfColumns())*dt.getShape().getNumberOfColumns() + (k%dt.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%dt.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
                                std::cout<< "pe, Columns: " << pe.getShape().getNumberOfColumns() << ", Rows: " << pe.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < pe.getShape().getN(); k++)
                                {
                                    std::cout<< pe[(k/pe.getShape().getNumberOfColumns())*pe.getShape().getNumberOfColumns() + (k%pe.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%pe.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }                                
#endif                          
                                ei = Numcy::concatenate(pe, is); 
                                                                
                                std::cout<< "::: DEBUG DATA -: Encoder Input(ei) :- :::"  << std::endl;
                                std::cout<< "Columns: " << ei.getShape().getNumberOfColumns() << ", Rows: " << ei.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < ei.getShape().getN(); k++)
                                {
                                    std::cout<< ei[(k/ei.getShape().getNumberOfColumns())*ei.getShape().getNumberOfColumns() + (k%ei.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%ei.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }

                                Encoder<t> encoder(ei.getShape().getNumberOfColumns(), DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);
                                encoder.forward(ei);                                
                                std::cout<< "*++++++++++++++++++++++++++++++++++++++*" << std::endl;

                                /* Reinitialize, input sequence and input sequence mask */
                                /*for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < p.getShape().getN(); k++)
                                {
                                    p[k] = 0;
                                }*/
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < is.getShape().getN(); k++)
                                {
                                    is[k] = 0;
                                }
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < mask.getShape().getN(); k++)
                                {
                                    mask[k] = 0;
                                }
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < dt.getShape().getN(); k++)
                                {
                                    dt[k] = 0;
                                }
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < pe.getShape().getN(); k++)
                                {
                                    pe[k] = 0;
                                }
                            }
                        }
                    }
                    catch (std::bad_alloc& e)
                    {
                        throw ala_exception(cc_tokenizer::String<char>("Model::startTrining() Error: ") + cc_tokenizer::String<char>(e.what()));
                    }
                    catch (std::length_error& e)
                    {
                        throw ala_exception(cc_tokenizer::String<char>("Model::startTrining() Error: ") + cc_tokenizer::String<char>(e.what()));\
                    }
                    catch (ala_exception& e)
                    {
                        throw ala_exception(cc_tokenizer::String<char>("Model::startTrining() -> ") + cc_tokenizer::String<char>(e.what()));                     
                    }
                }    
                break;

                default:
                break;
            }
        }
};

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
 * @mask  - Padding tokens should not receive valid position encodings because they do not contribute to the model’s 
 *          understanding of sequence structure(padding tokens are added to make all input sequences 
 *          uniform in length). 
 *          Since positional encodings influence attention weights, allowing padding tokens to have meaningful encodings
 *          might lead to misleading attention patterns.
 *          You need a mask that differentiates real tokens from padding tokens. The mask should have:
 *          Value (DEFAULT_VALID_WORD_VECTOR_MASK_VALUE) for real tokens (to keep their positional encoding).
 *          Value (DEFAULT_PADDING_WORD_VECTOR_VALUE) for padding tokens (to zero out their positional encoding).
 * @t     - The data type of embeddings (e.g., float, double).
 * @w1    - Matrix of pre-trained word embeddings where each row represents a word vector.
 * 
 * Implementation:
 * 1. Allocates memory for all tokens * embedding dimension in the current line
 * 2. For each token in the line:
 *    - Looks up the token's index in the vocabulary
 *    - If found, copies the corresponding word embedding from w1
 *    - Each word embedding is a row vector from the w1 matrix
 * 3. Mask Setting in this Macro:
 *    - Memory is allocated for the mask (`ptr_mask`), with all values initially set to `DEFAULT_PADDING_WORD_VECTOR_VALUE`.
 *    - Inside the loop, when a valid token is found in the vocabulary, its corresponding mask index is set to `DEFAULT_VALID_WORD_VECTOR_MASK` (typically 1).
 *    - Padding tokens remain with their initial value (`DEFAULT_PADDING_WORD_VECTOR_VALUE`, typically 0), ensuring they are ignored in position encoding calculations.
 *    - Finally, the mask is wrapped in a `Collective<t>` object for use in downstream processing.
 *    (PLEASE NOTE:-  Implementation can ignore trailing padding, example: If all sequences are mntpl=10, but only the first 7 positions contain valid tokens, you can use sequence_length=7 instead of a mask.) 
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
#define BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, v, icp, mntpl, mask, t, w1) {\
t *ptr = NULL, *ptr_mask = NULL;\
try\
{\
    ptr = cc_tokenizer::allocator<t>().allocate(/*icp.get_total_number_of_tokens()*/ mntpl*w1.getShape().getNumberOfColumns());\
    /* Post-Padding instead of Pre-Padding or Mixed-Padding */\
    /* Here, using post-padding by allocating memory for a fixed-length sequence (mntpl) and filling it with real tokens first, */\
    /* followed by padding. This means that all padding appears at the end, not in the middle or mixed with real tokens. */\
    ptr_mask = cc_tokenizer::allocator<t>().allocate(mntpl);\
    memset(ptr, (t)DEFAULT_PADDING_WORD_VECTOR_VALUE, mntpl*w1.getShape().getNumberOfColumns());\
    memset(ptr_mask, DEFAULT_PADDING_WORD_VECTOR_VALUE, mntpl);\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < icp.get_total_number_of_tokens(); i++)\
    {\
        /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */\
        cc_tokenizer::string_character_traits<char>::size_type index = v(icp.get_token_by_number(i + 1), icp.get_current_line_number(), i + 1);\
        /* If this condition is false, we are no longer strictly using post-padding; instead, padding tokens may appear */\
        /* between valid tokens, leading to mixed padding. */\
        /* TODO: Investigate whether the following statement can ever evaluate to be false, because under that circumstances */\
        /* mixed padding might occur. */\
        if (index != INDEX_NOT_FOUND_AT_VALUE)\
        {\
            /* Marking real tokens with a valid mask value */\
            ptr_mask[i] = DEFAULT_VALID_WORD_VECTOR_MASK_VALUE;\
            \
            /* we, Word Embedding */\
            Collective<t> we = w1.slice((index - INDEX_ORIGINATES_AT_VALUE)*w1.getShape().getNumberOfColumns(), w1.getShape().getNumberOfColumns());\
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < we.getShape().getN(); j++)\
            {\
                ptr[i*we.getShape().getN() + j] = we[j];\
            }\
        }\
        else\
        {\
            /* Handling Vocabulary Lookup Failure: */\
            /* ----------------------------------- */\
            /* If the token is not found in the vocabulary (`index == INDEX_NOT_FOUND_AT_VALUE`), we must halt processing immediately and raise an exception. */\
            /* It prevents mixed-padding: If we continue processing after encountering an unknown token, padding tokens may be inserted between valid tokens instead of at the end, violating  the post-padding strategy. */\
            throw ala_exception("BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE() Error: Encountered a token that is not present in the vocabulary. This should never happen if the inputs are within the expected range. Potential cause: Vocabulary is incomplete or incorrectly loaded.");\
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
/* Assigning the mask to ensure padding tokens (0) do not receive position encodings */\
mask = Collective<t>{ptr_mask, DIMENSIONS{mntpl, 1, NULL, NULL}};\
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
 * @param mntpl Each input sequence is padded to ensure uniform length across variable-length sequences per line. 
 *        The value of maximum number of tokens/sequences per line (mntpl) determines the size of all input sequences. 
 *        If an input line has fewer tokens, padding is added to match the required length. 
 * @param mask a mask that differentiates real tokens from padding tokens. 
 *        Padding tokens should not receive valid position encodings because they do not contribute to the model’s 
 *        understanding of sequence structure(padding tokens are added to make all input sequences 
 *        uniform in length).
 * @param t The data type for computations (e.g., float, double).
 *
 * Functionality:
 * - Computes position indices (`p`) using `Numcy::arange()`, representing sequence positions.
 * - Computes the scaling term (`dt`) using an exponential function with a predefined scaling factor.
 * - Initializes the position encoding tensor (`pe`) with zeros.
 * - Applies sine functions to compute position encoding values.
 * - Fills even and odd indices separately using helper macros.
 */
/*
           m    n
    p = mntpl x 1
    mask = 1 x mntpl
           n    p
           
    m x p      
           
    p * mask   
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, mntpl, mask, t) {\
try\
{\
    /* Generate position indices: range from 0 to input sequence length(exclusive), length is the number of tokens in the line */\
    p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)/*is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*/ mntpl, (t)1.0, DIMENSIONS{1, /*is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*/ mntpl, NULL, NULL}),  DIMENSIONS{1, /*is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*/ mntpl, NULL, NULL}};\
    \
        std::cout<< "p = " << p.getShape().getNumberOfColumns() << " - " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        std::cout<< "mask = " << mask.getShape().getNumberOfColumns() << " - " << mask.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    p = p * mask;\
        std::cout<< "p * mask = " << p.getShape().getNumberOfColumns() << " - " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    /* Compute scaling term dt using an exponential function */\
    dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, /*1*/ mntpl, NULL, NULL}), dm), DIMENSIONS{dm, /*1*/ mntpl, NULL, NULL}};\
    /* Scale dt by a predefined scaling factor */ \
    dt = dt * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm));\
    /* Initialize position encoding tensor with zeros */\
    pe = Numcy::zeros<t>(DIMENSIONS{dm, is.getShape().getDimensionsOfArray().getNumberOfInnerArrays(), NULL, NULL});\
        /* Please read the comments */\
        std::cout<< "dt = " << dt.getShape().getNumberOfColumns() << " - " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    /* Compute sine-transformed position encodings */\
    /*p * dt;*/\
    Collective<t> product = Numcy::sin<t>(p * dt);\
        std::cout<< "product = " << product.getShape().getNumberOfColumns() << " - " << product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    /* Fill even and odd indices of position encoding */ \
    /*FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe,  product);*/\
    /*FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, product);*/\
}\
catch (ala_exception& e)\
{\
    throw ala_exception(cc_tokenizer::String<char>("BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE() -> ") + cc_tokenizer::String<char>(e.what()));\
}\
}\

#define NEW_BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, mntpl, mask, t)\
{\
    try\
    {\
        /* Generate position indices: range from 0 to input sequence length(exclusive), length is the number of tokens in the line */\
        p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)mntpl, (t)1.0, DIMENSIONS{1, mntpl, NULL, NULL}),  DIMENSIONS{1, mntpl, NULL, NULL}};\
            std::cout<< "p = " << p.getShape().getNumberOfColumns() << " - " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
            std::cout<< "mask = " << mask.getShape().getNumberOfColumns() << " - " << mask.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        std::cout<< "P output = ";\
        for (int i = 0; i < p.getShape().getN(); i++)\
        {\
            std::cout<< p[i] << ", ";\
        }\
        std::cout<< std::endl;\
        std::cout<< "mask output = ";\
        for (int i = 0; i < mask.getShape().getN(); i++)\
        {\
            std::cout<< mask[i] << ", ";\
        }\
        std::cout<< std::endl;\
        /* Apply mask: Set position indices to zero where padding tokens are present */\
        p = p * mask;\
            std::cout<< "p * mask = " << p.getShape().getNumberOfColumns() << " - " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        std::cout<< "p*mask output = ";\
        for (int i = 0; i < p.getShape().getN(); i++)\
        {\
            std::cout<< p[i] << ", ";\
        }\
        std::cout<< std::endl;\
        /* Compute scaling term dt using an exponential function */\
        Collective<t> temp = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, mntpl, NULL, NULL}), DIMENSIONS{dm, mntpl, NULL, NULL}};\
        dt = Numcy::exp<t>(temp);\
        /*dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, mntpl, NULL, NULL}), dm), DIMENSIONS{dm, mntpl, NULL, NULL}};*/\
            std::cout<< "---> dt = " << dt.getShape().getNumberOfColumns() << " - " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        /* Scale dt by a predefined scaling factor */\
        dt = dt * (t)SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm);\
            std::cout<< "dt = " << dt.getShape().getNumberOfColumns() << " - " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        Collective<t> product = Numcy::sin<t>(p * dt);\
            std::cout<< "product = " << product.getShape().getNumberOfColumns() << " - " << product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        /* Initialize position encoding tensor with zeros */\
        pe = Numcy::zeros<t>(DIMENSIONS{dm, mntpl, NULL, NULL});\
        /* Fill even and odd indices of position encoding */\
        FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, product);\
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
                Collective<t> mask;\
                BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, mntpl, mask, t, w1);\
                std::cout<< "is = " << is.getShape().getNumberOfColumns() << " - " << is.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);\
                std::cout<< "ts = " << ts.getShape().getNumberOfColumns() << " - " << ts.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                NEW_BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, mntpl, mask, t);\
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