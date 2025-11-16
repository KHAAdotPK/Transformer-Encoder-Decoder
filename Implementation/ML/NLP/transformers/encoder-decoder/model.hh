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
#ifndef POSITIONAL_ENCODING_START_VALUE
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
#endif

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
         * @is             - An output parameter of type Collective<t> that will store the final input sequence.
         * @v              - Vocabulary object that maps tokens to indices
         * @icp            - Input CSV parser object representing the input corpus, which provides token-related information.
         * @mntpl_input    - Each input sequence is padded to ensure uniform length across variable-length sequences per line. 
         *                   The value of maximum number of tokens/sequences per line (mntpl_input) determines the size of all input sequences. 
         *                   If an input line has fewer tokens, padding is added to match the required length.              
         * @attentionMask  - Padding tokens should not receive valid position encodings because they do not contribute to the model’s 
         *                   understanding of sequence structure(padding tokens are added to make all input sequences 
         *                   uniform in length). 
         *                   Since positional encodings influence attention weights, allowing padding tokens to have meaningful encodings
         *                   might lead to misleading attention patterns.
         *                   You need a mask that differentiates real tokens from padding tokens. The mask should have:
         *                   Value (DEFAULT_VALID_WORD_VECTOR_MASK_VALUE) for real tokens (to keep their positional encoding).
         *                   Value (DEFAULT_PADDING_WORD_VECTOR_VALUE) for padding tokens (to zero out their positional encoding).
         *                   - attentionMask.getShape().getDimensionsOfArray()[0]  // Batch size
         *                   - attentionMask.getShape().getDimensionsOfArray()[1]  // Number of masks                             
         *                   - attentionMask.getShape().getDimensionsOfArray()[2]  // Mask length
         * @t              - The data type of embeddings (e.g., float, double).
         * @w1             - Matrix of pre-trained word embeddings where each row represents a word vector.
         * @redundancy     - Optional parameter that allows for multiple occurrences of the same token in the vocabulary.
         * @v              - Optional parameter that enables verbose output for debugging purposes.
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
         *    (PLEASE NOTE:-  Implementation can ignore trailing padding, example: If all sequences are mntpl_input=10, but only the first 7 positions contain valid tokens, you can use sequence_length=7 instead of a mask.) 
         * 
         * Error Handling:         
         * - Handles indexing errors         
         * - All errors are propagated with additional context
         *  
         * Note: The Vocabulary object uses internal indexing that starts at INDEX_ORIGINATES_AT_VALUE.
         *       In contrast, word embeddings use zero-based indexing (starting at 0).
         */    
        void buildInputSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& icp, CORPUS& iv, Collective<t>& is, Collective<t>& attentionMask, Collective<t>& W1, bool redundancy = ALLOW_REDUNDANCY, bool v = false) throw (ala_exception)
        {                                                                                                                                            
            try
            {   
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < attentionMask.getShape().getDimensionsOfArray()[0]; i++)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < attentionMask.getShape().getDimensionsOfArray()[1]; j++)
                    {                        
                        // The batch size (number of tokens) must not exceed the attention mask's length 
                        if ( !(icp.get_total_number_of_tokens() > attentionMask.getShape().getDimensionsOfArray()[2]) ) 
                        {
                            for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < icp.get_total_number_of_tokens() ; k++)
                            {
                                /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */
                                cc_tokenizer::string_character_traits<char>::size_type index = iv(icp.get_token_by_number(k + 1), icp.get_current_line_number(), k + 1, redundancy);

                                /* If this condition is false, we are no longer strictly using post-padding; instead, padding tokens may appear */
                                /* between valid tokens, leading to mixed padding. */
                                /* TODO: Investigate whether the following statement can ever evaluate to be false, because under that circumstances */
                                /* mixed padding might occur. */
                                if (index != INDEX_NOT_FOUND_AT_VALUE)
                                {
                                    attentionMask[i*(j*attentionMask.getShape().getDimensionsOfArray()[2]) + k] = DEFAULT_VALID_WORD_VECTOR_MASK_VALUE;

                                    /* we, Word Embedding */
                                    Collective<t> we = W1.slice((index - INDEX_ORIGINATES_AT_VALUE)*W1.getShape().getNumberOfColumns(), W1.getShape().getNumberOfColumns());
                                    for (cc_tokenizer::string_character_traits<char>::size_type l = 0; l < we.getShape().getN(); l++)
                                    {   
                                        is[k*we.getShape().getN() + l] = we[l];
                                    }
                                }
                                else
                                {
                                    /**********************************************************
                                     * VOCABULARY LOOKUP FAILURE HANDLING
                                     * 
                                     * If the token is not found in the vocabulary 
                                     * (index == INDEX_NOT_FOUND_AT_VALUE), we must halt 
                                     * processing immediately and raise an exception.
                                     * 
                                     * This prevents mixed-padding: If we continue processing 
                                     * after encountering an unknown token, padding tokens may 
                                     * be inserted between valid tokens instead of at the end, 
                                     * violating the post-padding strategy.
                                     **********************************************************/
                                    throw ala_exception("Model::buildInputSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>&, CORPUS&, Collective<t>&, Collective<t>&, Collective<t>&, bool, bool) Error: Encountered a token that is not present in the vocabulary. This should never happen if the inputs are within the expected range. Potential cause: Vocabulary is incomplete or incorrectly loaded.");
                                }                                                       
                            }
                        }
                        else
                        {
                            throw ala_exception("Model::buildInputSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>&, CORPUS&, Collective<t>&, Collective<t>&, Collective<t>&, bool, bool) Error: ");
                        }
                    }
                }                
#ifdef MAKE_THIS_MODEL_VERBOSE                     
#endif                
            }
            catch (ala_exception& e)
            {
                /**********************************************************************
                 * EXCEPTION PROPAGATION
                 * 
                 * Wrap and re-throw exceptions with additional context about the
                 * calling function to aid in debugging.
                 **********************************************************************/
                throw ala_exception(cc_tokenizer::String<char>("Model::buildInputSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>&, CORPUS&, Collective<t>&, Collective<t>&, Collective<t>&, bool, bool) -> ") + cc_tokenizer::String<char>(e.what()));   
            }
        }

        /**
         * @brief Constructs position encoding for a batch of input sequences using sinusoidal encoding.
         *
         * This function generates position encoding vectors that will be used in 
         * transformer-based models to retain positional information. It implements 
         * the standard sinusoidal position encoding scheme where:
         * - Even dimensions use sine function: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
         * - Odd dimensions use cosine function: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
         *
         * The function handles padding tokens by applying a mask to ensure they receive 
         * zero position encodings, maintaining the semantic meaning that padding tokens 
         * do not contribute to positional understanding.
         * 
         * @param pe An output parameter of type `Collective<t>` that stores the final position encodings.
         *           Shape: [sequence_length x dm]. Contains alternating sine/cosine values 
         *           for each position and embedding dimension.
         * 
         * @param dt An output parameter of type `Collective<t>` representing the division/scaling term.
         *           Shape: [sequence_length x dm]. Contains the exponential scaling factors 
         *           computed as exp(-log(10000) * 2*dim_pair / dm) for frequency modulation.
         * 
         * @param dm The model's embedding dimension. Must be even for proper sine/cosine pairing.
         * 
         * @param is An input tensor representing the input sequence batch (currently unused in implementation).
         * 
         * @param attentionMask A mask tensor of shape [batch size x number of masks x each mask length] that differentiates real tokens (1) 
         *        from padding tokens (0). Padding tokens should not receive valid position 
         *        encodings because they do not contribute to the model's understanding of 
         *        sequence structure. Padding tokens are added to make all input sequences 
         *        uniform in length. 
         * ------------------------------------------------------------------------------------------------------------------------*         
         *        AT THE MOMENT I AM JUST EXPERIEMENTING WITH SHAPES LIKE AS THIS PARAMETER HAS.                                   *
         *        I WILL NEED THIS SHAPE WHEN I HAVE BIGGER COMPUTE. IN THAT CASE I WILL BE USING A LOT OF DATA TO TRAIN MODELS.   *
         *        AT THAT TIME I WILL USE BATCH PROCESSING TO TRAIN MODELS.                                                        *
         *-------------------------------------------------------------------------------------------------------------------------*
         * 
         * @param mntpl_input Maximum number of tokens per line (sequence length). Each input 
         *        sequence is padded to ensure uniform length across variable-length sequences. 
         *        If an input line has fewer tokens, padding is added to match this required length.
         * 
         * @param sin_transformed_product An output parameter storing sine-transformed position*frequency products.
         *        Used as intermediate storage for sine calculations before final assignment.
         * 
         * @param cos_transformed_product An output parameter storing cosine-transformed position*frequency products.
         *        Used as intermediate storage for cosine calculations before final assignment.
         *
         * @throws ala_exception Thrown on memory allocation errors, length errors, or other exceptions
         *         during position encoding computation.
         *
         * Algorithm Steps:
         * 1. Generate position indices using arange() from start value to sequence length
         * 2. Apply mask to zero out positions corresponding to padding tokens
         * 3. Compute scaling factors (dt) using exponential decay based on dimension pairs
         * 4. Calculate position*frequency products for all positions and dimensions
         * 5. Apply sine transformation for even indices, cosine for odd indices
         * 6. Fill final position encoding matrix with masked sine/cosine values
         *
         * Mathematical Foundation:
         * - Frequency decreases exponentially with dimension: freq = 1/10000^(2i/d_model)
         * - Dimension pairing: dimensions (2i, 2i+1) share the same base frequency
         * - Masking ensures padding positions contribute zero to attention computations
         *
         * Memory Layout Notes:
         * - All matrices use row-major ordering for element access
         * - Position encodings are computed element-wise without broadcasting
         * - Final PE matrix has alternating sine/cosine pattern across dimensions
         */
        /*
            m = rows    n  = columns
            p = mntpl_input x 1
            mask = 1 x mntpl_input
            n = rows    p = columns          
            m x p                 
            p * mask   
         */
        void buildPositionEncoding(Collective<t>& pe, Collective<t>& dt, cc_tokenizer::string_character_traits<char>::size_type dm, Collective<t>& is, Collective<t>& attentionMask, cc_tokenizer::string_character_traits<char>::size_type mntpl_input, Collective<t>& sin_transformed_product, Collective<t>& cos_transformed_product) throw (ala_exception)
        {
            try
            {
                // PHASE 1: Generate positional terms using geometric progression
                // --------------------------------------------------------------
                // For each position i and dimension j, compute: i / (10000^(2j/dm))
                // This creates wavelengths forming a geometric progression from 2π to 10000·2π
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
                {
                    /* Generate position indices: range from POSITIONAL_ENCODING_START_VALUE(inclusive) to input sequence-length(exclusive), sequence-length is the number of tokens in a line. */
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < dm; j++)
                    {                        
                        // Pair dimensions to share frequencies: 
                        // Even j and j+1 (odd) use the same frequency base
                        // This creates sine/cosine pairs for robust position representation
                        cc_tokenizer::string_character_traits<char>::size_type dimension_pair = 0;

                        // For even j: use j/2, for odd j: use (j-1)/2 
                        // This ensures pairs of dimensions have the same base frequency 
                        if ( (j % 2) == 0 ) // Even
                        {
                            dimension_pair = (j / 2); // Integer division
                        } 
                        else // Odd 
                        {
                            dimension_pair = ((j - 1) / 2); // Integer division. For two consecutive dimensions, we use the same frequency
                        }

                        // Calculate exponent for geometric progression: (2j/dm) * ln(10000)
                        // This spaces frequencies logarithmically across dimensions
                        t exponent = (((2*dimension_pair)/(t)dm) * std::log(SCALING_FACTOR_CONSTANT));
                        
                        // Compute positional term: position / (10000^(2j/dm))
                        dt[i * dt.getShape().getNumberOfColumns() + j] =  (i/std::pow(10, exponent));
                    }              
                }                
                
                // PHASE 2: Apply sine and cosine transformations
                // ----------------------------------------------
                // Transform positional terms using sine (even dimensions) and cosine (odd dimensions)
                // This provides unique encoding for each (position, dimension) pair
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dt.getShape().getN(); i++)
                {
                    sin_transformed_product[i] = std::sin(dt[i]);
                    cos_transformed_product[i] = std::cos(dt[i]);
                }                          
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                
                /*std::cout<< "sin_transformed_product, Columns: " << sin_transformed_product.getShape().getNumberOfColumns() << ", Rows: " << sin_transformed_product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
#endif                                
                /* Fill even and odd indices separately */
                /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < pe.getShape().getN(); i+=2)
                {                       
                    //pe[i] = sin_transformed_product[i] * attentionMask[i/pe.getShape().getNumberOfColumns()];
                }
                for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i < pe.getShape().getN(); i+=2)
                {
                    //pe[i] = cos_transformed_product[i] * attentionMask[i/pe.getShape().getNumberOfColumns()];
                }*/
                
                // PHASE 3: Apply attention mask and assemble final positional encodings
                // ---------------------------------------------------------------------
                // Multiply positional encodings by attention mask to zero-out padding positions
                // This ensures padding tokens don't receive meaningful position information
                /*
                 * Validate that attention mask and positional encoding dimensions are compatible
                 * This ensures we can properly apply the mask to positional encodings
                 * TODO: Implement more comprehensive dimension validation earlier in the pipeline
                 */
                if (attentionMask.getShape().getNumberOfColumns() == pe.getShape().getDimensionsOfArray().getNumberOfInnerArrays())
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < attentionMask.getShape().getDimensionsOfArray()[0]; i++)
                    {
                        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < attentionMask.getShape().getDimensionsOfArray()[1]; j++)
                        {
                       
                            //attentionMask[i*attentionMask.getShape().getDimensionsOfArray()[1]*attentionMask.getShape().getDimensionsOfArray()[2] + j*attentionMask.getShape().getDimensionsOfArray()[2]];

                            /* Fill even and odd indices separately */

                            // Apply sine encodings to even dimensions
                            for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < pe.getShape().getN(); k+=2)
                            {
                                pe[k] = sin_transformed_product[k] * attentionMask[i*attentionMask.getShape().getDimensionsOfArray()[1]*attentionMask.getShape().getDimensionsOfArray()[2] + j*attentionMask.getShape().getDimensionsOfArray()[2] + k/pe.getShape().getNumberOfColumns()]; 
                            }

                            // Apply cosine encodings to odd dimensions 
                            for (cc_tokenizer::string_character_traits<char>::size_type k = 1; k < pe.getShape().getN(); k+=2)
                            {
                                pe[k] = cos_transformed_product[k] * attentionMask[i*attentionMask.getShape().getDimensionsOfArray()[1]*attentionMask.getShape().getDimensionsOfArray()[2] + j*attentionMask.getShape().getDimensionsOfArray()[2] + k/pe.getShape().getNumberOfColumns()]; 
                            }
                        }
                    }
                }
                else 
                {                       
                    // NO cleanup performed 
                    throw ala_exception("Model::buildPositionEncoding(Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&) Error: Attention mask and positional encoding dimension mismatch");
                }
            }
            catch (std::bad_alloc& e)
            {
                // CRITICAL: Memory allocation failure - system should terminate immediately
                // NO cleanup performed - this is a fatal error requiring process exit
                throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding(Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&) Error: ") + cc_tokenizer::String<char>(e.what()));
            }
            catch (std::length_error& e)
            {
                // CRITICAL: Length constraint violation - system should terminate immediately
                // NO cleanup performed - this is a fatal error requiring process exit
                throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding(Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&) Error: ") + cc_tokenizer::String<char>(e.what()));
            }            
            catch (ala_exception& e)
            {
                // Propagate existing ala_exception with additional context
                throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding(Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&, cc_tokenizer::string_character_traits<char>::size_type, Collective<t>&, Collective<t>&) -> ") + cc_tokenizer::String<char>(e.what()));
            }
        }

        void buildPositionEncoding_old(Collective<t>& p, Collective<t>& pe, Collective<t>& dt, cc_tokenizer::string_character_traits<char>::size_type dm, Collective<t>& is, Collective<t>& mask, cc_tokenizer::string_character_traits<char>::size_type mntpl_input, Collective<t>& sin_transformed_product, Collective<t>& cos_transformed_product) throw (ala_exception)
        {            
            try
            {   /*
                    Generate position indices: range from POSITIONAL_ENCODING_START_VALUE(inclusive) to input sequence-length(exclusive), sequence-length is the number of tokens in a line.
                    Placement new with Copy Construction
                 */
                p = Collective<t>{Numcy::arange<t, t>((t)POSITIONAL_ENCODING_START_VALUE, (t)mntpl_input + (t)(t)POSITIONAL_ENCODING_START_VALUE, (t)1.0, DIMENSIONS{1, mntpl_input, NULL, NULL}), DIMENSIONS{1, mntpl_input, NULL, NULL}};                
 
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
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < mntpl_input; i++)
                {             
                    p[i] = p[i] * mask[i];                 
                }                
                /*
                    // n in p must equal to m in mask
                    // p mow becomes matrix of colmns from mask and rows from p
                 */
                p = p * mask;
                                
                //p = Numcy::transpose<t>(p);

                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
                {
                    /*t value = (t)POSITIONAL_ENCODING_START_VALUE;*/

                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < dm; j++)
                    {                        
                        /* 
                            // Standard positional encoding:
                            // 1.0 / std::pow(10000.0, value / (t)dm); 
                         */

                        t dimension_pair = (t)0;

                        // For even j: use j/2, for odd j: use (j-1)/2 
                        // This ensures pairs of dimensions have the same base frequency 
                        if ( (j % 2) == 0 ) // Even
                        {
                            dimension_pair = (t)(j / 2); // Integer division
                        } 
                        else // Odd 
                        {
                            dimension_pair = (t)((j - 1) / 2); // Integer division
                        }

                        /* 
                            pos/antilog{(2i/d_model)*log(10,000)} => -1*(pos*antilog{(2i/d_model)*log(10,000)})
                         */

                        t exponent = (((2*dimension_pair)/(t)dm) * std::log(SCALING_FACTOR_CONSTANT));
                        
                        //t exponent = -std::log(SCALING_FACTOR_CONSTANT) * (2.0 * dimension_pair) / (t)dm; // I COMMENTED THIS.... 
                        dt[i * dt.getShape().getNumberOfColumns() + j] =  (i/std::pow(10, exponent));  /*std::exp(exponent);*/ // I COMMENTED THIS AS WELL 


                        // THIS IS OLD WORK DONOT CONSIDER THIS.....
                        /*t exponent = -std::log(10000.0) * (2.0 * j) / (t)dm;
                        dt[i*dt.getShape().getNumberOfColumns() + j] = std::exp(exponent);*/
                        /*dt[i*dt.getShape().getNumberOfColumns() + j] = std::exp(value * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm)));*/ 
                        /*value = value + (t)(2*i);  // Increments by 2*/
                    }              
                }                
 
                Collective<t> p_to_dt = p * dt;

                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < /*p_to_*/dt.getShape().getN(); i++)
                {
                    sin_transformed_product[i] = std::sin(/*p_to_*/dt[i]);
                    cos_transformed_product[i] = std::cos(/*p_to_*/dt[i]);
                }                          
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                
                /*std::cout<< "sin_transformed_product, Columns: " << sin_transformed_product.getShape().getNumberOfColumns() << ", Rows: " << sin_transformed_product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
#endif                                
                /* Fill even and odd indices separately */
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < pe.getShape().getN(); i+=2)
                {                       
                    pe[i] = sin_transformed_product[i] * mask[i/pe.getShape().getNumberOfColumns()];
                }
                for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i < pe.getShape().getN(); i+=2)
                {
                    pe[i] = cos_transformed_product[i] * mask[i/pe.getShape().getNumberOfColumns()];
                }

                // 64 rows, 3 columns                         3 rows and 1 column   
                //Numcy::transpose(sin_transformed_product) * Numcy::transpose(mask);
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
        /**
         * @brief Builds the target sequence and target sequence mask for transformer decoder training.
         * 
         * This function prepares two critical components for decoder training in an encoder-decoder transformer:
         * 1. Target Sequence (ts): The tokenized target sequence with special tokens
         * 2. Target Sequence Mask (tsm): A causal attention mask that prevents future token access
         * 
         * TARGET SEQUENCE (ts):
         * - Constructs a sequence starting with DECODER_INPUT_BEGINNING_OF_SEQUENCE (<START>)
         * - Maps each token from the corpus to its vocabulary index
         * - Handles unknown tokens by assigning DECODER_INPUT_UNKNOWN_VALUE (<UNK>)
         * - Appends DECODER_INPUT_END_OF_SEQUENCE (<END>) after all tokens
         * - Remaining positions are implicitly padded with DECODER_INPUT_PAD_VALUE (<PAD>)
         * 
         * TARGET SEQUENCE MASK (tsm):
         * - Creates a lower triangular matrix (causal mask) for decoder self-attention
         * - Each position can only attend to previous positions (including itself)
         * - Masks out padding tokens (sets to 0) to prevent attention to irrelevant positions
         * - Masks out unknown tokens (sets to 0) to reduce noise during training
         * - Allows attention to <START>, <END>, and valid vocabulary tokens (sets to 1)
         * 
         * Example:
         * Input tokens: ["hello", "world"]
         * Target sequence: [<START>, "hello", "world", <END>, <PAD>, <PAD>, ...]
         * Target mask: Lower triangular matrix with 1s for valid tokens, 0s for padding/unknown
         * 
         * @param tcp Tokenizer parser containing the target corpus tokens
         * @param tv Trget vocabulary callable that maps tokens to their corresponding indices
         * @param ts Output parameter storing the constructed target sequence
         * @param tsm Output parameter storing the target sequence attention mask (causal + padding mask)
         * @param mntpl_target Maximum number of tokens per line for padding (determines sequence length)
         * @param verbose Flag to enable detailed output logging
         * 
         * @throws ala_exception If tokenization or mask construction fails
         * 
         * @note 1: This function implements teacher forcing preparation where the decoder receives
         *          the ground truth sequence as input during training, with masking ensuring
         *          autoregressive behavior (no future token access).
         * @note 2: If your target sequence is [SOS, "hello", "world", EOS, PAD, PAD], the decoder receives [SOS, "hello", "world", EOS, PAD] as input (shifted right), 
         *          and the masking ensures it can't attend to padding positions or future positions during training. 
         */        
        void buildTragetSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& tcp, CORPUS& tv, Collective<t>& ts/*, Collective<t>& di*/, Collective<t>& tsm, Collective<t>& attentionMaskTargetSequence, cc_tokenizer::string_character_traits<char>::size_type mntpl_target = 0, bool verbose = false) throw (ala_exception)
        {   
            try 
            {
                ts[0] = DECODER_INPUT_BEGINNING_OF_SEQUENCE;

                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < tcp.get_total_number_of_tokens(); i++)
                {
                    /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */
                    cc_tokenizer::string_character_traits<char>::size_type index = tv(tcp.get_token_by_number(i + 1));

                    if (index != INDEX_NOT_FOUND_AT_VALUE)
                    {   
                        // Same thing but just stress testing things here... 
                        ts[i + 1] = tv(tcp.get_token_by_number(i + 1)) /* index */;
                        //std::cout<< "index = " << index << " ";
                    }
                    else
                    {

                        ts[i + 1] = DECODER_INPUT_UNKNOWN_VALUE;              
                    }
                }
            }
            catch (ala_exception &e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildTargetSequence() -> ") + cc_tokenizer::String<char>(e.what())); 
            }
            
            ts[tcp.get_total_number_of_tokens() + 1] = DECODER_INPUT_END_OF_SEQUENCE; 
            
            // Build decoder mask (target mask). Decoder self-attention - each position can only attend to previous positions (including itself)
            /*
                The mask is doing two things simultaneously:
                1. Causal masking: Lower triangular pattern prevents looking at future tokens (preventing future access)
                2. Padding masking: Stops at the actual sequence length (tcp.get_total_number_of_tokens() + 2 for <START> and <END>) (preventing attention to irrelevant tokens)
             */
            try
            {
                DIMENSIONSOFARRAY dimesionsOfArrayOfAttentionMaskTargetSequence = attentionMaskTargetSequence.getShape().getDimensionsOfArray();
                cc_tokenizer::string_character_traits<char>::size_type batch_size = dimesionsOfArrayOfAttentionMaskTargetSequence[0];
                cc_tokenizer::string_character_traits<char>::size_type number_of_masks = dimesionsOfArrayOfAttentionMaskTargetSequence[1];
                cc_tokenizer::string_character_traits<char>::size_type size_of_each_mask = dimesionsOfArrayOfAttentionMaskTargetSequence[2];

                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < tsm.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)                
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j <= i; j++)
                    {
                        if (ts[j] == DECODER_INPUT_BEGINNING_OF_SEQUENCE)
                        {
                            tsm[i*tsm.getShape().getNumberOfColumns() + j] = 1;
                        }
                        else if (ts[j] == DECODER_INPUT_END_OF_SEQUENCE)
                        {
                            tsm[i*tsm.getShape().getNumberOfColumns() + j] = 1;
                        }
                        else if (ts[j] == DECODER_INPUT_UNKNOWN_VALUE)
                        {
                            tsm[i*tsm.getShape().getNumberOfColumns() + j] = 0;
                        }
                        else if (ts[j] == DECODER_INPUT_PAD_VALUE)
                        {
                            tsm[i*tsm.getShape().getNumberOfColumns() + j] = 0;
                        }
                        else 
                        {
                            tsm[i*tsm.getShape().getNumberOfColumns() + j] = 1;
                        }
                    }
                }

                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < ts.getShape().getN(); i++)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j <= i; j++)
                    {
                        if (ts[j] == DECODER_INPUT_BEGINNING_OF_SEQUENCE)
                        {
                            attentionMaskTargetSequence[(0*number_of_masks*size_of_each_mask) + i*size_of_each_mask + j] = 1;                            
                        }
                        else if (ts[j] == DECODER_INPUT_END_OF_SEQUENCE)
                        {
                            attentionMaskTargetSequence[(0*number_of_masks*size_of_each_mask) + i*size_of_each_mask + j] = 1;     
                        }
                        else if (ts[j] == DECODER_INPUT_UNKNOWN_VALUE)
                        {
                            attentionMaskTargetSequence[(0*number_of_masks*size_of_each_mask) + i*size_of_each_mask + j] = 0; 
                        }
                        else if (ts[j] == DECODER_INPUT_PAD_VALUE)
                        {
                            attentionMaskTargetSequence[(0*number_of_masks*size_of_each_mask) + i*size_of_each_mask + j] = 0; 
                        }
                        else 
                        { 
                            attentionMaskTargetSequence[(0*number_of_masks*size_of_each_mask) + i*size_of_each_mask + j] = 1;                        
                        }
                    }
                }

                /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < batch_size; i++)
                {
                    for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < number_of_masks; j++)
                    {
                        for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < size_of_each_mask; k++)
                        {

                        }
                    }
                }*/
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildTargetSequence() -> ") + cc_tokenizer::String<char>(e.what())); 
            }                        
        }

        void buildDecoderInputFromTargetSequenceAndTargetMask(Collective<t>& di, Collective<t>& ts, Collective<t>& tsm, Collective<t>& attentionMaskTargetSequence) throw (ala_exception)
        {               
            DIMENSIONSOFARRAY dima = di.getShape().getDimensionsOfArray();

            if (dima.size() != 3)
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildDecoderInputFromTargetSequenceAndTargetMask(Collective<t>&, Collective<t>&, Collective<t>&) Error: Invalid decoder input dimensions: Expected 3D tensor (batch_size, seq_length, d_model+1), but received ") + cc_tokenizer::String<char>(dima.size()) + cc_tokenizer::String<char>("D tensor."));
            }

            cc_tokenizer::string_character_traits<char>::size_type batch_size = dima[0];
            cc_tokenizer::string_character_traits<char>::size_type shifted_target_sequence_length = dima[1];
            cc_tokenizer::string_character_traits<char>::size_type d_model = dima[2] - 1; // 1 is for TOKEN_ID - The total dimension is d_model + 1 where +1 reserves space for the token ID

            if (!(ts.getShape().getN() == shifted_target_sequence_length))
            {
                throw ala_exception(cc_tokenizer::String<char>("Model::buildDecoderInputFromTargetSequenceAndTargetMask(Collective<t>&, Collective<t>&, Collective<t>&) Error: Sequence length mismatch: Target sequence length (") + cc_tokenizer::String<char>(ts.getShape().getN()) + cc_tokenizer::String<char>(") does not match decoder input sequence dimension (") + cc_tokenizer::String<char>(shifted_target_sequence_length) + cc_tokenizer::String<char>(")")); 
            }
            
            cc_tokenizer::string_character_traits<char>::size_type i, j, k;

            for (i = 0; i < batch_size; i++)
            {
                for (j = 0; j < shifted_target_sequence_length; j++)
                {
                    // TOKEN_ID APPEARS HERE: Storing the actual token ID from target sequence
                    // This places the token ID at position [i, j, 0] in the 3D tensor
                    di[(i*shifted_target_sequence_length*(d_model + 1)) + (j*(d_model + 1) /* At TOKEN_ID */)] = ts[j]; // TOKEN_ID stored here

                    //Collective<t> rand_values = Numcy::Random::randn<t>(DIMENSIONS{d_model, 1, NULL, NULL});

                    Collective<t> rand_values = Numcy::Random::randn_xavier(DIMENSIONS{d_model, 1, NULL, NULL}, false);

                    for (k = 1; k <= d_model; k++)
                    {
                        di[(i*shifted_target_sequence_length*(d_model + 1 /* Passed TOKEN_ID */)) + (j*(d_model + 1 /* Passed TOKEN_ID */) + k)] = rand_values[k - 1];
                    }

                    // di[(i*shifted_target_sequence_length*(d_model + 1)) + (j*d_model)  ) => ts[j] /* token sequence value */
                    // di[(i*shifted_target_sequence_length*(d_model + 1)) + (j*(d_model + 1)) /* passed ts value */ + k ) => rand_value[k] k is upto d_model
                }
            }

            // Memory layout visualization for each sequence position [i, j]:
            // [TOKEN_ID, embedding_0, embedding_1, ..., embedding_{d_model-1}]
            // ^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            // Position 0            Positions 1 through d_model
        }
    
        /*
            @es, epochs
            @iv, input sequence vocabulary
            @tv, target sequence vocabulary
            @icp, input csv parser
            @tcp, target csv parser
            @is, input sequence
            @ts, target sequence
            @tsm, target sequence mask
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
        void startTraining(cc_tokenizer::string_character_traits<char>::size_type es, CORPUS& iv, CORPUS& tv, cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& icp, cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& tcp, Collective<t>& is, Collective<t>& ts, Collective<t>& tsm, Collective<t>& p, Collective<t>& pe, cc_tokenizer::string_character_traits<char>::size_type dm, Collective<t>& dt, Collective<t>& ei, Collective<t>& di, Collective<t>& W1, BatchType batch = SINGLE_LINE, bool v = false) throw (ala_exception)
        {            
            switch (batch)
            {
                case SINGLE_LINE:
                {
                    /* *************************************************************************************************************************** */
                    /*                                                 MAXIMUM SEQUENCE LENGTHS                                                    */
                    /*      If a inut/target sequence has fewer tokens then the sequence is padded to make it same as all the other sequecnces     */
                    /* *************************************************************************************************************************** */
                    /* Maximum number of tokens per line(number of tokens in the largest line of input text) */
                    cc_tokenizer::string_character_traits<char>::size_type mntpl_input = icp.max_sequence_length();
                    /* Maximum number of tokens per line(number of tokens in the largest line of target text) */
                    cc_tokenizer::string_character_traits<char>::size_type mntpl_target = tcp.max_sequence_length();


                    t* ptr = NULL;
                    Collective<t> mask;
                    Collective<t> sin_transformed_product;
                    Collective<t> cos_transformed_product;

                    try
                    {   
                        /* 
                             Post-Padding instead of Pre-Padding or Mixed-Padding 
                            ------------------------------------------------------
                            Here, using post-padding by allocating memory for a fixed-length sequence (mntpl_input) and filling it with real tokens first, 
                            followed by padding. This means that all padding appears at the end, not in the middle or mixed with real tokens. 
                         */

                        /*ptr = cc_tokenizer::allocator<t>().allocate(mntpl_input); 

                        memset(ptr, 0, sizeof(t)*mntpl_input);
                        p = Collective<t>{ptr, DIMENSIONS{1, mntpl_input, NULL, NULL}};*/
                    
                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl_input*dm); 
                        memset(ptr, 0, sizeof(t)*mntpl_input*dm);
                        sin_transformed_product = Collective<t>{ptr, DIMENSIONS{dm, mntpl_input, NULL, NULL}};

                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl_input*dm); 
                        memset(ptr, 0, sizeof(t)*mntpl_input*dm);
                        cos_transformed_product = Collective<t>{ptr, DIMENSIONS{dm, mntpl_input, NULL, NULL}};
                        
                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl_input*W1.getShape().getNumberOfColumns());
                        memset (ptr, 0, sizeof(t)*(mntpl_input*W1.getShape().getNumberOfColumns()));
                        is = Collective<t>{ptr, DIMENSIONS{W1.getShape().getNumberOfColumns(), mntpl_input, NULL, NULL}};

                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl_input);
                        memset(ptr, 0, sizeof(t)*mntpl_input);
                        mask = Collective<t>{ptr, DIMENSIONS{mntpl_input, 1, NULL, NULL}};

                        /* Allocate enough memory for scaling term dt and reset it to zeros */
                        ptr = cc_tokenizer::allocator<t>().allocate(dm*mntpl_input);
                        memset(ptr, 0, sizeof(t)*dm*mntpl_input);
                        dt = Collective<t>{ptr, DIMENSIONS{dm, mntpl_input, NULL, NULL}};

                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl_input*dm); 
                        memset(ptr, 0, sizeof(t)*mntpl_input*dm);
                        pe = Collective<t>{ptr, DIMENSIONS{dm, mntpl_input, NULL, NULL}};
                        
                        /* Target Sequence and Target Sequence Mask */
                        /* 2 is for two more memory locations to store <BOS> , <EOS> */
                        ptr = cc_tokenizer::allocator<t>().allocate(mntpl_target + 2);
                        memset(ptr, DECODER_INPUT_PAD_VALUE, sizeof(t)*(mntpl_target + 2));
                        ts = Collective<t>{ptr, DIMENSIONS{mntpl_target + 2, 1, NULL, NULL}};
                        ptr = cc_tokenizer::allocator<t>().allocate((mntpl_target + 2)*(mntpl_target + 2));
                        memset(ptr, 0, sizeof(t)*((mntpl_target + 2)*(mntpl_target + 2)));
                        tsm = Collective<t>{ptr, DIMENSIONS{mntpl_target + 2, mntpl_target + 2, NULL, NULL}};
                        
                        // Decoder related 
                        ptr = (t*)cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(DECODER_INPUT_DIMMENSIONS);
                        memset(ptr, 0, sizeof(cc_tokenizer::string_character_traits<char>::size_type)*DECODER_INPUT_DIMMENSIONS);
                        /*
                            Decoder input shape: [batch_size, shifted_target_sequence_length, d_model + 1]
                         */
                        ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[0] = 1; // Batch size
                        ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[1] = mntpl_target + 2; // Sequence length + begining marker + ending marker
                        /*
                            d_model is the standard term used in the Transformer architecture for the dimensionality of the model's hidden states. 
                            TOKEN_ID with their embedding representation of sequence token (where token could be BOS, EOS of the actual token of the batch). 
                            Due to +1 the dimensions are set so it accommodates both the token ID and its embedding representation.
                         */
                        ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[2] = dm + 1; // d_model + Place to store TOKEN_ID
                        
                        DIMENSIONSOFARRAY dimensionsOfInput((cc_tokenizer::string_character_traits<char>::size_type*)ptr, DECODER_INPUT_DIMMENSIONS);
                        DIMENSIONS decoderInputShape(dimensionsOfInput);                                            
                        /*std::cout<< "Decoder Input columns: " << decoderInputShape.getNumberOfColumns() << ", Rows: " << decoderInputShape.getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
                        ptr = cc_tokenizer::allocator<t>().allocate(decoderInputShape.getN());
                        memset(ptr, 0, sizeof(t)*decoderInputShape.getN());                        
                        di = Collective<t>{ptr, decoderInputShape};
                        /* ************************************************************************************** */
                        // Experimental mask..... attentionMaskInputSequence
                        /* ********************************************************************************************************************************* */
                            ptr = (t*)cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(DECODER_INPUT_DIMMENSIONS);
                            ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[0] = 1; // Batch size
                            ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[1] = 1; // Number of masks                             
                            ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[2] = mntpl_input; // Mask length
                            dimensionsOfInput = DIMENSIONSOFARRAY((cc_tokenizer::string_character_traits<char>::size_type*)ptr, DECODER_INPUT_DIMMENSIONS);
                            DIMENSIONS attentionMaskShape(dimensionsOfInput);
                            ptr = cc_tokenizer::allocator<t>().allocate(attentionMaskShape.getN());
                            memset(ptr, 0, sizeof(t)*attentionMaskShape.getN());
                            Collective<t> attentionMaskInputSequence(ptr, attentionMaskShape);                            
                        /* ********************************************************************************************************************************* */

                        /* ************************************************************************************** */
                        // Experimental mask..... attentionMaskTargetSequence 
                        /* ********************************************************************************************************************************* */
                           ptr = (t*)cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(DECODER_INPUT_DIMMENSIONS);
                           ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[0] = 1; // Batch size
                           ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[1] = mntpl_target + 2; // Number of masks                             
                           ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[2] = mntpl_target + 2; // Mask length
                           dimensionsOfInput = DIMENSIONSOFARRAY((cc_tokenizer::string_character_traits<char>::size_type*)ptr, DECODER_INPUT_DIMMENSIONS);
                           //DIMENSIONS attentionMaskShape(dimensionsOfInput);
                           attentionMaskShape = DIMENSIONS(dimensionsOfInput);
                           ptr = cc_tokenizer::allocator<t>().allocate(attentionMaskShape.getN());
                           memset(ptr, 0, sizeof(t)*attentionMaskShape.getN());
                           Collective<t> attentionMaskTargetSequence(ptr, attentionMaskShape);
                        /* ********************************************************************************************************************************* */
                                                                                                                        
                        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < es; i++)
                        {
                            if (v)
                            {                        
                                std::cout<< "Epoch " << (i + 1) << ", batch size set to a single line and total number of lines in input vocabulary is " << icp.get_total_number_of_lines()<< " and total number of lines in target vocabulary is " << tcp.get_total_number_of_lines() << std::endl;
                            }
#ifdef  MAKE_THIS_MODEL_VERBOSE_FOR_DECODER_INPUT                                
                            //std::cout<< "Decoder Input columns: " << decoderInputShape.getNumberOfColumns() << ", Rows: " << decoderInputShape.getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
#endif                              
                            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.get_number_of_lines(); j++)
                            {
                                icp.get_line_by_number(j + 1);
                                tcp.get_line_by_number(j + 1);

                                if (v)
                                {
                                    std::cout << "Status of Forward Pass " << (j + 1) << ", input tokens# "<< icp.get_total_number_of_tokens() << ", target tokens# "<< tcp.get_total_number_of_tokens() << std::endl;
                                }

                                buildInputSequence(icp, iv, is, attentionMaskInputSequence, W1, !ALLOW_REDUNDANCY);                                
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_INPUT_SEQUENCE                                
                                std::cout<< "::: DEBUG DATA -: Model::buildInputSequence() :- :::"  << std::endl;
                                std::cout<< "Number of tokens in this line: " << icp.get_total_number_of_tokens() << std::endl; 
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
                                std::cout<< "attentionMask, Columns: " << attentionMaskInputSequence.getShape().getNumberOfColumns() << ", Rows: " << attentionMaskInputSequence.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < attentionMaskInputSequence.getShape().getN(); k++)
                                {
                                    std::cout<< attentionMaskInputSequence[(k/attentionMaskInputSequence.getShape().getNumberOfColumns())*attentionMaskInputSequence.getShape().getNumberOfColumns() + (k%attentionMaskInputSequence.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%attentionMaskInputSequence.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
#endif
                                buildTragetSequence(tcp, tv, ts/*, di*/, tsm, attentionMaskTargetSequence, mntpl_target, v);                                
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_TARGET_ENCODING                                
                                std::cout<< "::: DEBUG DATA -: Model::buildTargetSequence() :- :::"  << std::endl;
                                std::cout<< "ts(Target Sequence), Columns: " << ts.getShape().getNumberOfColumns() << ", Rows: " << ts.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                std::cout<< "Actual number of tokens (tcp.get_total_number_of_tokens()): " << tcp.get_total_number_of_tokens() << std::endl;
                                /*for (int k = 0; k < ts.getShape().getN(); k++)
                                {
                                    std::cout<< ts[k] << " ";
                                }
                                std::cout<< std::endl;*/
                                for (int k = 0; k < ts.getShape().getN(); k++)
                                {
                                    std::cout<< ts[(k/ts.getShape().getNumberOfColumns())*ts.getShape().getNumberOfColumns() + (k%ts.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%ts.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
                                /*std::cout<< "::: DEBUG DATA -: Model::buildTargetSequence() :- :::"  << std::endl;*/
                                std::cout<< "tsm(Target Sequence Mask), Columns: " << tsm.getShape().getNumberOfColumns() << ", Rows: " << tsm.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < tsm.getShape().getN(); k++)
                                {
                                    std::cout<< tsm[(k/tsm.getShape().getNumberOfColumns())*tsm.getShape().getNumberOfColumns() + (k%tsm.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%tsm.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
                                std::cout<< "attentionMaskTragetSequence(tsm), Batch Size: " << attentionMaskTargetSequence.getShape().getDimensionsOfArray()[0] << ", Number of masks = " << attentionMaskTargetSequence.getShape().getDimensionsOfArray()[1] << ", Size of each mask = " << attentionMaskTargetSequence.getShape().getDimensionsOfArray()[2] << std::endl;
                                {
                                    DIMENSIONSOFARRAY dimesionsOfArrayOfAttentionMaskTargetSequence = attentionMaskTargetSequence.getShape().getDimensionsOfArray();
                                    cc_tokenizer::string_character_traits<char>::size_type batch_size = dimesionsOfArrayOfAttentionMaskTargetSequence[0];
                                    cc_tokenizer::string_character_traits<char>::size_type number_of_masks = dimesionsOfArrayOfAttentionMaskTargetSequence[1];
                                    cc_tokenizer::string_character_traits<char>::size_type size_of_each_mask = dimesionsOfArrayOfAttentionMaskTargetSequence[2]; 
                                    
                                    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < batch_size; i++)
                                    {
                                        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < number_of_masks; j++)
                                        {
                                            for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < size_of_each_mask; k++)
                                            {
                                               std::cout<< attentionMaskTargetSequence[i*number_of_masks*size_of_each_mask + j*size_of_each_mask + k] << " ";
                                            }                                        
                                            std::cout<< std::endl;
                                        }
                                        std::cout<< std::endl;
                                    }
                                }
#endif
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                                
#endif                          
                                buildPositionEncoding(pe, dt, dm, is, attentionMaskInputSequence /*mask*/, mntpl_input, sin_transformed_product, cos_transformed_product);                                
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
                                std::cout<< "sin_transformed_product, Columns: " << sin_transformed_product.getShape().getNumberOfColumns() << ", Rows: " << sin_transformed_product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < sin_transformed_product.getShape().getN(); k++)
                                {
                                    if (!(k%2))
                                    {
                                        std::cout<< sin_transformed_product[(k/sin_transformed_product.getShape().getNumberOfColumns())*sin_transformed_product.getShape().getNumberOfColumns() + (k%sin_transformed_product.getShape().getNumberOfColumns())] << " ";
                                    }
                                    else
                                    {
                                        std::cout<< " --ODD-- "; 
                                    }

                                    if ((k + 1)%sin_transformed_product.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }
                                std::cout<< "cos_transformed_product, Columns: " << cos_transformed_product.getShape().getNumberOfColumns() << ", Rows: " << cos_transformed_product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                for (int k = 0; k < cos_transformed_product.getShape().getN(); k++)
                                {
                                    if (k%2)
                                    {
                                        std::cout<< cos_transformed_product[(k/cos_transformed_product.getShape().getNumberOfColumns())*cos_transformed_product.getShape().getNumberOfColumns() + (k%cos_transformed_product.getShape().getNumberOfColumns())] << " ";
                                    }
                                    else
                                    {
                                        std::cout<< " --EVEN-- "; 
                                    }

                                    if ((k + 1)%cos_transformed_product.getShape().getNumberOfColumns() == 0)
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
                                /* 
                                    Concatenation Instead of Addition
                                    ---------------------------------
                                    A quick look at the values confirms that you have concatenated the pe matrix (3x64) and the is matrix (3x16) side-by-side. 64 + 16 = 80.
                                    This is a fundamental architectural error based on the original Transformer paper ("Attention Is All You Need").                                    
                                    The fundamental rule for combining token embeddings and positional encodings is that... 
                                    the encoder input should be the element-wise sum of the input embeddings and the positional encodings, and...
                                    this is only possible if they have the same dimensions

                                    What is wrong with concatination
                                    --------------------------------
                                    By concatenating them, you are creating a new, much larger feature vector (d_model=80) that is not standard and...
                                    will force all subsequent layers (like the Multi-Head Attention and Feed-Forward networks) to handle this larger dimension.
                                 */                                
                                //ei = Numcy::concatenate(pe, is); 
                                ei = pe + is;
                                DIMENSIONSOFARRAY ei_new_shape;
                                ptr = (t*)cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(ei.getShape().getNumberOfLinks() + DIMENSIONS_RESHAPE_CONSTANT);                                
                                ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[0] = 1; // Batch size
                                ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[1] = ei.getShape().getNumberOfRows();                             
                                ((cc_tokenizer::string_character_traits<char>::size_type*)ptr)[2] = ei.getShape().getNumberOfColumns();
                                /*std::cout<< "------>>>>>>>> " << ei.getShape().getReferenceCounts()[0] << std::endl;*/
                                ei_new_shape = DIMENSIONSOFARRAY{(cc_tokenizer::string_character_traits<char>::size_type*)ptr, ei.getShape().getNumberOfLinks() + DIMENSIONS_RESHAPE_CONSTANT, ei.getShape().getReferenceCounts()[0] /*NUMCY_DEFAULT_REFERENCE_COUNT*/};

                                /*for (int i = 0; i < ei_new_shape.size(); i++)
                                {
                                    std::cout<< ei_new_shape[i] << " ";
                                }
                                std::cout<< std::endl;*/

                                ei.reShape(ei_new_shape);

                                /*std::cout<< "HELLO -> " << ei.getShape(1).getReferenceCounts()[0] << ", Size = " << ei.getShape().getDimensionsOfArray().size() << std::endl;

                                std::cout<< "---------------------------------------------------------------------------------" << std::endl;*/

                                /*ei_new_shape = ei.getShape().getDimensionsOfArray();
                                for (int i = 0; i < ei_new_shape.size(); i++)
                                {
                                    std::cout<< ei_new_shape[i] << " ";
                                }
                                std::cout<< std::endl;*/
                                
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_ENCODER_INPUT                                                                
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
#endif                                
                                Encoder<t> encoder(ei.getShape().getNumberOfColumns(), DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);                                 
                                Collective<t> encoder_output = encoder.forward(ei, mask, attentionMaskInputSequence);                                
                                /*                                    
                                    In the encoder input, rows (or lines) containing all zeros represent sequences with fewer tokens. 
                                    These rows typically arise due to padding when processing variable-length sequences.

                                    Issue:
                                    - During the encoding process, these initially all-zero rows may start containing non-zero values.
                                    - The exact reason for this is not yet fully identified, but possible causes include:
                                    1. **Unintended weight initialization effects**: 
                                        - Some layers may apply transformations that introduce small non-zero values.
                                          EncoderFeedForwardNetwork class contains biases (bias1 and bias2), which are added to the transformed input.
                                          This means that even if the input is entirely zero, the biases can introduce nonzero values

                                    2. **Layer normalization or residual connections**: 
                                        - Some architectures (e.g., Transformer-based models) use residual connections or 
                                          normalization layers that may propagate non-zero values, even for padded rows.
                                          For example in EncoderFeedForwardNetwork class, we have following statements...
                                          // First Linear Transformation
                                          local_input = Numcy::matmul(local_input, weights1) + bias1;
                                          // Apply ReLU Activation Function. ReLU, short for Rectified Linear Unit                                
                                          local_input = Numcy::ReLU(local_input);
                                          // Second Linear Transformation
                                          local_input = Numcy::matmul(local_input, weights2) + bias2;  

                                    3. **Self-attention operations**: 
                                        - If masking isn't applied correctly, self-attention may allow interactions 
                                          between padded and non-padded sequences, introducing non-zero values.
                                          In Attention::forward() we've the following statement...
                                          // Compute scaled dot-product attention scores
                                          scores = Numcy::matmul<t>(query, Numcy::transpose(key)); // scaleFactor
                                          The above statement and other statements in the same method may
                                          propagate non-zero values to zero-padded sequences if masking is not
                                          correctly applied before the softmax operation

                                    4. **Floating-point precision errors**: 
                                        - Certain operations, such as matrix multiplications, may lead to very small 
                                          non-zero values due to numerical precision issues.
                                          (Possible cause: Feed Forward Network & matrix multiplications)  
                                            - Even when theoretically all-zero inputs pass through a linear layer (`W * X + b`) = FFN,  
                                            floating-point arithmetic may introduce small non-zero values due to numerical imprecision.  
                                            - Feed Forward Networks (`FFN`) with `ReLU` activations can also introduce minor deviations  
                                            when handling zero vectors

                                    ADHOC Solution:
                                        - To maintain the integrity of the original input, it is crucial to ensure that any row 
                                          that was originally all zeros remains all zeros throughout the encoding process.
                                        - The following statement explicitly enforces this constraint by applying a masking operation
                                 */

                                ADHOC_IMPLEMENTATION_OF_MASK_QUERY(encoder_output, attentionMaskInputSequence /*mask*/, true);                                

#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_ENCODER_OUTPUT
                                std::cout<< "::: DEBUG DATA -: Encoder Output(eo) :- :::"  << std::endl;
                                std::cout<< "Columns: " << eo.getShape().getNumberOfColumns() << ", Rows: " << eo.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                                /*std::cout<< "Columns: " << mask.getShape().getNumberOfColumns() << ", Rows: " << mask.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
                                for (int k = 0; k < eo.getShape().getN(); k++)
                                {
                                    std::cout<< eo[(k/eo.getShape().getNumberOfColumns())*eo.getShape().getNumberOfColumns() + (k%eo.getShape().getNumberOfColumns())] << " ";
                                    if ((k + 1)%eo.getShape().getNumberOfColumns() == 0)
                                    {
                                        std::cout<< std::endl;
                                    }
                                }                                
#endif                          
                                /*std::cout<< di.getShape().getNumberOfColumns() << std::endl;
                                std::cout<< encoder_output.getShape().getNumberOfColumns() + 1 << std::endl;
                                std::cout<< attentionMaskTargetSequence.getShape().getNumberOfColumns() << std::endl;*/                                    
                                Decoder<t> decoder(di.getShape().getNumberOfColumns() /*encoder_output.getShape().getNumberOfColumns() + 1*/ /*attentionMaskTargetSequence.getShape().getNumberOfColumns()*/, DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER);
                                buildDecoderInputFromTargetSequenceAndTargetMask(di, ts, tsm, attentionMaskTargetSequence);
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_DECODER_INPUT
                                std::cout<< "::: DEBUG DATA -: Decoder Input(di) :- :::"  << std::endl; 
                                std::cout<< "Batch Size = " << di.getShape().getDimensionsOfArray()[0] << ", Shifted Right Sequence Length = " << di.getShape().getDimensionsOfArray()[1] << ", TOKEN_ID + d_model = " << di.getShape().getDimensionsOfArray()[2] << std::endl;
                                {
                                    cc_tokenizer::string_character_traits<char>::size_type x, y, z;
                                    for (x = 0; x < di.getShape().getDimensionsOfArray()[0]; x++)
                                    {
                                        for (y = 0; y < di.getShape().getDimensionsOfArray()[1]; y++)
                                        {
                                            std::cout<< di[x*(di.getShape().getDimensionsOfArray()[1]*di.getShape().getDimensionsOfArray()[2]) + (y*(di.getShape().getDimensionsOfArray()[2] - 0))] << " [";

                                            for (z = 1; z <= (di.getShape().getDimensionsOfArray()[2] - 1); z++)
                                            {
                                                std::cout<< di[x*(di.getShape().getDimensionsOfArray()[1]*di.getShape().getDimensionsOfArray()[2]) + (y*(di.getShape().getDimensionsOfArray()[2] - 0) + z)] << " ";
                                            }

                                            std::cout<< "]" << std::endl;
                                        }

                                        std::cout<< std::endl;
                                    }
                                }
#endif                                                                                                
                                Collective<t> decoder_output = decoder.forward(di, encoder_output, attentionMaskTargetSequence /*tsm*/, attentionMaskInputSequence /*mask*/);
                                /* *********************************************************************************************************************************************************************************************** */
                                /* *********************************************************************************************************************************************************************************************** */
                                /*                                                                      Reinitialize, input sequence and input sequence mask                                                                       */
                                /* *********************************************************************************************************************************************************************************************** */
                                /*for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < p.getShape().getN(); k++)
                                {
                                    p[k] = 0;
                                }*/
                                                                
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < attentionMaskInputSequence.getShape().getN(); k++)
                                {
                                    attentionMaskInputSequence[k] = 0;                                    
                                } 
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < attentionMaskTargetSequence.getShape().getN(); k++)
                                {
                                    attentionMaskTargetSequence[k] = 0;                                    
                                }                                 
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < sin_transformed_product.getShape().getN(); k++)
                                {
                                    sin_transformed_product[k] = 0;                                    
                                }
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < cos_transformed_product.getShape().getN(); k++)
                                {
                                    cos_transformed_product[k] = 0;
                                }
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
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < ts.getShape().getN(); k++)
                                {
                                    ts[k] = DECODER_INPUT_PAD_VALUE;
                                }
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < tsm.getShape().getN(); k++)
                                {
                                    tsm[k] = 0;
                                }                                
                                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < di.getShape().getN(); k++)
                                {
                                    di[k] = 0;
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
   dL/dx_j = sum_of_k dL/dy_k * dy_k/dx_j
   --------------------------------------    
   Where dL/dy_k(is "Collective<t> grad_output" here) is the gradient(gradient_attention_weights) of the loss with respect to the softmax output y_k(visit Attention::forward() and look for line scores = softmax(), so scores is y_k a.k.a cached_attention_weights).
   The Jacobian matrix of the softmax function is given by: dy_k/dx_j = softmax(x_j) * (delta(j, k) - softmax(x_k))...
   Wwhere delta(j, k) is the Kronecker delta function, which is 1 if j = k and 0 otherwise. 
   This means that the gradient of the softmax_backward function with respect to its input is a matrix where each element is computed as follows:
   - If j = k, the element is softmax(x_j) * (1 - softmax(x_j)) -> (softmax_output[i * grad_output.getShape().getNumberOfColumns() + j] * (1 - softmax_output[i * grad_output.getShape().getNumberOfColumns() + k])) 
   - If j != k, the element is -softmax(x_j) * (softmax(x_k) ->  (- softmax_output[i * grad_output.getShape().getNumberOfColumns() + j] * softmax_output[i * grad_output.getShape().getNumberOfColumns() + k])
 */

/*
    @brief Computes the gradient of the softmax function with respect to its input.

    This function computes the gradient of the softmax function using the chain rule. 
    The gradient is computed as follows:
      - For each element in the output, if it corresponds to the same index as the input, 
        it is multiplied by (1 - softmax_output).
      - For all other elements, it is multiplied by -softmax_output.

    Parameters:
      - grad_output: Collective<T> 
        The gradient of the loss with respect to the softmax output.
      - softmax_output: Collective<T> 
        The output of the softmax function.

    Returns:
      - Collective<T>
        The gradient of the softmax function with respect to its input.
 */
template <typename t>
Collective<t> softmax_backward(Collective<t>& grad_output, Collective<t>& softmax_output) throw (ala_exception)
{
    // grad_output: dL/dy (gradient of the loss w.r.t. softmax output)
    // softmax_output: y = softmax(x), the output from the softmax layer
    
    Collective<t> grad_input = Numcy::zeros<t>(grad_output.getShape());

    try
    {        
        // Loop over each sample in the batch
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < grad_output.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
        { // Loop over rows (batch size)
        
            // Loop over each element of the softmax output (j: output dimension)
            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < grad_output.getShape().getNumberOfColumns(); j++)
            { // Loop over columns (output dimension)
            
                // Loop again for Jacobian-vector product (k: softmax dimension)
                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < grad_output.getShape().getNumberOfColumns(); k++)
                {
                    if (j == k)
                    {
                        // Diagonal element of the Jacobian: s_j * (1 - s_j)
                        grad_input[i * grad_output.getShape().getNumberOfColumns() + j] += 
                        softmax_output[i * grad_output.getShape().getNumberOfColumns() + j] * 
                        (1 - softmax_output[i * grad_output.getShape().getNumberOfColumns() + k]) * 
                        grad_output[i * grad_output.getShape().getNumberOfColumns() + k];
                    } 
                    else 
                    {
                        // Off-diagonal element: -s_j * s_k
                        grad_input[i * grad_output.getShape().getNumberOfColumns() + j] -= 
                        softmax_output[i * grad_output.getShape().getNumberOfColumns() + j] * 
                        softmax_output[i * grad_output.getShape().getNumberOfColumns() + k] * 
                        grad_output[i * grad_output.getShape().getNumberOfColumns() + k];
                    }
                }
            }
        }
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("softmax_backward() -> ") + cc_tokenizer::String<char>(e.what()));
    }

    return grad_input;
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
 * @mntpl_input - Each input sequence is padded to ensure uniform length across variable-length sequences per line. 
 *          The value of maximum number of tokens/sequences per line (mntpl_input) determines the size of all input sequences. 
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
 *    (PLEASE NOTE:-  Implementation can ignore trailing padding, example: If all sequences are mntpl_input=10, but only the first 7 positions contain valid tokens, you can use sequence_length=7 instead of a mask.) 
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
#define BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, v, icp, mntpl_input, mask, t, w1) {\
t *ptr = NULL, *ptr_mask = NULL;\
try\
{\
    ptr = cc_tokenizer::allocator<t>().allocate(/*icp.get_total_number_of_tokens()*/ mntpl_input*w1.getShape().getNumberOfColumns());\
    /* Post-Padding instead of Pre-Padding or Mixed-Padding */\
    /* Here, using post-padding by allocating memory for a fixed-length sequence (mntpl_input) and filling it with real tokens first, */\
    /* followed by padding. This means that all padding appears at the end, not in the middle or mixed with real tokens. */\
    ptr_mask = cc_tokenizer::allocator<t>().allocate(mntpl_input);\
    memset(ptr, (t)DEFAULT_PADDING_WORD_VECTOR_VALUE, mntpl_input*w1.getShape().getNumberOfColumns());\
    memset(ptr_mask, DEFAULT_PADDING_WORD_VECTOR_VALUE, mntpl_input);\
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
is = Collective<t>{ptr, DIMENSIONS{w1.getShape().getNumberOfColumns(), /*static_cast<cc_tokenizer::string_character_traits<char>::size_type>(icp.get_total_number_of_tokens())*/ mntpl_input, NULL, NULL}};\
/* Assigning the mask to ensure padding tokens (0) do not receive position encodings */\
mask = Collective<t>{ptr_mask, DIMENSIONS{mntpl_input, 1, NULL, NULL}};\
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
 * @param mntpl_input Each input sequence is padded to ensure uniform length across variable-length sequences per line. 
 *        The value of maximum number of tokens/sequences per line (mntpl_input) determines the size of all input sequences. 
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
    p = mntpl_input x 1
    mask = 1 x mntpl_input
           n    p
           
    m x p      
           
    p * mask   
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, mntpl_input, mask, t) {\
try\
{\
    /* Generate position indices: range from 0 to input sequence length(exclusive), length is the number of tokens in the line */\
    p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)/*is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*/ mntpl_input, (t)1.0, DIMENSIONS{1, /*is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*/ mntpl_input, NULL, NULL}),  DIMENSIONS{1, /*is.getShape().getDimensionsOfArray().getNumberOfInnerArrays()*/ mntpl_input, NULL, NULL}};\
    \
        std::cout<< "p = " << p.getShape().getNumberOfColumns() << " - " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        std::cout<< "mask = " << mask.getShape().getNumberOfColumns() << " - " << mask.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    p = p * mask;\
        std::cout<< "p * mask = " << p.getShape().getNumberOfColumns() << " - " << p.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
    /* Compute scaling term dt using an exponential function */\
    dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, /*1*/ mntpl_input, NULL, NULL}), dm), DIMENSIONS{dm, /*1*/ mntpl_input, NULL, NULL}};\
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

#define NEW_BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, mntpl_input, mask, t)\
{\
    try\
    {\
        /* Generate position indices: range from 0 to input sequence length(exclusive), length is the number of tokens in the line */\
        p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)mntpl_input, (t)1.0, DIMENSIONS{1, mntpl_input, NULL, NULL}),  DIMENSIONS{1, mntpl_input, NULL, NULL}};\
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
        Collective<t> temp = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, mntpl_input, NULL, NULL}), DIMENSIONS{dm, mntpl_input, NULL, NULL}};\
        dt = Numcy::exp<t>(temp);\
        /*dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, DIMENSIONS{dm, mntpl_input, NULL, NULL}), dm), DIMENSIONS{dm, mntpl_input, NULL, NULL}};*/\
            std::cout<< "---> dt = " << dt.getShape().getNumberOfColumns() << " - " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        /* Scale dt by a predefined scaling factor */\
        dt = dt * (t)SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm);\
            std::cout<< "dt = " << dt.getShape().getNumberOfColumns() << " - " << dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        Collective<t> product = Numcy::sin<t>(p * dt);\
            std::cout<< "product = " << product.getShape().getNumberOfColumns() << " - " << product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
        /* Initialize position encoding tensor with zeros */\
        pe = Numcy::zeros<t>(DIMENSIONS{dm, mntpl_input, NULL, NULL});\
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
    cc_tokenizer::string_character_traits<char>::size_type mntpl_input = icp.max_sequence_length();\
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
                BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, mntpl_input, mask, t, w1);\
                std::cout<< "is = " << is.getShape().getNumberOfColumns() << " - " << is.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);\
                std::cout<< "ts = " << ts.getShape().getNumberOfColumns() << " - " << ts.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;\
                NEW_BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, mntpl_input, mask, t);\
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