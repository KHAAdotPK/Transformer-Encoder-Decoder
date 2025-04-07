/*
    ML/NLP/transformers/encoder-decoder/attention.hh
    Multi-head attention implementation in the encoder of a sequence-to-sequence Transformer model
    Q@khaa.pk
 */

/*
    Attention(Q,K,V) = softmax(Q*Numcy::transpose(K)) * 1 / sqrt(d_k)) * V
    Where:
    - Q is query
    - K is key
    - d_k is the dimensions per attention head

    scaling factor = 1 / sqrt(d_k)

    Without scaling: 
    - The dot products of Q and K(transposed) can become large when d_k is large
    - This pushes softmax into regions with very small gradients (bad for learning)

    With scaling:
    - By multiplying the dot products(output of softmax function) by "scaling factor", we normalize them
    - This keeps the softmax output in a numerically stable range
    - Thus "scaling factor" ensures gradients don't vanish or explode

    Taking the floor:    
    The floor function returns the greatest integer that is less than or equal to the given value.
    As a result, if the input to floor is less than 1, it will return 0.

    Example:
    floor(0.9) -> 0
    floor(0.1) -> 0
    floor(1.7) -> 1
    
    This behavior can lead to unintended results, especially when dealing with small fractional values
 */ 

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH

/*
    Multi head attention.
 */
template <typename t = double>
class Attention // Is all you need.
{    
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfAttentionHead /* Size of each attention head(in dimensions per head is) d_k = (d_model/num_heads), in other words dimensionsOfAttentionHead is our d_k  */ , dimensionsOfTheModel /* Model dimension (d_model) */, numberOfAttentionHeads /* Number of attention heads */;

    // Projection matrices for Q, K, V and final projection matrix
    Collective<t> queryWeights, keyWeights, valueWeights, outputWeights;
    t scaleFactor /* Scaling factor for attention scores */;

    public:
        //  Default constructor
        Attention(void) : dimensionsOfAttentionHead(floor((t)(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER/DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER))), dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER)
        {   
            /*DIMENSIONS dim3 = {10, 3, NULL, NULL};
            DIMENSIONS dim2 = {0, 10, &dim3, NULL};
            dim3.prev = &dim2;
            DIMENSIONS dim1 = {0, 78, &dim2, NULL};
            dim2.prev = &dim1;
            DIMENSIONS dim = {0, 9, &dim1, NULL};
            dim1.prev = &dim;               
            Numcy::Random::randn(dim);*/
           //Numcy::Random::randn(DIMENSIONS{0, 0, NULL, NULL});

           scaleFactor = (1.0 / std::sqrt(dimensionsOfAttentionHead));
        }

        // Parameterized constructor
        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads.            
         */
        Attention(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads) : dimensionsOfAttentionHead(floor((t)(d_model/num_heads))), dimensionsOfTheModel(d_model), numberOfAttentionHeads(num_heads)
        { 
            /*DIMENSIONS dim3 = DIMENSIONS{10, 3, NULL, NULL};
            DIMENSIONS dim2 = DIMENSIONS{0, 10, &dim3, NULL};
            dim3.prev = &dim2;
            DIMENSIONS dim1 = DIMENSIONS{0, 78, &dim2, NULL};
            dim2.prev = &dim1;
            DIMENSIONS dim = DIMENSIONS{0, 9, &dim1, NULL};
            dim1.prev = &dim;*/

            //DIMENSIONS dim3(DIMENSIONS{10, 3, NULL, NULL});
            //DIMENSIONS dim2(DIMENSIONS{0, 10, &dim3, NULL});

            //cc_tokenizer::string_character_traits<char>::size_type *ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(5);            
            //cc_tokenizer::string_character_traits<char>::size_type *ptr = reinterpret_cast<cc_tokenizer::string_character_traits<char>::size_type*>(cc_tokenizer::allocator<unsigned int>().allocate(5));
           
            DIMENSIONS dim = DIMENSIONS{d_model, d_model, NULL, NULL};
            
            try
            {            
                queryWeights = Numcy::Random::randn<t>(dim);
                keyWeights = Numcy::Random::randn<t>(dim);
                valueWeights = Numcy::Random::randn<t>(dim);
                outputWeights = Numcy::Random::randn<t>(dim);
            }
            catch (ala_exception& e)
            {                
                throw ala_exception(cc_tokenizer::String<char>("Attention::Attention() -> ") + e.what());
            }
           
            //scaleFactor = std::sqrt(d_model / num_heads);  // Scaling factor for stability
            /*
                Since the scaling factor in self-attention is typically sqrt(d_k), where d_k is the dimension of a single attention head.
                // scaleFactor = std::sqrt(d_model / num_heads);  // Scaling factor for stability
             */
            scaleFactor = (1.0 / std::sqrt(dimensionsOfAttentionHead));
        }

        /*
            In the context of Transformers, self-attention is a mechanism where the model attends to different positions within the 
            same sequence to capture relationships and dependencies between words. Essentially, it's like asking questions about 
            different parts of the input sequence and using the answers to understand the overall meaning.

            Here's how the self-attention works with the same argument for query, key, and value:
            1. Each element in input sequence(@ei) acts as a query, key, and value:- 
               Every element (word) in the input sequence(@ei) is used as a query 
               to ask a question about the relationship with other words. Simultaneously, it also acts as a key for other elements to 
               compare against and as a value that holds information potentially relevant to the queries. 
            2. Attention scores:- 
               The Multi-Head Attention module calculates attention scores by comparing each query with all the keys.
               These scores indicate how relevant each value is to the specific query,
               considering both the content of the words and their positions.
            3. Weighted sum:-
               The attention scores are used to compute a weighted sum of the values.
               Words with higher attention scores contribute more to the final output, 
               effectively focusing on the most relevant parts of the sequence for each query.

            By using the same encoder input sequence(@ei) for query, key, and value, the self-attention mechanism allows the 
            model to learn relationships between any two words within the sequence, regardless of their order. 
            This is crucial for capturing long-range dependencies and the overall context of the input.         
         */
        /*            
            @ei_query, 
            - Represents the sequence of queries on which attention is being computed.
            - Each query is a vector that's compared to keys(@ei_key) to determine relevant values.
            - It's like asking a question to get a focus on specific parts of the input.            
            @ei_key,
            - Contains the keys used to compute attention scores for each query.
            - Each key corresponds to a value in the @ei_value tensor.
            - Keys(@ei_keys) are used to compare with queries(@ei_query) to find relevant information.
            @ei_value,
            - Holds the values that will be selectively attended to based on the attention scores.
            - The values are the actual information that the attention mechanism extracts. 

            Key Points:
            1 Attention scores are computed using the dot product between queries(@ei_query) and keys(@ei_key), scaled by the square root of the head_dim().
            2 The model learns to assign higher attention scores to key-value pairs that are more relevant to the given query.
            - The return value...
            3 The final output of the forward method is the weighted sum of the values, where the weights are the normalized attention scores.
         */        
        Collective<t> forward(Collective<t>& ei_query, Collective<t>& ei_key, Collective<t>& ei_value, Collective<t>& mask)
        {                          
            /*
                Linear transformations, compute queries, keys, and values
             */

            /*
                It makes sense to use singular names (query, key, value) because:
                - Each line is processed independently (not as a batch).
                - Each token gets transformed into a query, key, and value vector separately before attention is applied.

                The "scores" matrix will always be square if, ei_query and ei_key has the same shape... 
                Which is true in the case of this implementation and...
                that is why we can do away with just a query mask and we donot need a seperate query mask 
             */
            Collective<t> query, key, value, scores;
            /*
                It makes sense to keep scores and attention_weights in plural form because:
                - Each query attends to multiple keys → The result is a matrix of scores.
                - Softmax produces multiple attention weights (one for each query-key pair).
             */
            Collective<t> attention_weights;

            Collective<t> output;
            
            try
            {
                query = Numcy::matmul<t>(ei_query, queryWeights) * sqrt(1.0 / dimensionsOfTheModel);                
                key = Numcy::matmul<t>(ei_key, keyWeights) * sqrt(1.0 / dimensionsOfTheModel);                
                value = Numcy::matmul<t>(ei_value, valueWeights) * sqrt(1.0 / dimensionsOfTheModel);
                
                // Zero out padded rows in the projected value matrix (after matmul(ei_value, valueWeights) but before attention)                 
                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < value.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); k++)
                {
                    if (mask[k] == 0)
                    {
                        for (cc_tokenizer::string_character_traits<char>::size_type l = 0; l < value.getShape().getNumberOfColumns(); l++)
                        {
                            //query[k*value.getShape().getNumberOfColumns() + l] = /*std::numeric_limits<t>::lowest()*/ 0;
                            //key[k*value.getShape().getNumberOfColumns() + l] = /*std::numeric_limits<t>::lowest()*/ 0;
                            value[k*value.getShape().getNumberOfColumns() + l] = /*std::numeric_limits<t>::lowest()*/ 0;
                        }
                    }
                }  

                /* I have checked with ADHOC_DEBUG_MACRO for the first run of above three functions their outputs keep the padding rows */

                // *************************************** //
                //  Proceed with attention calculation...  //
                // *************************************** //

                /* Compute scaled dot-product attention scores */
                scores = Numcy::matmul<t>(query, Numcy::transpose(key)) * sqrt(1.0 / dimensionsOfTheModel); 
                static_assert(std::is_same<cc_tokenizer::allocator<double>, cc_tokenizer::allocator<double>>::value, "Double allocator specialization missing");

                /**
                 * WORKAROUND IMPLEMENTATION FOR SCALAR DIVISION
                 * 
                 * Original Issue:
                 * - The template operator/(F x) that uses cc_tokenizer::allocator fails despite:
                 *   1. Confirmed allocator<double> specialization exists (static_assert passes)
                 *   2. scaleFactor is verified to be of type double (typeid shows 'd')
                 * - The root cause appears to be template instantiation/visibility issue in complex inheritance chain
                 *
                 * Current Solution:
                 * 1. Creates a temporary Collective<t> with shape [1,1] initialized to zeros
                 *    - Uses Numcy::zeros instead of allocator to avoid template issues
                 *    - Explicitly sets the single element to scaleFactor value
                 * 2. Uses existing Collective<t>/Collective<t> operator
                 *
                 * Advantages:
                 * - Avoids problematic allocator path entirely
                 * - Uses already tested/working matrix division
                 * - Maintains numerical consistency with other operations
                 *
                 * Trade-offs:
                 * - Slightly less efficient than direct scalar division:
                 *   - Allocates temporary matrix (though small)
                 *   - Uses full matrix division machinery
                 * - Requires scaleFactor to be convertible to type t
                 *
                 * Future Improvements:
                 * 1. Could implement optimized scalar division operator later:
                 *    template<typename t>
                 *    Collective<t> operator/(t scalar) { element-wise division }
                 * 2. Should investigate why allocator path fails despite proper specialization
                 *
                 * Debugging Notes:
                 * - Verified working for float/double cases
                 * - Maintains proper dimensionality in output
                 * - Preserves exception safety guarantees
                 */
                Collective<t> divisor = Numcy::zeros<t>(DIMENSIONS{1, 1, NULL, NULL});    
                divisor[0] = scaleFactor;
                scores = scores / divisor;                                
                //scores = scores / static_cast<double>(scaleFactor);
                //std::cout << "Type of scaleFactor: " << typeid(decltype(scaleFactor)).name() << std::endl;

                ADHOC_IMPLEMENTATION_OF_MASK_QUERY(scores, mask);
                ADHOC_IMPLEMENTATION_OF_MASK_KEY(scores, mask);
                /* ADHOC_DEBUG_MACRO(scores); */
                
                /*
                    Do You Need src_mask?
                    If input sequences are of equal length and don't have padding, then src_mask might not be meeded. However, it's best to support it for flexibility later.

                    In a Transformer encoder, src_mask (source mask) is typically used in the self-attention mechanism to:
                    1. Prevent attending to padding tokens (mask out padded positions in the input).
                    2. Control which tokens can attend to which (if needed, like in some structured data cases).

                    What You Need to Do?
                    If you're using matmul(Q, K^T), apply the mask before softmax:
                    attention_scores = attention_scores + src_mask;  // Apply mask  

                    Make sure src_mask has negative infinity (-inf) where padding exists, so softmax turns those values into 0.

                    Check Attention Class:
                    If attention implementation already accepts a mask parameter, pass src_mask from the encoder when calling forward()
                 */

                // Apply softmax to get attention weights   
                attention_weights = softmax<t>(scores);
                         
                /* ADHOC_DEBUG_MACRO(value); */
                
                // Multiply by value
                output = Numcy::matmul<t>(attention_weights, value);  
                // Apply output transformation
                output = Numcy::matmul<t>(output, outputWeights);                
            }
            catch(ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("Attention::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }
            
            return output;
        }

        ~Attention()
        {                        
        }


} /*ATTENTION, MULTIHEADATTENTION*/;

#endif