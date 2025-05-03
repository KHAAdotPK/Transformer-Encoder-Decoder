/*
    ML/NLP/transformers/encoder-decoder/attention.hh
    Multi-head attention implementation in the encoder of a sequence-to-sequence Transformer model
    Q@khaa.pk
 */

 /*
    Yooo that's the vibe! Dua Lipa blasting, terminal flying, neurons firing, and code compiling! I am in the debug-and-dance zone. Just keep saving those files before lift-off!
  */

/*
    Multi-Head Attention (MHA) Layer
    --------------------------------
    A neural network layer that allows the model to focus on different parts of the input sequence simultaneously.
    It splits the input into multiple "heads", applies attention separately in each head, and then combines the results.
    This helps the model capture different types of relationships and patterns in the data
 */ 

/*
    Masking has to be consistently applied in both forward and backward passes to avoid leaking gradient through padded tokens.
 */ 

/*
    ------------------------------
    | Forward Propogation Detail |
    ------------------------------

    Attention Output(O):         
    O = Attention(Q,K,V) = softmax(Q*Numcy::transpose(K)) * 1 / sqrt(d_k)) * V
    Where:
    - Q is query
    - K is key
    - d_k is the dimensions per attention head
 */
/*
    -------------------------------  
    | Backward Propogation Detail |
    -------------------------------
    TODO,

 */   
/*
    The Scaling Factor = 1 / sqrt(d_k):
    In this implementation of the Attention class, "d_k" is synonymous with "dimensionsOfAttentioHead".

    Without scaling: 
    - The dot products of Q and K(transposed) can become large when d_k is large
    - This pushes softmax into regions with very small gradients (bad for learning)

    With scaling:
    - By multiplying the dot products(output of softmax function) by "scaling factor", we normalize them
    - This keeps the softmax output in a numerically stable range
    - Thus "scaling factor" ensures gradients don't vanish or explode
 */ 
/*
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

    /*
        Separate weights:
        - If during the forward pass you kept Q, K, V separate, you would have to do the same in the backward pass that is in backprop, you don't need to concatenate the gradients.
        - If the implementation used a single large projection matrix ... like: W_qkv = concatenate({W^Q, W^K, W^V}, axis=1); then during backprop, you’ll also concatenate their gradients to match the structure: dW_qkv = concatenate({dW^Q, dW^K, dW^V}, axis=1);
     */
    // Projection matrices for Q, K, V (respectively W^Q, W^K, W^V) and "output projection weights" matrix in back propogation it is known as "Wo"
    Collective<t> queryWeights, keyWeights, valueWeights, outputWeights;
    t scaleFactor /* Scaling factor for attention scores */;
     
    
    Collective<t> X_ei_query, X_ei_key, X_ei_value; // Input tensors for the attention mechanism (Q, K, V)

    /* 
        "The computation graph records these operations." 
        The above statement refers to how this class manually tracks intermediate values during the 
        forward pass to enable gradient computation in the backward pass        
        Thus, what is a "Computation Graph"?
        - A computation graph is a directed graph where nodes represent operations or variables, and edges represent dependencies between them.
        - Or in our case... A computation graph is a record of mathematical operations performed during the forward pass, along with their dependencies (inputs/outputs).
        - In frameworks like PyTorch, this graph is built automatically. In this custom code of this "attention layer", 
          it got manually emulated by storing intermediate values in the following variables/private properties
     */
    Collective<t> masked_cached_query, masked_cached_key, masked_cached_value;
    Collective<t> cached_attention_weights;
    Collective<t> cached_output_before_projection;

    static constexpr t ATTENTION_LAYER_DEFAULT_LEARNING_RATE = 0.01;

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
            ------------------------------------------------------------------------------------------------------------------------------------
           | In the backward pass, we differentiate the loss with respect to parameters (gradients), and use those gradients to update weights. |
            ------------------------------------------------------------------------------------------------------------------------------------ 

            -> Compute the gradients of the loss L with respect to all inputs and parameters involved in the attention mechanism during backpropagation.         
            The backward pass of the attention mechanism involves computing gradients with respect to the input tensors (queries, keys, values) and the weights.
            -> We'll start from the final output of the attention mechanism and work our way backward through the operations performed during the forward pass.
            This is typically done using backpropagation through the attention mechanism
            
            @incoming_gradient (dL/dY, Y = OWo and hence dL/dOWo but then Y = Output as well), the incoming gradient from the next layer in the network.
            @return, The gradient with respect to the input tensors (queries, keys, values) and the weights.            
         */
        /**
         * @brief Backward pass for the Multi-Head Attention mechanism.
         * 
         * This function implements the complete backpropagation logic for the attention layer.
         * Given the incoming gradient from the next layer (dL/dY), it computes gradients with respect to:
         * - Output projection weights (W^O)
         * - Attention weights (A)
         * - Value vectors (V), Key vectors (K), and Query vectors (Q)
         * - Their respective projection weights (W^V, W^K, W^Q)
         * - The original input X (used for Q, K, V projections)
         * 
         * Steps followed:
         * 1. Compute dL/dW^O using the cached output before projection.
         * 2. Backpropagate to get dL/dO (gradient w.r.t. attention output).
         * 3. From dL/dO, compute:
         *    - dL/dA (attention weights)
         *    - dL/dV (value vectors)
         * 4. Apply softmax derivative to get dL/dS (attention scores).
         * 5. Use dL/dS to compute:
         *    - dL/dK (key vectors)
         *    - dL/dQ (query vectors)
         * 6. Use gradients from Q, K, and V to compute:
         *    - dL/dW^Q
         *    - dL/dW^K
         *    - dL/dW^V
         * 7. Update weights (Q, K, V) using gradients and learning rate.
         * 8. Compute gradients with respect to the original input X, by propagating gradients backward
         *    through the linear projection layers.
         * 
         * @param incoming_gradient The gradient from the next layer (∂L/∂Y).
         * @param learning_rate The learning rate used for weight updates.
         * @return Gradient with respect to the input of the attention layer (∂L/∂X).
         * 
         * @throws ala_exception if any step of the computation fails.
         */
        Collective<t> backward(Collective<t>& incoming_gradient, t learning_rate = ATTENTION_LAYER_DEFAULT_LEARNING_RATE) throw (ala_exception)
        {
            Collective<t> input_gradient; // Gradient with respect to the input tensors (queries, keys, values) and the weights.

            /*
                The backward pass of the attention mechanism involves computing gradients with respect to the input tensors (queries, keys, values) and the weights.
                This is typically done using backpropagation through the attention mechanism.
             */

            try 
            {   
                /*  
                    1. Gradient of Loss w.r.t. Output Projection Weights (Wo), dL/dWo = O^T * dL/dY
                    Where O = cached_output_before_projection, Wo = outputWeights, dL/dY = incoming_gradient when Y = OWo
                 */
                Collective<t> gradient_output_weights = Numcy::matmul<t>(Numcy::transpose(cached_output_before_projection), incoming_gradient);

                /*
                    2. Gradient of Loss w.r.t. Attention Output (O), dL/dO = dL/dY * Wo^T
                    where Wo is the "output projection weights", dL/dY is the "final Projected Output" (a.k.a incoming_gradient) 
                    therefore, dL/dO is the gradient of the loss with respect to the attention output (a.k.a gradient_attention_output)
                 */
                Collective<t> gradient_attention_output = Numcy::matmul<t>(incoming_gradient, Numcy::transpose(this->outputWeights));

                /*
                    3. Gradient of Loss w.r.t. Attention Weights (A), dL/dA = dL/dO * V^T
                    where V is the "value weights", we must use exactly the same V that was used in computing the attention output(forward pass) O = A * V
                    and then dL/dO is the "gradient_attention_output" of step 2 
                    therefore, dL/dA is the "gradient of the loss with respect to the attention weights" (a.k.a gradient_attention_weights)
                 */
                Collective<t> gradient_attention_weights = Numcy::matmul<t>(gradient_attention_output, Numcy::transpose(this->masked_cached_value));

                /*                    
                    4. Gradient Loss w.r.t. Value Vector (V = X.W^V), dL/dV = A^T * dL/dO
                    where A is the attention weights(a.k.a cached_attention_weights or just attention_weights), dL/dO is the gradient_attention_output
                 */
                Collective<t> gradient_value = Numcy::matmul<t>(Numcy::transpose(this->cached_attention_weights), gradient_attention_output);

                /*
                    5. Gradient of Loss w.r.t. Attention Scores, dL/dS = dL/dA * softmax'(A) (a.k.a softmax_backward(A))
                    where A is the attention weights(a.k.a cached_attention_weights), dL/dA is the "gradient_attention_weights" 
                    herefore, dL/dS is the "gradient of the loss with respect to the attention scores" (a.k.a gradient_attention_scores)
                 */
                Collective<t> gradient_attention_scores = softmax_backward(gradient_attention_weights, this->cached_attention_weights);

                /*
                    6. Gradient of Loss w.r.t. Key Vector (K = X.W^K), dL/dK = 1/sqrt(d_k) * ((dL/dS)^T * Q)
                    where Q is the query weights(a.k.a cached_query), dL/dS is the gradient_attention_scores and 1/sqrt(d_k) is the scaling factor(a.k.a scaleFactor)
                    herefore, dL/dK is the gradient of the loss with respect to the key vectors (a.k.a gradient_key)
                    6.1 => dL/dK = (dL/dS)^T * Q 
                    6.2 => dL/dK = dL/dK * scaleFactor
                 */                
                Collective<t> gradient_key = Numcy::matmul<t>(Numcy::transpose(gradient_attention_scores), this->masked_cached_query);
                gradient_key = gradient_key * scaleFactor;
                
                /*
                    7. Gradient of Loss w.r.t. Query Vector (Q = X.W^Q), dL/dQ = 1/sqrt(d_k) * ((dL/dS)^T * K)
                    where K is the key weights(a.k.a cached_key), dL/dS is the gradient_attention_scores and 1/sqrt(d_k) is the scaling factor(a.k.a scaleFactor)
                    herefore, dL/dQ is the gradient of the loss with respect to the query vectors (a.k.a gradient_query)
                    7.1 => dL/dQ = (dL/dS)^T * K 
                    7.2 => dL/dQ = dL/dQ * scaleFactor
                 */
                Collective<t> gradient_query = Numcy::matmul<t>(Numcy::transpose(gradient_attention_scores), this->masked_cached_key);
                gradient_query = gradient_query * scaleFactor; 
                
                /*
                    --------------------------------------------------------------------------------------------------------------------------------- 
                   | Finally.                                                                                                                        |  
                   | These three following steps are the correct final gradient calculations for the weights of the multi-head attention (MHA) layer |     
                    ---------------------------------------------------------------------------------------------------------------------------------
                 */
                /*
                    8. Gradient of Loss w.r.t. Query Weights (W^Q), dL/dW^Q = X^T * dL/dQ
                    where X is the input to the MHA layer, W^Q is projection matrix for Q(a.k.a queryWeights),
                    dL/dQ is the gradient_query(calculated in step 7)
                 */ 
                Collective<t> gradient_query_weights = Numcy::matmul<t>(Numcy::transpose(this->X_ei_query), gradient_query);
                /*
                    9. Gradient of Loss w.r.t. Key Weights (W^K), dL/dW^K = X^T * dL/dK
                    where X is the input to the MHA layer, W^K is projection matrix for K(a.k.a keyWeights),
                    dL/dK is the gradient_key(calculated in step 6)
                 */
                Collective<t> gradient_key_weights = Numcy::matmul<t>(Numcy::transpose(this->X_ei_key), gradient_key);
                /*  
                    10. Gradient of Loss w.r.t. Value Weights (W^V), dL/dW^V = X^T * dL/dV
                    where X is the input to the MHA layer, W^V is projection matrix for V(a.k.a valueWeights),
                    dL/dV is the gradient_value(calculated in step 4)
                 */
                Collective<t> gradient_value_weights = Numcy::matmul<t>(Numcy::transpose(this->X_ei_value), gradient_value);

                /*`                                                       
                    Learning Rate Scaling:
                    - During weight updates, we multiply the computed gradients by the learning rate.
                    - This controls the size of the update step taken towards minimizing the loss.
                    - A smaller learning rate means smaller updates (more stable but slower learning).
                    - A larger learning rate means bigger updates (faster but can cause instability if too large).
                    - Mathematically:  new_weight = old_weight - learning_rate * gradient
                 */
                gradient_query_weights = gradient_query_weights * learning_rate;
                this->queryWeights = this->queryWeights - gradient_query_weights;

                gradient_key_weights = gradient_key_weights * learning_rate;
                this->keyWeights = this->keyWeights - gradient_key_weights;

                gradient_value_weights = gradient_value_weights * learning_rate;
                this->valueWeights = this->valueWeights - gradient_value_weights;
                                
                /* 
                    Backpropogation: gradients flowing backwards from Q, K, V to their respectve Xs or inputs 
                    Chain Rule Logic in terms of Q and then same for K and V as well ...
                    - When Q = X.W^Q 
                      - W^Q is the projection matrix for Q, and it is constant  
                      - and X is the input to the MHA layer.                    
                    then dL/dX (changes in X affect the final loss) = dL/dQ * dQ/dX
                    there fore, dL/dX = dL/dQ * (W^Q)^T
                 */
                /*
                    Backpropagation: gradients flowing backward from Q, K, and V to their respective input X tensors.
                    Chain Rule Logic in terms of Q (similar for K and V as well):
                    
                    - When Q = X.W^Q:
                      - W^Q is the projection matrix for Q (learnable weights), considered constant during backpropagation at this point.
                      - X is the original input to the MHA layer.

                    Then, by chain rule:
                      dL/dX_from_Q (changes in X affect the final loss from Q side) = dL/dQ * (dQ/dX) 
                                                                                    = dL/dQ * (W^Q)^T

                    Similarly:
                      dL/dX_from_K (changes in X affect the final loss from K side) = dL/dK * (W^K)^T
                      dL/dX_from_V (changes in X affect the final loss from V side) = dL/dV * (W^V)^T

                    Since X was used three times to create Q, K, and V separately,
                    the total gradient with respect to X is the **sum** of these three contributions.
                    
                    That is:
                    
                    dL/dX = dL/dX (from Q path) + dL/dX (from K path) + dL/dX (from V path)

                    Finally, total dL/dX = dL/dX_from_Q + dL/dX_from_K + dL/dX_from_V
                    because the input X branches into three projections (Q, K, V) during the forward pass.
                 */
                /* Gradient of Loss w.r.t the input (X) that produced, Q(= X.W^Q) => dL/dX_query = dL/dQ * (W^Q)^T */
                Collective<t> input_gradient_from_query = Numcy::matmul(gradient_query, Numcy::transpose(this->queryWeights));
                /* Gradient of Loss w.r.t the input (X) that produced, K(= X.W^K) => dL/dX_key = dL/dK * (W^K)^T */
                Collective<t> input_gradient_from_key = Numcy::matmul(gradient_key, Numcy::transpose(this->keyWeights));
                /* Gradient of Loss w.r.t the input (X) that produced, V(= X.W^V) => dL/dX_value = dL/dV * (W^V)^T */
                Collective<t> input_gradient_from_value = Numcy::matmul(gradient_value, Numcy::transpose(this->valueWeights));
                
                // Summing all the gradients flowing into X
                /*incoming_gradient*/ input_gradient = input_gradient_from_query = input_gradient_from_query * learning_rate;
            } 
            catch (ala_exception& e) 
            {
                throw ala_exception(cc_tokenizer::String<char>("Attention::backward() -> ") + e.what());
            }
                        
            return /*incoming_gradient*/ input_gradient; // Placeholder return value
        }

        /*
            --------------------------------------------------------------------
           | In the forward pass, we compute outputs using the current weights. |
            --------------------------------------------------------------------

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
                that is why we can do away with just a query mask and we do not need a separate query mask 
             */
            Collective<t> query, key, value, scores;
            /*
                It makes sense to keep scores and attention_weights in plural form because:
                - Each query attends to multiple keys → The result is a matrix of scores.
                - Softmax produces multiple attention weights (one for each query-key pair).
             */
            Collective<t> attention_weights;

            Collective<t> output;

            X_ei_query = ei_query; // Cache the input for later use in backward pass
            X_ei_key = ei_key;     // Cache the input for later use in backward pass
            X_ei_value = ei_value; // Cache the input for later use in backward pass

            // this->cached_value = ei_value; // Cache the value for later use in backward pass
            
            try
            {

                /*
                    Use one and only one of the following scaling strategies:

                    1. Option 1: Scale Q and K during projections:
                        - Q = X * W^Q / sqrt(d_k)
                        - K = X * W^K / sqrt(d_k)
                        - V = X * W^V (no scaling needed in either case)
                        scores = query · key^T;
                    
                    2. Option 2: Scale scores after computing them:
                        - Q = X * W^Q
                        - K = X * W^K
                        - V = X * W^V (no scaling needed in either case)
                        scores = query · key^T / sqrt(d_k);    
                 */

                /*
                    (where X is the input to the MHA(Multi-Head Attention) layer, the one used for the value projection)
                 */

                /**********************************************************************************************************************************************************/
                /*Note: Only query and key are scaled by the square root of the head dimension (d_k) in the forward pass                                                  */
                /*      because the attention scores are computed as the dot product of query and key.                                                                    */
                /*      This scaling prevents the dot-product values from growing too large in magnitude, which would push softmax into regions with very small gradients.*/
                /**********************************************************************************************************************************************************/ 
                // Q: XW^Q, X is the input to the MHA layer(a.k.a ei_query)                
                query = Numcy::matmul<t>(ei_query, queryWeights) * scaleFactor;
                // K: XW^K, X is the input to the MHA layer(a.k.a ei_key) 
                key = Numcy::matmul<t>(ei_key, keyWeights) * scaleFactor;
                // V: XW^V, X is the input to the MHA layer(a.k.a ei_value)                
                value = Numcy::matmul<t>(ei_value, valueWeights); // No scaling for V

                /* I have checked with ADHOC_DEBUG_MACRO for the first run of above three functions their outputs keep the padding rows */
                
                /*
                    Masking has to be consistently applied in both forward and backward passes to avoid leaking gradient through padded tokens.                  
                    Zero out padded rows in the projected value matrix (after matmul(ei_value, valueWeights) but before attention)

                    Note:-
                    Scores corresponding to masked tokens should be set to -inf (or a very negative number) before softmax so they get zero weight.
                 */
                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < value.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); k++)
                {
                    if (mask[k] == 0)
                    {
                        for (cc_tokenizer::string_character_traits<char>::size_type l = 0; l < value.getShape().getNumberOfColumns(); l++)
                        {
                            query[k*value.getShape().getNumberOfColumns() + l] = /*std::numeric_limits<t>::lowest()*/ 0;
                            key[k*value.getShape().getNumberOfColumns() + l] = /*std::numeric_limits<t>::lowest()*/ 0;
                            value[k*value.getShape().getNumberOfColumns() + l] = /*std::numeric_limits<t>::lowest()*/ 0;
                        }
                    }
                }

                // Cache the transformed Q, K, V for backward pass
                /*
                    Make sure that it is the same value which is used in final attention projection output
                    O = A · V
                 */
                this->masked_cached_value = value;

                this->masked_cached_query = query;
                this->masked_cached_key = key;

               
                // *************************************** //
                //  Proceed with attention calculation...  //
                // *************************************** //

                /* Compute scaled dot-product attention scores */
                scores = Numcy::matmul<t>(query, Numcy::transpose(key)); 
                static_assert(std::is_same<cc_tokenizer::allocator<double>, cc_tokenizer::allocator<double>>::value, "Double allocator specialization missing");

                /* ********************************************** */
                /* IT IS HERE JUST FOR THE DOCUMENTATION PURPOSES */
                /* ********************************************** */
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
                /* // Collective<t> divisor = Numcy::zeros<t>(DIMENSIONS{1, 1, NULL, NULL});    
                   // divisor[0] = scaleFactor;
                   // scores = scores / divisor;*/
                /* // scores = scores / static_cast<double>(scaleFactor);
                   // std::cout << "Type of scaleFactor: " << typeid(decltype(scaleFactor)).name() << std::endl;*/

                ADHOC_IMPLEMENTATION_OF_MASK_QUERY(scores, mask, false);
                ADHOC_IMPLEMENTATION_OF_MASK_KEY(scores, mask, false);

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
                
                 /*
                    - A Attention weights, which are the normalized scores indicating how much focus each word should receive.
                        These weights are sometimes called just "attention weights"  and other times are called "cached attention weights"
                  */    
                // Apply softmax to get (attention weights a.k.a "A")  
                attention_weights = softmax<t>(scores);
                
                /*
                    - A cached
                      Attention weights, which are the normalized scores indicating how much focus each word should receive.
                      These weights are sometimes called just "attention weights"  and other times are called "cached attention weights"
                 */
                this->cached_attention_weights = attention_weights;
                
                /*
                    Multiply by value
                    O = A · V
                 */
                output = Numcy::matmul<t>(attention_weights, value);                                
                /*
                    - O  
                      Output from attention before output projection
                 */
                this->cached_output_before_projection = output;
                
                /*
                     Final Projected Output: Attention Projection Output = O*Wo = OWo Matrix
                     Y = O · Wo

                    - O
                      Output from attention before output projection (a.k.a "output")
                    - Wo 
                      Output projection weights (a.k.a "outputWeights")
                    
                    Let Y = O*Wo = OWo Matrix (a.k.a "Output matrix")
                    In Step-1 of the backward pass, we have dL/dY = incoming_gradient when Y = OWo
                 */
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