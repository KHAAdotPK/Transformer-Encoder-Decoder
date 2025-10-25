/*
    ML/NLP/transformers/encoder-decoder/EncoderLayer.hh
    Q@khaa.pk
 */

/*
    Transformer, has "encoder layers" (instead of one encoder we have few). Each "encoder layer" (or just an encoder) is very similar to other or all encoders have same architecture. 
    Each encoder or "encoder layer" consists of two layers: Self-attention and a feed Forward Neural Network. 

    As is the case in NLP applications in general, we begin by turning each input word into a vector.
    After embedding the words in input sequence, each of them flows through each of the two layers of the encoder.   
    The embedding only happens in the bottom most encoder, but in other encoders, it would be the output of the encoder that is directly below.

    Each encoder layer in a transformer consists of:
    - Multi-head self-attention(attention.hh).
    - A feedforward network (FFN, EncoderFeedForwardNetwork.hh).
    - Layer normalization and dropout rate.    
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH

/*
template <typename t>
class EncoderLayer;
 
template <typename t = double>
struct EncoderLayerList
{
*/
    /*
        Transformer encoder layer
     */
/*    
    class EncoderLayer<t>* ptr; 

    struct EncoderLayerList<t>* next;
    struct EncoderLayerList<t>* previous;
};
*/

/*
    The encoder consists of many encoder layers.
 */
template <typename t = double>
class EncoderLayer
{       
    Attention<t> attention;
    EncoderFeedForwardNetwork<t> ffn; // Forward Feed Network
    /*
        The two layer norms correspond to:
        After self-attention and feed-forward network, we apply layer normalization to stabilize the training process.
        
        Layer Normalization
        -------------------
        - Layer normalization is a technique to normalize the inputs across the features for each training example.
        - It helps stabilize and accelerate training by reducing internal covariate shift.
        - It is applied after the multi-head attention and feed-forward network in the encoder layer.
        - The normalization is done independently for each training example, making it suitable for variable-length sequences.
     */
    EncoderLayerNormalization<t> /*norm1*/ attention_norm, /*norm2*/ ffn_norm; // Layer Normalization
    
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfAttentionHeads;
    t dropOutRate;

    MultiHeadAttentionList<t>* multiHeadAttentionListHead;

    public:
        //EncoderLayer() : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER), attention(), ffn(dimensionsOfTheModel, dropOutRate), norm1(dimensionsOfTheModel), norm2(dimensionsOfTheModel)
        //EncoderLayer() :  attention(), ffn(dimensionsOfTheModel, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER), norm1(dimensionsOfTheModel), norm2(dimensionsOfTheModel), dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER) 
        /*EncoderLayer() : 
            dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER),
            numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), 
            dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER),
            attention(dimensionsOfTheModel, numberOfAttentionHeads),
            ffn(dimensionsOfTheModel, dropOutRate),
            norm1(dimensionsOfTheModel),
            norm2(dimensionsOfTheModel) 
        {  
            EncoderLayerNormalization::ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE; // Replaces macro, type-safe
        }*/

        EncoderLayer(void) 
            : attention(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER),
              ffn(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER),
              /*norm1*/ attention_norm(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER),
              /*norm2*/ ffn_norm(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER),
              dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER),
              numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER),
              dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER),
              multiHeadAttentionListHead(NULL)
        {
            /*
                Layer Normalization Epsilon Call: The following line doesn’t actually do anything
             */
            EncoderLayerNormalization::ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE; // Replaces macro, type-safe
            
            MultiHeadAttentionList<t>* current = NULL;

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfAttentionHeads; i++)
            {
                if (current == NULL)
                {                    
                    current = reinterpret_cast<MultiHeadAttentionList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(MultiHeadAttentionList<t>)));
                    multiHeadAttentionListHead = current;
                    current->previous = NULL;                    
                }
                else
                {                 
                    current->next = reinterpret_cast<MultiHeadAttentionList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(MultiHeadAttentionList<t>)));
                    current->next->previous = current;
                    current = current->next;
                }
                
                current->next = NULL;
                /*
                    In a Transformer-based encoder (such as in BERT or GPT-like models), each encoder layer consists of multiple sublayers:
                    1. Self-Attention Layer, it consists of multiple attention heads 
                    2. Feedforward Layer
                    3. Layer Normalization (before or after these)
                 */    
                current->ptr = new Attention<t>(dimensionsOfTheModel, numberOfAttentionHeads);     
            }
        }      

        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads. 
            @dropout_rate, Dropout rate for regularization. The dropout_rate in the Transformer model is a regularization technique to prevent overfitting.
         */
        //EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : dimensionsOfTheModel(d_model), dropOutRate(dropout_rate), attention(d_model, num_heads), ffn(d_model, dropout_rate), norm1(d_model), norm2(d_model)
        /*EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate)
            : attention(d_model, num_heads),
              ffn(d_model, dropout_rate),
              norm1(d_model),
              norm2(d_model),
              dimensionsOfTheModel(d_model),
              numberOfAttentionHeads(num_heads),
              dropOutRate(dropout_rate)
        {                        
        }*/

        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads. 
            @dropout_rate, Dropout rate for regularization. The dropout_rate in the Transformer model is a regularization technique to prevent overfitting.
         */
        EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate)
            : attention(d_model, num_heads),      /* Initialize attention module */
              ffn(d_model, dropout_rate),         /* Initialize FeedForward Network */
              /*norm1*/ attention_norm(d_model),  /* Initialize Layer Normalization 1 */
              /*norm2*/ ffn_norm(d_model),        /* Initialize Layer Normalization 2 */
              dimensionsOfTheModel(d_model), 
              numberOfAttentionHeads(num_heads), 
              dropOutRate(dropout_rate),
              multiHeadAttentionListHead(NULL)
        {   
            MultiHeadAttentionList<t>* current = NULL;

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfAttentionHeads; i++)
            {
                if (current == NULL)
                {                    
                    current = reinterpret_cast<MultiHeadAttentionList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(MultiHeadAttentionList<t>)));
                    multiHeadAttentionListHead = current;
                    current->previous = NULL;                    
                }
                else
                {                 
                    current->next = reinterpret_cast<MultiHeadAttentionList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(MultiHeadAttentionList<t>)));
                    current->next->previous = current;
                    current = current->next;
                }
                
                current->next = NULL;
                /*
                    In a Transformer-based encoder (such as in BERT or GPT-like models), each encoder layer consists of multiple sublayers, typically:
                    1. Self-Attention Layer, it consits of multiple of attention heads
                    2. Feedforward Layer
                    3. Layer Normalization (before or after these)
                 */    
                current->ptr = new Attention<t>(dimensionsOfTheModel, numberOfAttentionHeads);     
            }
        }
        
        /**
         * @brief Forward pass through the Transformer Encoder Layer.
         *
         * This method processes the input through a multi-head self-attention mechanism followed by
         * a position-wise feed-forward network, with optional layer normalization applied before or
         * after each sub-layer depending on the specified normalization position strategy.
         *
         * @tparam t                The numeric type used in computations (e.g., float, double).
         * @param ei                Input tensor (Collective<t>) representing embedded input tokens.
         * @param mask              Attention mask to prevent attending to certain positions (e.g., padding).
         * @param norm_position     Specifies whether to apply layer normalization before (Pre-LN) or
         *                          after (Post-LN) the attention and feed-forward sub-layers.
         *                          Options are:
         *                            - PreAttentionAndFeedForwardNetwork (Pre-LN)
         *                            - PostAttentionAndFeedForwardNetwork (Post-LN)
         * @param is_training       Indicates whether the model is in training mode.
         *                          Used to toggle dropout and gradient-related logic.
         *
         * @return Collective<t>    Output tensor after applying attention, residual connections,
         *                          feed-forward network, and normalization.
         *
         * @throws ala_exception    If any internal computation fails or input constraints are violated.
         */       
        Collective<t> forward(Collective<t>& ei, Collective<t>& mask, Collective<t>& attentionMaskInputSequence, ENCODER_LAYER_NORM_POSITION_TYPE norm_position = PreAttentionAndFeedForwardNetwork, bool is_training = true) throw (ala_exception)
        {

            /*std::cout<< "--> " << ei.getShape().getNumberOfColumns() << ", " << ei.getShape().getNumberOfRows() << std::endl;
            std::cout<< "Floor " << floor((t)(ei.getShape().getN()/DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER)) << std::endl;*/

            /*
                The output of MULTIHEADATTENTION::forward() is typically a transformed representation of the input sequence, where each token's representation has been updated based on attention over all tokens in the sequence. In a Transformer encoder, this output is usually processed further in the following steps:

                What to Do with the Output of MULTIHEADATTENTION::forward()?
                1. Pass it to the Feed-Forward Network (FFN)
                   - The multi-head self-attention output is fed into a position-wise feed-forward network (FFN).
                   - In this implementation it corespondes to passing it to EncoderFeedForwardNetwork::forward():
                     EncoderFeedForwardNetwork::forward(output);

                2. Apply Layer Normalization
                   - After adding a residual connection (if applicable), the output is normalized.
                   - In this implementation it correspondes to EncoderLayerNormalization::forward():
                     EncoderLayerNormlization::forward(output);
                
                3. Add Residual Connection (Optional, but Important in Transformer)
                   - If this is a standard Transformer encoder implementation, then add a residual connection before applying layer normalization.
                   output = input + output;  // Residual connection
                   EncoderLayerNormalization::forward(output);
                
                4. Use it as Input for the Next Encoder Layer (If Using Stacked Encoders)
                   - If Transformer encoder implementation has multiple layers, the output of this layer becomes the input to the next encoder layer.
                
                5. Pass to the Decoder (If Implementing an Encoder-Decoder Model)
                   - If this is part of an encoder-decoder Transformer (e.g., for machine translation), the final encoder output will be used as input to the decoder.   
             */
            Collective<t> output, residual /*Store the output for the residual connection*/;

            try 
            {
                /*
                    Apply attention mechanism with residual connection

                    The forward method of the ENCODERLAYER class call the forward method of the MULTIHEADATTENTION class with the same argument ei(encoder input) passed three times for query, key, and value.
                    This might seem redundant at first glance, but there's a specific reason for it.

                    While it may seem like a repetition, using the same argument for query, key, and value in the MultiHeadAttention call 
                    enables self-attention, a fundamental mechanism for Transformers to understand the relationships within a sequence.

                    Read more about in the comment section of MULTIHEADATTENTION::forward()
    
                    ====================== SELF-ATTENTION EXPLANATION ======================
                    Self-attention is a key component of Transformer models. In this context,
                    we pass the same input (ei) as the query, key, and value to the attention 
                    mechanism:

                    output = attention.forward(ei, ei, ei, mask);

                    This is not a mistake—this is how *self*-attention works by design.

                    Why pass the same input for all three?
                    -------------------------------------
                    - Query (Q): What we are searching for in the input.
                    - Key   (K): What we are matching against.
                    - Value (V): What we use to compute the final output.

                    In self-attention, we want each token (or word) in the input sequence to:
                    - Look at all other tokens (including itself),
                    - Compute how much attention it should give to each,
                    - And then create a new, weighted representation of itself.

                    By using the same input for Q, K, and V, we allow the model to:
                    - Capture dependencies and relationships between all tokens,
                    - Learn which other tokens are important for understanding each token's context,
                    - Dynamically adapt based on sequence position and content.

                    This enables the model to "pay attention" to relevant tokens no matter where 
                    they appear in the sequence—something especially useful in NLP tasks like 
                    translation, summarization, or sentiment analysis.

                    Summary:
                    --------
                    Self-attention = Attention(query=input, key=input, value=input)
                    This is what allows the model to understand each token in the *context* of all others.

                    For example:
                    Input sentence: "The cat sat on the mat"
                    The representation for "sat" will consider its attention to "cat", "on", and even "mat",
                    allowing the model to understand its role better in the sentence.
                    =========================================================================
                 */                                           
                    /*output = attention.forward(ei, ei, ei, mask);*/ // Attention output 

                // For Pre-LN, you already have attention_norm applied before (not shown in the code snippet)
                // Add the residual connection
                /*output = ei + output;*/ // Residual connection around attention
                /*
                    The output of the attention mechanism is passed through a layer normalization step.
                    This helps stabilize the training process and improve convergence.
                    The layer normalization is applied to the output of the attention mechanism.
                 */

                if (norm_position == PreAttentionAndFeedForwardNetwork)
                {
                    // Pre-LN for attention
                    residual = attention_norm.forward(ei);  // Normalize first
                    residual = attention.forward(residual, residual, residual/*, mask*/, attentionMaskInputSequence);
                    output = ei + residual;  // Add residual connection
                }
                else if (norm_position == PostAttentionAndFeedForwardNetwork)
                {
                    // Post-LN for attention
                    residual = attention.forward(ei, ei, ei/*, mask*/, attentionMaskInputSequence);
                    output = ei + residual;  // Add residual connection
                    output = attention_norm.forward(output);  // Normalize after residual
                }
 
                /*
                    Valid only for Post-LN, where layer normalization is applied after the attention sublayer and the residual connection.

                    Apply layer normalization after attention
                    The output of the attention mechanism is passed through a layer normalization step.
                    This helps stabilize the training process and improve convergence.
                    The layer normalization is applied to the output of the attention mechanism.


                    However, if you're introducing configurability for Pre-LN vs Post-LN (like you did for the feed-forward sublayer), you must wrap this in a conditional, just like you did before.
                 */
                if (norm_position == PostAttentionAndFeedForwardNetwork)
                {
                    /*output =*/ /*norm1*/ /*attention_norm.forward(output);*/ 
                }

                if (is_training)
                {
#ifdef STRESS_TEST_BACKWARD_PASS_IN_FORWARD_PASS
                    // Backpropagation logic for layer normalization
                    // The backward() method of the EncoderLayerNormalization class is called to compute gradients for the layer normalization step and it should not be here                     
                    output = attention.backward(output); // Backpropagation logic for attention layer
#endif                    
                }
                                
                /*output = ei + output;*/ // Residual connection around attention
                
                        /*output = ffn.forward(output); // Feed-forward network output*/

                if (is_training)
                {
#ifdef STRESS_TEST_BACKWARD_PASS_IN_FORWARD_PASS
                    // Backpropagation logic for layer normalization
                    // The backward() method of the EncoderLayerNormalization class is called to compute gradients for the layer normalization step and it should not be here                                        
                    output = ffn.backward(output); // Backpropagation logic for feed-forward network
#endif                      
                }

                // Apply layer normalization
                        //output = /*norm1*/ attention_norm.forward(output); // Layer normalization after attention
                /*
                    The encoder layer should only call backward() when running in training mode and,
                    during training, gradients will flow in the reverse order

                    // -----------------------------------------------------------------------------
                    // NOTE: Backward calls inside forward() for stress testing only
                    //
                    // In most deep learning libraries (e.g., PyTorch, TensorFlow), forward propagation 
                    // is responsible for computing and storing intermediate values such as activations 
                    // and inputs, while backward propagation is triggered later — typically via a call 
                    // like loss.backward(). This separation allows for flexibility, proper gradient 
                    // flow through the entire model, and memory management via a computation graph.
                    //
                    // However, in this custom implementation, backward() calls are placed inside 
                    // the forward() method temporarily **for stress testing purposes only**. 
                    // This allows us to immediately verify the correctness of the attention and 
                    // feed-forward layers' gradient calculations right after the forward pass.
                    //
                    // This is not standard practice and should not be used in production training code. 
                    // In a full training setup, backward() should be invoked separately and in coordination 
                    // with a loss function and optimizer.
                    //
                    // -----------------------------------------------------------------------------
                 */
                if (is_training)
                {
#ifdef STRESS_TEST_BACKWARD_PASS_IN_FORWARD_PASS
                    // Backpropagation logic for layer normalization
                    // The backward() method of the EncoderLayerNormalization class is called to compute gradients for the layer normalization step and it should not be here                     
                    output = /*norm1*/ attention_norm.backward(output);
#endif
                }

                Collective<t> residual; // Store the output for the residual connection
                if (norm_position == PreAttentionAndFeedForwardNetwork)
                {
                    // Apply layer normalization before feed-forward network
                        //residual = /*norm2*/ ffn_norm.forward(output); // Layer normalization before feed-forward network
                        //output = output + residual; // Residual connection around FFN

                    // Apply feed-forward network with residual connection
                        //residual = ffn.forward(output); // Feed-forward network output
                        //output = output + residual; // Residual connection around FFN
                    // Apply layer normalization after feed-forward network*/

                    // Pre-LN: Normalize before feed-forward network
                    output = ffn_norm.forward(output);             // Layer norm before FFN
                    /*Collective<t>*/ residual = ffn.forward(output);  // Apply FFN
                    output = output + residual;                    // Add residual                    
                }
                else if (norm_position == PostAttentionAndFeedForwardNetwork)
                {
                    // Apply feed-forward network with residual connection
                        //residual = ffn.forward(output); // Feed-forward network output
                        //output = output + residual; // Residual connection around FFN
                    // Apply layer normalization after feed-forward network
                        //residual = /*norm2*/ ffn_norm.forward(output); // Layer normalization after feed-forward network
                        //output = output + residual; // Residual connection around FFN

                    residual = ffn.forward(output);  // Apply FFN
                    output = output + residual;                    // Add residual
                    output = ffn_norm.forward(output);             // Layer norm after residual (Post-LN)
                }

                /*
                    Collective<t> residual = ffn.forward(output); // Feed-forward network output
                    output = output + residual; // Residual connection around FFN
                    output = ffn_norm.forward(output); // Apply layer normalization after feed-forward network
                 */

                    /*Collective<t> residual = ffn_norm.forward(output); // Feed-forward network output
                    output = output + residual; // Residual connection around FFN*/
                /*output = ffn.forward(output);*/ // Feed-forward network output 

                // Apply feed-forward network with residual connection                
                // ************************************************************************************ //
                //  The following commented statement has been replaced with the two statements below.  //
                //  This change was made due to the reason explained in the comment block of the        //
                //  operator+ overloaded method in the Numcy::Collective parameter class.               //
                // ************************************************************************************ //
                /*
                    output = output + ffn.forward(output); // Residual connection around FFN                    
                 */
                // The output of the feed-forward network is added to the input of the feed-forward network (output) to create a residual connection.
                // This residual connection helps the model learn better by allowing gradients to flow more easily during backpropagation
                // and helps mitigate the vanishing gradient problem.         
                    /*Collective<t>*/ /*residual = ffn.forward(output); // Feed-forward network output
                    output = output + residual; // Residual connection around FFN*/
                                
                // The output of the feed-forward network is then passed through layer normalization to stabilize the training process.                
                    //output = /*norm2*/ ffn_norm.forward(output); // Apply layer normalization after feed-forward network

                /*
                    The encoder layer should only call backward() when running in training mode and,
                    during training, gradients will flow in the reverse 
                    
                    // -----------------------------------------------------------------------------
                    // NOTE: Backward calls inside forward() for stress testing only
                    //
                    // In most deep learning libraries (e.g., PyTorch, TensorFlow), forward propagation 
                    // is responsible for computing and storing intermediate values such as activations 
                    // and inputs, while backward propagation is triggered later — typically via a call 
                    // like loss.backward(). This separation allows for flexibility, proper gradient 
                    // flow through the entire model, and memory management via a computation graph.
                    //
                    // However, in this custom implementation, backward() calls are placed inside 
                    // the forward() method temporarily **for stress testing purposes only**. 
                    // This allows us to immediately verify the correctness of the attention and 
                    // feed-forward layers' gradient calculations right after the forward pass.
                    //
                    // This is not standard practice and should not be used in production training code. 
                    // In a full training setup, backward() should be invoked separately and in coordination 
                    // with a loss function and optimizer.
                    //
                    // -----------------------------------------------------------------------------
                 */
                if (is_training)
                {
#ifdef STRESS_TEST_BACKWARD_PASS_IN_FORWARD_PASS
                    // Backpropagation logic for layer normalization
                    // The backward() method of the EncoderLayerNormalization class is called to compute gradients for the layer normalization step and it should not be here                       
                    output = /*norm2*/ ffn_norm.backward(output);
#endif  
                }
            }
            catch(ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayer::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }

            return output;
        }

        ~EncoderLayer()
        {
            if (multiHeadAttentionListHead != NULL)
            {
                MultiHeadAttentionList<t>* current = multiHeadAttentionListHead;
                MultiHeadAttentionList<t>* next;
                                
                while (current != NULL)
                {
                    next = current->next;

                    /*
                        The `delete[]` operator is used for deleting arrays allocated with `new[]`.
                        *(current->ptr) is a single object (not an array), you should use `delete` instead of `delete[]`.
                     */
                    delete current->ptr;
                    current->ptr = NULL;

                    cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current), sizeof(MultiHeadAttentionList<t>));
                    current = next;                    
                }                                
            }           
        }
};

#endif