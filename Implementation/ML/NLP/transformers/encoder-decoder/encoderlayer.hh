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

    // Projection matrices for Q, K, V (respectively W^Q, W^K, W^V) and "output projection weights" matrix in back propogation it is known as "W^O"
    Collective<t> queryWeights, keyWeights, valueWeights, outputWeights;

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

        /*
            // Input: [batch, seq_len, d_model]
            // Mask: [batch, seq_len, d_model]
         */
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
        Collective<t> forward(Collective<t>& ei, Collective<t>& attentionMaskInputSequence, ENCODER_LAYER_NORM_POSITION_TYPE norm_position = PreAttentionAndFeedForwardNetwork, bool is_training = true) throw (ala_exception)
        {   
            /*  
             * Validate input dimensions
             * -------------------------
             * Each head needs to get a clean slice of the feature dimension. If the number of heads doesn't divide evenly into the feature size, then without padding or adjustments, some heads would end up with fractional features, which isn't valid.
             * Adding padding or some adjustment/resolution ensures that each head gets equal numbers of features, thus maintaining the integrity of the multi-head attention mechanism.
             * At the moment, an exception is just being thrown if the number of heads does not divide evenly into the feature size
             */           
            if (ei.getShape().getNumberOfColumns() % numberOfAttentionHeads)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayer<t>::forward(Collective<t>&, Collective<t>&, ENCODER_LAYER_NORM_POSITION_TYPE, bool) Error: The number of columns \"") + cc_tokenizer::String<char>(ei.getShape().getNumberOfColumns()) + cc_tokenizer::String<char>("\" must be evenly divisible by the number of attention heads \"") + cc_tokenizer::String<char>(numberOfAttentionHeads) + cc_tokenizer::String<char>("\" for multi-head attention."));
            }

            /*
             * Ensure the input feature dimension matches the model's expected dimension
             */
            if (ei.getShape().getNumberOfColumns() != dimensionsOfTheModel)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayer<t>::forward(Collective<t>&, Collective<t>&, ENCODER_LAYER_NORM_POSITION_TYPE, bool) Error: The number of input columns  \"") + cc_tokenizer::String<char>(ei.getShape().getNumberOfColumns()) + cc_tokenizer::String<char>("\" must be equal to the model dimension \"") + cc_tokenizer::String<char>(dimensionsOfTheModel) + cc_tokenizer::String<char>("\"."));
            }

            // Pointer to traverse the linked list of attention heads
            MultiHeadAttentionList<t>* current = multiHeadAttentionListHead;

            // Variables to manage array dimensions and slicing
            DIMENSIONSOFARRAY dimensionOfArray; 
            DIMENSIONS /*dimension_ei_slice,*/ dimension_qkv_weights, dimension_qkv_slice;

            // Collective objects for storing concatenated results and individual slices
            Collective<t> ei_concatenated/*, ei_slice*/, q_slice, k_slice, v_slice;
            // Counter for tracking slice positions
            cc_tokenizer::string_character_traits<char>::size_type i = 0;

            // Projection matrices for Q, K, V (respectively W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>) and "output projection weights" matrix in back propogation it is known as "W<sup>O</sup>"
            Collective<t> queryWeights, keyWeights, valueWeights, outputWeights; // In the original paper "output projection weights" has same shape as the other three weight matrices.
                                                                                 // In reality "output projection weights" should have the same shape projection weights for value input because these weights are multiplied with the output of product between attention scores and value weights

            Collective<t> w_q_slice, w_k_slice, w_v_slice;                                                                                 

            Collective<t> attention_head_output, attention_head_outputs, attention_output;

            Collective<t> x = ei;


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
            Collective<t> final_output, residual;

            try
            {  
                // ============================================================
                // STEP 1: MULTI-HEAD SELF-ATTENTION
                // ============================================================

                /*std::cout<< "::: DEBUG DATA -: Encoder Input(ei) :- :::"  << std::endl;
                std::cout<< "number_of_batches(batch_size): " << x.getShape().getDimensionsOfArray()[0] << ", number_of_inputs(seq_len): " << x.getShape().getDimensionsOfArray()[1] << ", size_of_each_input(feature_dim): " << x.getShape().getDimensionsOfArray()[2] << std::endl;
                std::cout<< "Columns: " << x.getShape().getNumberOfColumns() << ", Rows: " << x.getShape().getNumberOfRows() << std::endl;
                for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < ei.getShape().getN(); k++)
                {
                    std::cout<< x[ (k/(x.getShape().getDimensionsOfArray()[1]*x.getShape().getDimensionsOfArray()[2]))*(x.getShape().getDimensionsOfArray()[1]*x.getShape().getDimensionsOfArray()[2]) + (k - (k/(x.getShape().getDimensionsOfArray()[1]*x.getShape().getDimensionsOfArray()[2]))*(x.getShape().getDimensionsOfArray()[1]*x.getShape().getDimensionsOfArray()[2])) ] << " ";
                    if ((k + 1)%x.getShape().getDimensionsOfArray()[2] == 0)
                    {
                        std::cout<< std::endl;
                    }
                }*/
                
                // Pre-LN for attention
                if (norm_position == PreAttentionAndFeedForwardNetwork)
                {                    
                    x = attention_norm.forward(x);
                }

                /*
                 *  --- Multi-Head Attention Computation ---
                 *  Each head gets d_model/num_heads features
                 */

                // Shape of ei = [batch_size, seq_len, feature_dim] = [number_of_batches, number_of_inputs, size_of_each_input]
                // Each batch is divided into input sequence for each attention head... We have already made sure that feature_dim is evenly divisible by number_of_attention heads.
                // Each attention head will get feature_dim/number_of_attention_heads features for each input in the sequence... so
                // [batch_size, seq_len, feature_dim/number_of_attention_heads] for each attention head
                
                // Get the dimensions of the input array 'ei'      
                dimensionOfArray = ei.getShape().getDimensionsOfArray();

                /* 
                 * Divide the input 'ei' into equal slices for each attention head.
                 * Modify the last dimension to split across attention heads.
                 * Divide the column dimension by number of attention heads for equal partitioning
                 */ 
                    /*dimensionOfArray[dimensionOfArray.size() - 1] = ei.getShape().getNumberOfColumns() / numberOfAttentionHeads; // h in original paper*/                
                // Create dimension object with modified dimensions for slicing of 'ei' a.k.a encoder input
                    /*dimension_ei_slice = DIMENSIONS(dimensionOfArray);*/

                //std::cout<< "OK = " << dimension.getNumberOfColumns() << ",  = " << dimension.getN() << std::endl;

                /*
                 * Initialize weight matrices if they haven't been initialized yet
                 * This lazy initialization creates the W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> projection weights on first use
                 * Each weight matrix has same shape as the input 'ei' to this encoder layer
                 */
                if (queryWeights.getShape().getN() == 0 && keyWeights.getShape().getN() == 0 && valueWeights.getShape().getN() == 0 /*&& outputWeights.getShape().getN() == 0*/)
                {
                    // Set dimensions for all weight matrices to [batch_size, d_model, d_model]
                    // dimensionsOFaRRAY[dimensionOfArray.size() - 3] is already set to batch_size
                    dimensionOfArray[dimensionOfArray.size() - 2] =   ei.getShape().getDimensionsOfArray()[ei.getShape().getDimensionsOfArray().size() - 2] /*dimensionsOfTheModel*/; // d_model in original paper; nimber of rows
                    dimensionOfArray[dimensionOfArray.size() - 1] =   ei.getShape().getDimensionsOfArray()[ei.getShape().getDimensionsOfArray().size() - 1] /*dimensionsOfTheModel*/; // d_model in original paper; number of columns
                    dimension_qkv_weights = DIMENSIONS(dimensionOfArray);


                        /*std::cout<< dimension_qkv_weights.getNumberOfRows() << ", " << dimension_qkv_weights.getNumberOfColumns() << std::endl;*/
 
                    // Initialize W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> weight matrices with random values, they all will have same shape as ei (shape of input to this layer of encoder)
                    queryWeights = Numcy::Random::randn<t>(dimension_qkv_weights); 
                    keyWeights = Numcy::Random::randn<t>(dimension_qkv_weights); 
                    valueWeights = Numcy::Random::randn<t>(dimension_qkv_weights); // Value weights can have fewer or more fetures than input features. In original paper, they are same. Here we keep them same for simplicity.
 
                    /*
                     * Output projection weights
                     * W<sup>O</sup> has shape d_model×(d<sub>v</sub>·h), where h is the number of attention heads and d<sub>v</sub> is the dimension of the value vectors. 
                     * Since W<sup>V</sup> shape is no different than the other two projection weights (W<sup>Q</sup>, W<sup>K</sup>), the shape of W<sup>O</sup> will be same as W<sup>V</sup>
                     * We will not work on slices of these weights. This will be used as a right operand in a dot product operation, the other operand is concatenation of output of all (h many) attention heads
                     */
                    outputWeights = Numcy::Random::randn(dimension_qkv_weights);
                    
                    // Set dimensions for sliced weight matrices (per attention head, d_model/h), settng number of columns to d_model/number_of_attention_heads
                    dimensionOfArray[dimensionOfArray.size() - 1] = dimensionsOfTheModel / numberOfAttentionHeads; // d_q, d_k, d_v. d_q and d_k are interchangeable but d_v can be different.
                                                                                                                   // In the original paper, you would see d_k, d_k where d_q, d_k would have been used.  
                    dimension_qkv_slice = DIMENSIONS(dimensionOfArray);
                }
                
                // Iterate through all MultiHeadAttention modules in the linked list                
                while (current != NULL)
                {   
                    // AXIS_ROWS means we are slicing along and across rows vertically
                    // ---------------------------------------------------------------
                    // Extract a slice from input 'ei' starting at calculated position
                    // Each slice corresponds to one attention head's portion of the input                     
                            /*ei_slice = ei.slice(i*dimension_ei_slice.getNumberOfColumns(), dimension_ei_slice, AXIS_ROWS);*/

                    // Extract corresponding slices from Q, K, V weight matrices for this attention head
                    /*q_slice*/ w_q_slice = queryWeights.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS);
                    /*k_slice*/ w_k_slice = keyWeights.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS);
                    /*v_slice*/ w_v_slice = valueWeights.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS); 
                    

                        /*std::cout<< w_q_slice.getShape().getNumberOfRows() << ", " << w_q_slice.getShape().getNumberOfColumns() << std::endl;*/

                    //std::cout<< "-->>>>> "  << dimension_qkv_slice.getNumberOfColumns() << ", " << dimension_qkv_slice.getNumberOfRows() << "dims = " << numberOfAttentionHeads  << std::endl;


                    q_slice = ei.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS);
                    k_slice = ei.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS);
                    v_slice = ei.slice(i*dimension_qkv_slice.getNumberOfColumns(), dimension_qkv_slice, AXIS_ROWS);
                                              
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
                    // AXIS_COLUMN means we are concatenating horizontally (along columns)
                    // -------------------------------------------------------------------
                    attention_head_output = MultiHeadAttention<t>::worker(/*ei*/ q_slice, /*ei*/ k_slice, /*ei*/ v_slice, /*q_slice*/ w_q_slice, /*k_slice*/ w_k_slice, /*v_slice*/ w_v_slice);

                    /*std::cout<< "attention_head_output = " << attention_head_output.getShape().getDimensionsOfArray().size() << std::endl;*/

                    /*std::cout<< "w_q_slice = " << w_q_slice.getShape().getDimensionsOfArray().size() << ", cols = " << w_q_slice.getShape().getNumberOfColumns() << ", rows = " << w_q_slice.getShape().getNumberOfRows() << std::endl;
                    std::cout<< "q_slice = " << q_slice.getShape().getDimensionsOfArray().size() << ", cols = " << q_slice.getShape().getNumberOfColumns() << ", rows = " << q_slice.getShape().getNumberOfRows() << std::endl;
                    std::cout<< "attention_head_output dima = " << attention_head_output.getShape().getDimensionsOfArray().size() << ", cols = " << attention_head_output.getShape().getNumberOfColumns() << ", rows = " << attention_head_output.getShape().getNumberOfRows() << std::endl;*/
                    
                    //std::cout<< "attention_head_ouput OK = " << attention_head_output.getShape().getNumberOfColumns() << ",  = " << attention_head_output.getShape().getNumberOfRows() << std::endl;

                    // AXIS_COLUMN means we are concatenating horizontally (along columns)
                    // -------------------------------------------------------------------
                    /*
                     * Concatenate individual attention head outputs along feature dimension
                     *
                     * Each attention head produces context vectors of shape (sequence_length × d_v)
                     * By concatenating along columns (feature axis), we combine the outputs from
                     * all h attention heads into a unified representation:
                     *
                     * Result shape: (sequence_length × h·d_v)
                     *
                     * This concatenated output preserves the diverse contextual information
                     * captured by each attention head, maintaining their unique specialized
                     * representations before the final linear projection.
                     */
                    attention_head_outputs = Numcy::concatenate(attention_head_outputs, attention_head_output, AXIS_COLUMN);

                    /*std::cout<< "attention_head_outputs = " << attention_head_outputs.getShape().getDimensionsOfArray().size() << std::endl;*/

                    // AXIS_COLUMN means we are concatenating horizontally (along columns)
                    // -------------------------------------------------------------------
                    // Concatenate the current slice with previous slices along columns
                    // This builds up the complete processed output across all attention heads                    
                            /*ei_concatenated = Numcy::concatenate(ei_concatenated, ei_slice, AXIS_COLUMN);*/
                    
                    // AXIS_ROWS means we are slicing along and across rows vertically
                    // ---------------------------------------------------------------
                    // Update the weight matrices with the sliced portions
                    // This distributes different parts of the weight matrices to different attention heads
                    queryWeights.update(i*dimension_qkv_slice.getNumberOfColumns(), w_q_slice, AXIS_ROWS);
                    keyWeights.update(i*dimension_qkv_slice.getNumberOfColumns(), w_k_slice, AXIS_ROWS);
                    valueWeights.update(i*dimension_qkv_slice.getNumberOfColumns(), w_v_slice, AXIS_ROWS);
                                        
                    // Increment slice counter and move to next attention head
                    i = i + 1;
                    current = current->next;                                        
                }

                // Debug output to verify concatenated dimensions
                /*std::cout<< "Concatenated OK = " << ei_concatenated.getShape().getNumberOfColumns() << ",  = " << ei_concatenated.getShape().getNumberOfRows() << std::endl;*/                
                /*std::cout<< "attention_head_ouputs OK = " << attention_head_outputs.getShape().getNumberOfColumns() << ",  = " << attention_head_outputs.getShape().getNumberOfRows() << std::endl;*/

                /*for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < ei.getShape().getN(); k++)
                {
                    if (ei_concatenated[k] == ei[k])
                    {
                        std::cout<< "Mismatch at index " << k << ": ei_concatenated = " << ei_concatenated[k] << ", ei = " << ei[k] << std::endl;
                    }
                }*/

                /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfAttentionHeads; i++)
                {
                    ei_slice = ei.slice(i*dimension.getNumberOfColumns(), dimension, AXIS_ROWS);    

                    std::cout<< ei_slice.getShape().getN() << std::endl;

                    std::cout<< ei_slice.getShape().getDimensionsOfArray().size() << std::endl;
                }*/

                // Apply output projection: W^O
                // This projects concatenated head outputs back to d_model dimension
                attention_output = Numcy::dot(attention_head_outputs, outputWeights);   
                
                // Apply dropout if training
                if (is_training && dropOutRate > 0)
                {
                   // Apply dropout to attention_output
                   // (You need to implement dropout mechanism)
                }

                /*
                    The output of the attention mechanism is passed through a layer normalization step.
                    This helps stabilize the training process and improve convergence.
                    The layer normalization is applied to the output of the attention mechanism.
                 */
                // ============================================================
                // STEP 2: ADD & NORM (First Residual Connection)
                // ============================================================
                if (norm_position == PreAttentionAndFeedForwardNetwork)
                {
                    // Pre-LN: Add residual to original input (before normalization)
                    residual = ei + attention_output;
                } 
                else 
                {
                    // Post-LN: Add residual, then normalize
                    residual = ei + attention_output;
                    residual = attention_norm.forward(residual);
                }

                // ============================================================
                // STEP 3: FEED-FORWARD NETWORK
                // ============================================================

                Collective<t> ffn_input = residual;

                // Pre-LN: Normalize BEFORE FFN
                if (norm_position == PreAttentionAndFeedForwardNetwork)
                {
                    ffn_input = ffn_norm.forward(residual);
                }

                // Apply feed-forward network
                Collective<t> ffn_output = ffn.forward(ffn_input);
    
                // Apply dropout if training
                if (is_training && dropOutRate > 0)
                {
                    // Apply dropout to ffn_output
                }

                // ============================================================
                // STEP 4: ADD & NORM (Second Residual Connection)
                // ============================================================

                //Collective<t> final_output;
    
                if (norm_position == PreAttentionAndFeedForwardNetwork)
                {
                    // Pre-LN: Add residual to input of FFN block
                    final_output = residual + ffn_output;
                }
                else
                {
                    // Post-LN: Add residual, then normalize
                    final_output = residual + ffn_output;
                    final_output = ffn_norm.forward(final_output);
                }
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayer<t>::forward(Collective<t>&, Collective<t>&, ENCODER_LAYER_NORM_POSITION_TYPE, bool) -> ") + e.what());
            }

            //std::cout<< "Columns = " << ei.getShape().getNumberOfColumns() << ", Rows = " << ei.getShape().getNumberOfRows() << std::endl;

            //std::cout<< ei.getShape().getDimensionsOfArray()[ei.getShape().getDimensionsOfArray().size() - 1] << std::endl;
            //std::cout<< attentionMaskInputSequence.getShape().getDimensionsOfArray()[attentionMaskInputSequence.getShape().getDimensionsOfArray().size() - 1] << std::endl;

            //return Collective<t>{NULL, DIMENSIONS{0, 0, NULL, NULL}};

            // Return the concatenated result from all attention heads
            return /*ei_concatenated*/ /*attention_head_outputs*/ /*ei*/ /*x*/ final_output;
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
        Collective<t> forward_old(Collective<t>& ei, Collective<t>& mask, Collective<t>& attentionMaskInputSequence, ENCODER_LAYER_NORM_POSITION_TYPE norm_position = PreAttentionAndFeedForwardNetwork, bool is_training = true) throw (ala_exception)
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