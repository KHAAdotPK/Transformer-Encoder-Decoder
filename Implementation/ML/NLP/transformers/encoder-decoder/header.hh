/*
    ML/NLP/transformers/encoder-decoder/header.hh
    Q@khaa.pk
 */

#include "./../../../../lib/argsv-cpp/lib/parser/parser.hh"
#include "./../../../../lib/corpus/corpus.hh"
#include "./../../../../lib/sundry/cooked_read_new.hh"
#include "./../../../../lib/read_write_weights/header.hh"
#include "./../../../../lib/Numcy/header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH

#ifdef INDEX_ORIGINATES_AT_VALUE
#undef INDEX_ORIGINATES_AT_VALUE
#endif
#define INDEX_ORIGINATES_AT_VALUE 4

/* ************************************************************************************************************************************* */
/*                                      EXPLANAION OF decoder_input, decoder_mask STARTS HERE                                            */
/* ************************************************************************************************************************************* */ 
/*
    // Decoder forward method takes the following parameters:
    // - `input`: The input sequence to the decoder, typically a tensor of shape (batch_size, sequence_length, d_model).
    // - `encoder_output`: The output from the encoder, which provides context for the decoder. 
    // - `encoder_mask`: A mask to prevent the decoder from attending to certain positions in the encoder output.
    // - `decoder_mask`: A mask to prevent the decoder from attending to certain positions in its own output.
 */
#define DECODER_INPUT_PAD_VALUE 0
#define DECODER_INPUT_BEGINNING_OF_SEQUENCE 1 
#define DECODER_INPUT_END_OF_SEQUENCE 2
#define DECODER_INPUT_UNK_VALUE 3

/*
    1. decoder_input (parameter 1):
       - This IS the target input (shifted right during training)
       - Contains the target sequence tokens that decoder should generate
       - During TRAINING: Target sequence shifted right with <START> token
         Example: Target = "Hello World" -> decoder_input = "<START> Hello World"
       - During INFERENCE: Previously generated tokens + current prediction
       - Shape: [batch_size, target_sequence_length, d_model]
       - Gets processed through masked self-attention (can't see future tokens)

    3. decoder_mask (Look-ahead mask) (parameter 3):
       - Prevents decoder from attending to future tokens
       - Lower triangular matrix (causal mask)
       - Ensures autoregressive property during training
       - Shape: [target_seq_len, target_seq_len]
       - Example for sequence length 4:
         [[1, 0, 0, 0],    # Token 0 can only see token 0
          [1, 1, 0, 0],    # Token 1 can see tokens 0,1
          [1, 1, 1, 0],    # Token 2 can see tokens 0,1,2
          [1, 1, 1, 1]]    # Token 3 can see tokens 0,1,2,3   

    Example Target Corpus:
    Sentence 1: "I love cats"
    Sentence 2: "I love dogs"
    Sentence 3: "I hate cats"
    Sentence 4: "You love programming"
    Sentence 5: "We hate programming"
    
    STEP 2: EXTRACT ALL UNIQUE WORDS
    Unique words found: {"I", "love", "cats", "dogs", "hate", "You", "We", "programming"}
    
    STEP 3: CREATE VOCABULARY WITH SPECIAL TOKENS
    
    Token ID | Token        | Purpose
    ---------|--------------|------------------
    0        | "<PAD>"      | Padding (for batching)
    1        | "<START>"    | Start of sequence
    2        | "<END>"      | End of sequence
    3        | "<UNK>"      | Unknown words
    4        | "I"          | Regular vocabulary
    5        | "love"       | Regular vocabulary
    6        | "cats"       | Regular vocabulary
    7        | "dogs"       | Regular vocabulary
    8        | "hate"       | Regular vocabulary
    9        | "You"        | Regular vocabulary
    10       | "We"         | Regular vocabulary
    11       | "programming"| Regular vocabulary
    
    STEP 4: TOKENIZE ALL SENTENCES USING SAME VOCABULARY
    
    Original Sentences -> Token IDs
    "I love cats"       -> [1, 4, 5, 6, 2]     // <START> I love cats <END>
    "I love dogs"       -> [1, 4, 5, 7, 2]     // <START> I love dogs <END>
    "I hate cats"       -> [1, 4, 8, 6, 2]     // <START> I hate cats <END>
    "You love programming" -> [1, 9, 5, 11, 2]  // <START> You love programming <END>
    "We hate programming"  -> [1, 10, 8, 11, 2] // <START> We hate programming <END>
    
    NOTICE THE CONSISTENCY:
    - Word "I" ALWAYS gets token ID 4 (in sentences 1, 2, 3)
    - Word "love" ALWAYS gets token ID 5 (in sentences 1, 2, 4)
    - Word "cats" ALWAYS gets token ID 6 (in sentences 1, 3)
    - Word "programming" ALWAYS gets token ID 11 (in sentences 4, 5)
 */
/*
    The start and end tokens (like <BOS> for beginning of sequence and <EOS> for end of sequence) are not related to batch size, they're related to sequence structure and the decoding process itself.
    Purpose of Start/End Tokens
    Start Token (<BOS>, <START>, etc.):
    - Signals to the decoder where to begin generating the output sequence.
    - During training, the decoder input is shifted right, meaning the first token is always the start token.
    - It helps the model understand that it should start generating from this point.
    - Provides initial context for the first token prediction
    - Essential for the autoregressive nature of decoding
    - Example: If the target sequence is "Hello World", the decoder input during training would be "<START> Hello World".
    
    End Token (<EOS>, <END>, etc.):
    - Signals to the decoder when to stop generating tokens.
    - During training, the decoder learns to predict this token when it has completed generating the sequence
    - It helps the model understand when to stop generating further tokens.
    - Example: If the target sequence is "Hello World", the decoder input during training would be "<START> Hello World <END>".
    - The model learns to predict the end token when it has completed generating the sequence.
    - It is crucial for tasks like text generation, where the model needs to know when to stop producing output.
    - Prevents the model from generating infinite sequences  
 
    Why Batch Size Doesn't Matter
    Whether you have:
        Batch size = 1: [<BOS> token1 token2 ... tokenN <EOS>]
        Batch size = 32: 32 sequences, each still needing [<BOS> ... <EOS>]

    Each sequence in the batch needs its own start/end tokens because:

    1. The decoder processes each sequence independently
    2. Each sequence needs to know its own boundaries
    3. The attention mechanism relies on these positional cues       
 */
/*
    DECODER MASK STRUCTURE:
    
    For sequence "I love cats" with tokens [1, 4, 5, 6, 2]:
    Position:  0(<START>)  1(I)  2(love)  3(cats)  4(<END>)
    
    Decoder Mask (Lower Triangular Matrix):
    ```
         0  1  2  3  4
    0 [  1  0  0  0  0 ]  # <START> can only see <START>
    1 [  1  1  0  0  0 ]  # I can see <START>, I
    2 [  1  1  1  0  0 ]  # love can see <START>, I, love
    3 [  1  1  1  1  0 ]  # cats can see <START>, I, love, cats
    4 [  1  1  1  1  1 ]  # <END> can see all previous tokens
    ```    
    Values: 1 = allowed to attend, 0 = masked (not allowed)
 */
/* ************************************************************************************************************************************* */
/*                                        EXPLANAION OF decoder_input, decoder_mask ENDS HERE                                            */
/* ************************************************************************************************************************************* */   

/* 
    The position of the layer normalization in the encoder layer.
    - Pre: Layer normalization is applied before the attention and feed-forward network.
    - Post: Layer normalization is applied after the attention and feed-forward network.
    - In the original Transformer paper, layer normalization is applied after the attention and feed-forward networks (Post).
    - In some implementations, it may be applied before (Pre) for different reasons, such as improving convergence or stability.
 */
typedef enum { PreAttentionAndFeedForwardNetwork, PostAttentionAndFeedForwardNetwork } ENCODER_LAYER_NORM_POSITION_TYPE;

/*
    ADHOC_IMPLEMENTATION_OF_MASK(instance, mask)

    This macro applies a masking operation on `instance` based on the values in `mask`.

    Parameters:
    - instance: A tensor or matrix-like structure with a defined shape.
    - mask: A binary vector (or array) where:
        - `mask[i] == 0` means the corresponding row in `instance` should be zeroed out.
        - `mask[i] == 1` means the row remains unchanged.

    Algorithm:
    1. Iterate over all elements in `instance` using `k`, which represents a flattened index.
    2. When reaching the last column of a row (`(k + 1) % instance.getShape().getNumberOfColumns() == 0`):
        - Determine the corresponding row index (`k / instance.getShape().getNumberOfColumns()`).
        - If `mask[row_index] == 0`, zero out all elements in that row.
    3. Skip ahead by a full row (`k += instance.getShape().getNumberOfColumns()`) after modifying a row.

    Notes:
    - The macro ensures that if a row is marked by `mask` as zero, all its elements are explicitly set to `0`.
    - The use of `cc_tokenizer::string_character_traits<char>::size_type l` for looping over columns ensures compatibility with various data types.
    - The macro modifies `instance` in-place, meaning it directly alters the input tensor/matrix.

    Example:
        Given `instance` as:
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        and `mask = [1, 0, 1]`,
        after applying the macro:
            [[1, 2, 3],
             [0, 0, 0],  // This row is zeroed out because mask[1] == 0
             [7, 8, 9]]
*/
#define ADHOC_IMPLEMENTATION_OF_MASK_QUERY(instance, mask, mask_with_zero_or_lowest_value)\
{\
    try\
    {\
        for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < instance.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); k++)\
        {\
            if (mask[k] == 0)\
            {\
                for (cc_tokenizer::string_character_traits<char>::size_type l = 0; l < instance.getShape().getNumberOfColumns(); l++)\
                {\
                    if (mask_with_zero_or_lowest_value == false)\
                    {\
                        instance[k*instance.getShape().getNumberOfColumns() + l] = std::numeric_limits<t>::lowest();\
                    }\
                    else\
                    {\
                        instance[k*instance.getShape().getNumberOfColumns() + l] = 0;\
                    }\
                    /*instance[k*instance.getShape().getNumberOfColumns() + l] = std::numeric_limits<t>::lowest()*/ /*0*/;\
                }\
            }\
        }\
    }\
    catch (ala_exception& e)\
    {\
        throw ala_exception(cc_tokenizer::String<char>("ADDHOC_IMPLEMENTATION_OF_MASK() -> ") + cc_tokenizer::String<char>(e.what()));\
    }\
}\

#define ADHOC_IMPLEMENTATION_OF_MASK_KEY(instance, mask, mask_with_zero_or_lowest_value)\
{\
    try\
    {\
        for (cc_tokenizer::string_character_traits<char>::size_type k = 0; k < mask.getShape().getNumberOfColumns(); k++)\
        {\
            if (mask[k] == 0)\
            {\
                for (cc_tokenizer::string_character_traits<char>::size_type l = 0; l < instance.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); l++)\
                {\
                    if (mask_with_zero_or_lowest_value == false)\
                    {\
                        instance[l*instance.getShape().getNumberOfColumns() + k] = std::numeric_limits<t>::lowest();\
                    }\
                    else\
                    {\
                        instance[l*instance.getShape().getNumberOfColumns() + k] = 0;\
                    }\
                    /*instance[l*instance.getShape().getNumberOfColumns() + k] = std::numeric_limits<t>::lowest()*/ /*0*/;\
                }\
            }\
        }\
    }\
    catch (ala_exception& e)\
    {\
        throw ala_exception(cc_tokenizer::String<char>("ADDHOC_IMPLEMENTATION_OF_MASK() -> ") + cc_tokenizer::String<char>(e.what()));\
    }\
}\

#define ADHOC_DEBUG_MACRO(instance)\
{\
    std::cout<< "::: ADHOC DEBUG DATA -: Columns: " << instance.getShape().getNumberOfColumns() << ", Rows: " << instance.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " :- :::"  << std::endl;\
    for (int k = 0; k < instance.getShape().getN(); k++)\
    {\
        std::cout<< instance[(k/instance.getShape().getNumberOfColumns())*instance.getShape().getNumberOfColumns() + (k%instance.getShape().getNumberOfColumns())] << " ";\
        if ((k + 1)%instance.getShape().getNumberOfColumns() == 0)\
        {\
            std::cout<< std::endl;\
        }\
    }\
}\

/*
    In transformers, a common technique for incorporating sequence information is by adding positional encodings to the input embeddings.
    The positional encodings are generated using sine and cosine functions of different frequencies.
    The expression wrapped in the following macro is used to scale the frequency of the sinusoidal functions used to generate positional encodings. 

    div_term
    ----------    
    This expression is used in initializing "div_term".

    The expression is multiplyed by -1
    ------------------------------------
    The resulting "div_term" array contains values that can be used as divisors when computing the sine and cosine values for positional encodings.
    
    Later on "div_term" and "positions" are used with sin() cos() functions to generate those sinusoidal positional encodings.
    The idea is that by increasing the frequency linearly with the position,
    the model can learn to make fine-grained distinctions for smaller positions and coarser distinctions for larger positions.
    
    @sfc, Scaling Factor Constant.
    @d_model, Dimensions of the transformer model.
 */
#define SCALING_FACTOR(sfc, d_model) -1*(log(sfc)/d_model)

/*
    It's worth noting that there's no universally "correct" value for this constant; the choice of using 10000.0 as a constant in the expression wrapped
    in macro "SCALING_FACTOR" is somewhat arbitrary but has been found to work well in practice. 
    Consider trying different values for this constant during experimentation to see if it has an impact on your model's performance.
 */
#define SCALING_FACTOR_CONSTANT 10000.0

/*  Hyperparameters */
/* -----------------*/
/*
    Commonly used values for d_model range from 128 to 1024,
    with 256 being a frequently chosen value because in "Attention is All You Need" paper, the authors used an embedding dimension of 256.
 */
//0x100 = 256
//0x080 = 128
//0x040 = 64
#define DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER 0x040

/*
    The dropout_rate is a hyperparameter typically set between 0.1 and 0.5

    Dropout Rate: Dropout is a regularization technique commonly used in neural networks, including the Transformer model. 
    The dropout_rate represents the probability of randomly "dropping out" or deactivating units (neurons) in a layer during training.
    It helps prevent overfitting and encourages the model to learn more robust and generalized representations by reducing interdependence between units.
    The dropout_rate is a hyperparameter typically set between 0.1 and 0.5, indicating the fraction of units that are dropped out during training.
 */
#define DEFAULT_DROP_OUT_RATE_HYPERPARAMETER 0.1

/*
    TODO, big description about this hyperparameter is needed
 */
#define DEFAULT_EPOCH_HYPERPARAMETER 1

/*
    Number of attention heads
 */
#define DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER 0x08

/*
    Typical values for number of layers in the Transformer model range from 4 to 8 or even higher, depending on the specific application.
    It's common to have an equal number of layers in both the encoder and decoder parts of the Transformer.

    Section 3.1 of "Attention Is All You Need".
    The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. 
    The first is a multi-head self-attention mechanism, and the second is a simple, position wise fully connected 
    feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer 
    normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the 
    function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model,
    as well as the embedding layers, produce outputs of dimension d_model = 512
 */
#define DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER 0x04

/* Hyperparameters end here */
/* ------------------------ */

#include "./hyperparameters.hh"
#include "./attention.hh"
#include "./EncoderFeedForwardNetwork.hh"
#include "./EncoderLayerNormalization.hh"
#include "./EncoderLayerList.hh"
#include "./EncoderLayer.hh"
#include "./encoder.hh"

#include "./DecoderLayerList.hh"
#include "./DecoderLayer.hh"
#include "./decoder.hh"

#include "./model.hh"

#endif