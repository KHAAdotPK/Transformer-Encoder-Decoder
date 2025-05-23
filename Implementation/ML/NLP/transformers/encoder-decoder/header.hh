/*
    ML/NLP/transformers/encoder-decoder/header.hh
    Q@khaa.pk
 */

#include "./../../../../lib/Numcy/header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_HEADER_HH

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