/*
    ML/NLP/transformers/encoder-decoder/EncoderFeedForwardNetwork.hh
    Q@khaa.pk
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_FEED_FORWARD_NETWORK_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_FEED_FORWARD_NETWORK_HH

/*
    The FeedForwardNetwork class implements a two-layer feedforward neural network with a ReLU activation function, as described in the "Attention is All You Need" paper.
    It is designed to process the output of the multi-head attention mechanism and apply a non-linear transformation to it.

    Each encoder layer in the Transformer model contains a Feed-Forward Network (FFN), which consists of:
    1. A linear transformation that expands the input dimension from d_model to 4 * d_model.
    2. A ReLU activation.
    3. Another linear transformation that reduces the dimension back to d_model.

    Mathematically, the FFN operates as follows: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
                                         .OR. 
    FFN(input) = Numcy::matmul(ReLU(0, (Numcy::matmul(input, weights1) + bias1)), weights2) + bias2

    - W_1 (weights1) has shape (d_model, 4 * d_model), which expands the representation.
    - ReLU is applied to introduce non-linearity.
    - W_2 (weights2) has shape (4 * d_model, d_model), reducing it back to the original shape.

    The ReLU, implementation
    ---------------------------
    - If the input is positive, the output is the input itself
    - If the input is zero or negative, the output is zero  
 */

template <typename t = double>
class EncoderFeedForwardNetwork 
{
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel;
    t dropOutRate;
    Collective<t> weights1, weights2, bias1, bias2;

    public:
                
        EncoderFeedForwardNetwork(cc_tokenizer::string_character_traits<char>::size_type d_model, t dropout_rate) : dimensionsOfTheModel(d_model), dropOutRate(dropout_rate)
        {            
            /*
                The factor 4 in "4 * d_model" comes from the original Transformer architecture as described in the paper "Attention Is All You Need" (Vaswani et al., 2017).
                It refers to the dimensional expansion in the Feed-Forward Network (FFN) inside each Transformer layer

                The "4 * d_model" expansion is not arbitrary, it ensures the Transformer has sufficient capacity to process input features efficiently.
                If you reduce this factor (e.g., 2 instead of 4), the model may lose expressive power.

                Intuition Behind 4× Expansion
                --------------------------------
                1. Higher capacity for learning complex transformations.
                   Expanding the dimension allows the network to capture richer feature representations
                2. Matches the parameter scaling of self-attention layers
                   Self-attention layers have d_model split into multiple attention heads.
                   The FFN compensates by increasing capacity before reducing it again
                3. Empirical Success.
                   The 4× factor was chosen empirically and has been found to work well in large-scale NLP tasks
             */
            /*
                Example (if d_model = 80)
                - weights1: (320 x 80) (expansion)
                - bias1: (320 x 1)
                ReLU applied (ReLU, short for Rectified Linear Unit)
                - weights2: (80 x 320) (reduction)
                - bias2: (80 x 1)
             */
            DIMENSIONS dim1 = DIMENSIONS{4 * d_model, d_model, NULL, NULL};  // Expand
            DIMENSIONS dim2 = DIMENSIONS{d_model, 4 * d_model, NULL, NULL};  // Reduce back

            weights1 = Numcy::Random::randn<t>(dim1);
            weights2 = Numcy::Random::randn<t>(dim2);
            bias1 = Numcy::Random::randn<t>(DIMENSIONS{4 * d_model, 1, NULL, NULL});
            bias2 = Numcy::Random::randn<t>(DIMENSIONS{d_model, 1, NULL, NULL});
        }

        void forward(Collective<t>& input)
        {
            std::cout<< "input columns = " << input.getShape().getNumberOfColumns() << ", " << input.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            std::cout<< "weights1 columns = " << weights1.getShape().getNumberOfColumns() << ", " << weights1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;

            try
            {                                        
                // First Linear Transformation
                input = Numcy::matmul(input, weights1) + bias1;

                // Apply ReLU Activation Function. ReLU, short for Rectified Linear Unit                                
                input = Numcy::ReLU(input);

                // Second Linear Transformation
                input = Numcy::matmul(input, weights2) + bias2;  
                
                // Optional Dropout (if implemented in Numcy)
            }
            catch (ala_exception& e)
            {                
                throw ala_exception(cc_tokenizer::String<char>("EncoderFeedForwardNetwork::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }

            /*input = relu(matmul(input, weights1) + bias1);
            input = matmul(input, weights2) + bias2;*/
            // Apply dropout
        }
};

#endif
