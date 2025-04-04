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
                                        ...OR... 
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
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel; // Model dimensionality
    /*
        Dropout Rate and Its Function in Neural Networks
        ---------------------------------------------------
        Dropout is a regularization technique used in neural networks to prevent overfitting by randomly deactivating (setting to zero) a fraction of the neurons during training.

        Dropout Rate (dropOutRate)
        -----------------------------
        - It is a probability value (typically between 0 and 1), representing the fraction of neurons to drop
        - A dropout rate of 0.1 means 10% of neurons are randomly set to zero during training

        Function of Dropout Rate
        ---------------------------
        - Prevents Overfitting → Forces the network to learn redundant representations, making it more generalizable
        - Adds Noise → The network becomes robust to small changes in input
        - Works Only During Training → During inference, dropout is disabled, and neuron outputs are scaled accordingly
     */
    t dropOutRate;

    Collective<t> weights1, weights2; // Weights, W1 and W2
    Collective<t> bias1, bias2; // Biases

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

            // Initialize dimensions/values for weight matrices W1, W2
            weights1 = Numcy::Random::randn<t>(dim1);
            weights2 = Numcy::Random::randn<t>(dim2);

            // Initialize dimensions/values for biases
            bias1 = Numcy::Random::randn<t>(DIMENSIONS{4 * d_model, 1, NULL, NULL});
            bias2 = Numcy::Random::randn<t>(DIMENSIONS{d_model, 1, NULL, NULL});

            /*bias1 = Numcy::zeros<t>(DIMENSIONS{4 * d_model, 1, NULL, NULL});
            bias2 = Numcy::zeros<t>(DIMENSIONS{d_model, 1, NULL, NULL});*/
            
        }

        Collective<t> forward(Collective<t>& input, bool is_training = true)
        {
            //std::cout<< "input columns = " << input.getShape().getNumberOfColumns() << ", " << input.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            //std::cout<< "weights1 columns = " << weights1.getShape().getNumberOfColumns() << ", " << weights1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            
            try
            {                                   
                // First Linear Transformation
                Collective<t> z1 = Numcy::matmul(input, weights1) + bias1;
                
                // Apply ReLU Activation Function. ReLU, short for Rectified Linear Unit
                Collective<t> a1 = Numcy::ReLU(z1); 
                
                // Second Linear Transformation
                Collective<t> z2 = Numcy::matmul(a1, weights2) + bias2; 
                                
                /*
                    Optional Dropout (if implemented in Numcy)
                    Dropout is disabled during inference
                 */
                if (is_training && dropOutRate > 0)
                {
                    z2 = Numcy::dropout(z2, dropOutRate);
                }

                return z2;
            }
            catch (ala_exception& e)
            {                
                throw ala_exception(cc_tokenizer::String<char>("EncoderFeedForwardNetwork::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }                        
        }
};

#endif
