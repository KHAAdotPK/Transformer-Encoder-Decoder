/*
    ML/NLP/transformers/encoder-decoder/EncoderFeedForwardNetwork.hh
    Q@khaa.pk
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_FEED_FORWARD_NETWORK_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_FEED_FORWARD_NETWORK_HH

/*   
    EncoderFeedForwardNetwork Class
    --------------------------------
    This class implements a FeedForward Neural Network (FFN) for the Transformer encoder-decoder model, as described in the paper "Attention is All You Need" (Vaswani et al., 2017). It processes the output of the multi-head attention mechanism by applying a non-linear transformation.

    The FFN architecture consists of:
    1. A linear transformation that expands the input dimension from `d_model` to `4 * d_model`.
    2. A ReLU activation function to introduce non-linearity.
    3. Another linear transformation that reduces the dimension back to `d_model`.

    FFN Functionality:
    ------------------
    Mathematically, the FFN operates as:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        OR
        FFN(input) = Numcy::matmul(ReLU(Numcy::matmul(input, weights1) + bias1), weights2) + bias2

    Where:
    - `weights1` has shape `(d_model, 4 * d_model)` for expansion.
    - A ReLU activation is applied to introduce non-linearity.
    - `weights2` has shape `(4 * d_model, d_model)` to reduce the dimension back.

    Dropout Regularization:
    -------------------------
    Dropout is used as a regularization technique to prevent overfitting. During training, a fraction of neurons is randomly deactivated, based on the `dropOutRate`. During inference, dropout is disabled, and outputs are scaled accordingly.

    Constructor:
    ------------
    The constructor initializes the weights and biases for the two linear transformations, using a random initialization (Gaussian distribution). The model dimensionality `d_model` and dropout rate `dropOutRate` are provided as inputs.

    Forward Pass:
    -------------
    The `forward()` method implements the forward pass of the FFN, applying the two linear transformations, ReLU activation, and optional dropout (during training).
    
    Author: Q@khaa.pk
*/

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
            /*weights1 = Numcy::Random::randn<t>(dim1);
            weights2 = Numcy::Random::randn<t>(dim2);*/

            weights1 = Numcy::Random::randn_xavier<t>(DIMENSIONS{4 * d_model, d_model, NULL, NULL});
            weights2 = Numcy::Random::randn_xavier<t>(DIMENSIONS{d_model, 4 * d_model, NULL, NULL});

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
                else
                {
                    z2 = z2 * (1 - dropOutRate); // Scale the output during inference (to account for deactivated neurons during training)
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

/*
    Optional But Consider for Later(not urgent for now)
    -----------------------------------------------------
    Once this model is wired up and is up and running, consider the following things which could be added to the implementation...
    - Weight initialization strategy (Xavier/He for FFN layers, if you go deeper)
    - Parameter loading/saving methods for inference
    - Unit test for this class with a fixed input
    - Tracking number of parameters (in case you want to print total model size later)

    Abstraction and Reusability
    ------------------------------
    While this class and other classes making up encoder and decoder are complete and clean,
    you might consider pushing toward a more abstract Layer base class with forward() as a
    virtual function. That’d help if you build more layers(like EncoderFeedForwardNetwork is one such layer) later

    template <typename t = double>
    class Layer {
        public:
            virtual Collective<t> forward(Collective<t>& input, bool is_training = true) = 0;
            virtual ~Layer() = default;
    };

    template <typename t = double>
    class EncoderFeedForwardNetwork : public Layer<t> {
        // Your FFN logic
        Collective<t> forward(Collective<t>& input, bool is_training = true) override { ... }
    };
 */
