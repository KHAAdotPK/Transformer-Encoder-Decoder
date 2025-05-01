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

    Collective<t> cached_a1; // Cached activation from the first layer (for backpropagation)
    Collective<t> cached_input; // Cached input (for backpropagation)
    Collective<t> cached_z1; // Cached input to the first layer (for backpropagation)
    
    static constexpr t DEFAULT_FEED_FORWARD_NETWORK_LAYER_LEARNING_RATE = 0.01;

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

        /*
            @incoming_gradienbt,  // it is dL/dz2, the gradient of the loss with respect to the output of the FFN (z2)
            @learning_rate,       // learning_rate is the step size used in the weight update rule. It controls how much to adjust the weights based on the computed gradients.
         */
        Collective<t> backward( Collective<t>& incoming_gradient, t learning_rate = DEFAULT_FEED_FORWARD_NETWORK_LAYER_LEARNING_RATE) throw (ala_exception)
        {

            /*
                The backward pass of the feedforward network involves computing gradients with respect to the weights and biases of the two linear transformations.
                This is typically done using backpropagation through the network.
            */

            Collective<t> input_gradient; // Gradient with respect to the input tensors (input, weights1, weights2, bias1, bias2)

            try
            {                  
                /*
                    1. dL/da1 = dL/dz2 · weights2^T
                    dL/da1 = gradient of loss with respect to activation(ReLu) of the first layer (a1) and this 
                    dl/dz2 is the incoming gradient from the next layer.                     
                 */
                Collective<t> gradient_a1 = Numcy::matmul(incoming_gradient, Numcy::transpose(weights2));
                
                /* 
                    2. dL/dweights2 = gradient of loss with resppect to weights2,  a1^T = cached_a1^T, dL/dz2  = incoming_gradient 
                    dL/dweights2 = cached_a1^T * incoming_gradient
                */ 
                Collective<t> gradient_weights2 = Numcy::matmul(Numcy::transpose(cached_a1), incoming_gradient);

                /*
                    3. dL/dbias2 = gradient of loss with resppect to bias2 dL/dbias2 = incoming_gradient
                 */
                Collective<t> gradient_bias2 = Numcy::sum(incoming_gradient);

                /*
                    4. dL/dz1 = dL/da1 · ReLU'(z1) Compute dL/dz1 from dL/da1 and ReLU derivative

                    - ReLU'(z1) is the derivative of the ReLU activation function applied to z1.
                    - The derivative of ReLU is 1 for positive values and 0 for negative values.
                    - This means that the gradient will only flow back through the neurons that were activated (i.e., those with positive values in z1).
                    - The gradient of the loss with respect to the input of the first layer (z1) is computed by multiplying the gradient of the loss with respect to the activation (gradient_a1) by the derivative of ReLU and the transpose of weights1.
                    - This step is crucial for backpropagation, as it allows the gradients to flow backward through the network.
                 */                
                Collective<t> gradient_z1 = Numcy::ReLU_prime(cached_z1);
                gradient_z1 = gradient_a1 * gradient_z1; // Element-wise multiplication

                /*
                    5. dL/dweights1 = gradient of loss with resppect to weights1, z1^T = cached_z1^T, dL/dz1  = gradient_z1 
                    dL/dweights1 = cached_z1^T * gradient_z1                    
                 */
                Collective<t> gradient_weights1 = Numcy::matmul(Numcy::transpose(cached_input), gradient_z1);

                /*
                    6. dL/dbias1 = gradient of loss with resppect to bias1 dL/dbias1 = gradient_z1
                 */
                Collective<t> gradient_bias1 = Numcy::sum(gradient_z1);

                /*
                    7. Update weights and biases using gradients and learning rate
                 */
                Collective<t> residual = gradient_weights2 * learning_rate;
                weights2 = weights2 - residual; // Update weights2
                residual = gradient_weights1 * learning_rate;   
                weights1 = weights1 - residual; // Update weights1
                residual = gradient_bias2 * learning_rate;
                bias2 = bias2 - residual; // Update bias2
                residual = gradient_bias1 * learning_rate;
                bias1 = bias1 - residual; // Update bias1
                
                /*
                    8. Compute gradients with respect to the original input (input) by propagating gradients backward through the linear projection layers.                    
                    dL/dinput = dL/dz1 * weights1^T
                    where weights1 is the projection matrix for the first layer, dL/dz1 is the gradient of the loss with respect to the input of the first layer (a.k.a gradient_z1)
                    therefore, dL/dinput is the gradient of the loss with respect to the input of the FFN (a.k.a input_gradient)
                    This step is crucial for backpropagation, as it allows the gradients to flow backward through the network.
                    This is the final step in the backward pass of the feedforward network.
                    The computed gradients can then be used to update the weights and biases of the network using an optimization algorithm (e.g., SGD, Adam).

                    The input_gradient is the gradient of the loss with respect to the input of the FFN, which can be used for further backpropagation in the overall model

                    input_gradient = gradient_z1 * weights1^T
                    where weights1 is the projection matrix for the first layer, gradient_z1 is the gradient of the loss with respect to the input of the first layer (a.k.a cached_z1)
                 */

                    input_gradient = Numcy::matmul(gradient_z1, Numcy::transpose(weights1)); // Gradient with respect to the input tensors (input, weights1, weights2, bias1, bias2)
                    // Note: The cached_input is used here to compute the gradient with respect to the input of the FFN.
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderFeedForwardNetwork::backward() -> ") + cc_tokenizer::String<char>(e.what()));
            }      
                        
            return input_gradient; // Return the gradient with respect to the input of the FFN
        }

        Collective<t> forward(Collective<t>& input, bool is_training = true)
        {
            //std::cout<< "input columns = " << input.getShape().getNumberOfColumns() << ", " << input.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
            //std::cout<< "weights1 columns = " << weights1.getShape().getNumberOfColumns() << ", " << weights1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                        
            try
            {                                   
                this->cached_input = input; // Cache the input for backpropagation

                // First Linear Transformation
                Collective<t> z1 = Numcy::matmul(input, weights1) + bias1;
                this->cached_z1 = z1; // Cache the input for backpropagation
                
                // Apply ReLU Activation Function. ReLU, short for Rectified Linear Unit
                Collective<t> a1 = Numcy::ReLU(z1); 
                this->cached_a1 = a1; // Cache the activation for backpropagation
                
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
