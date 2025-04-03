/*   
    ML/NLP/transformers/encoder-decoder/EncoderLayerNormalization.hh
    Q@khaa.pk 
 */

/*
    EncoderLayerNormalization Class
    ----------------------------------

    Overview:
    ------------
    This class implements Layer Normalization, a key component in Transformer models.
    Unlike Batch Normalization, which normalizes across the batch dimension, 
    Layer Normalization normalizes across feature dimensions, making it independent of batch size.

    Purpose:
    ------------
    - Stabilizes training by normalizing activations within each input sequence.
    - Helps prevent internal covariate shift.
    - Ensures consistent scaling of activations for efficient learning.

    Mathematical Formulation:
    -------------------------
    Given an input tensor X of shape (batch_size, seq_length, hidden_dim), normalization is done per sequence:

        μ = mean(X)                          // Compute mean across hidden_dim
        σ² = variance(X)                     // Compute variance across hidden_dim
        X_norm = (X - μ) / sqrt(σ² + ε)      // Normalize input
        Y = γ * X_norm + β                   // Scale and shift using learnable parameters γ and β

    Components:
    ------------
    - **Forward Pass:** Computes mean, variance, and normalizes the input.
    - **Backward Pass:** Computes gradients for gamma, beta, and input tensor.
    - **Gamma (γ) & Beta (β):** Learnable parameters that scale and shift the normalized output.
    - **Epsilon (ε):** Small constant to avoid division by zero.

    Role in Transformer Model:
    ----------------------------
    - Used in both **Encoder** and **Decoder** layers of a Transformer.
    - Applied after **Multi-Head Self-Attention** and **Feed-Forward Networks** with residual connections.
    - Ensures stable gradients during training.

    Implementation Notes:
    ----------------------------
    - The class stores intermediate results (mean, variance, normalized input) for use in backpropagation.
    - Parameter updates are usually handled in the training loop by an optimizer (e.g., Adam, SGD).
    - The backward() method includes a temporary parameter update section for debugging/stress testing.

    Next Steps:
    ----------------------------
    After implementing this class, the next major component to implement is the **EncoderFeedForwardNetwork**,
    which applies a position-wise feedforward network after layer normalization in the Transformer Encoder.
 */


/*
    Core Algorithm/Formula: output(y) = gamma(Y) * (input - mean) / sqrt(variance + epsilon) + beta(B)
 */
  
#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH

template <typename t = double>
class EncoderLayerNormalization
{    
    t epsilon /* To avoid division by zero */ ; 
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel;    
    // Trainable Parameters: current implementation, gamma and beta are initialized but not yet properly set up as trainable parameters
    Collective<t> gamma, beta;

    Collective<t> input, input_mean, input_variance, input_normalized;
    Collective<t> input_dispersion, input_variance_stabilized;

    /*
        Epsilon, arbitrarily small positive quantity(small or close to zero)
        Using a macro for epsilon is not type-safe and can lead to issues if the type of t changes.
        Using a constexpr variable ensures that the value is type-safe and can be used in template classes.
     */
    static constexpr t ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE = 1e-6;
    static constexpr t ENCODER_LAYER_NORMALIZATION_DEFAULT_LEARNING_RATE = 0.01;

    public:        
        EncoderLayerNormalization(cc_tokenizer::string_character_traits<char>::size_type d_model, t eps = ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE) throw (ala_exception)
        {
            dimensionsOfTheModel = d_model;
            epsilon = eps;
            
            try
            {   
                /*         
                    Initialize gamma (scaling parameter) to ones 
                    Even though it starts as ones, it gets updated during training in EncoderLayerNormalization::backward()
                 */
                gamma = Numcy::ones<t>(DIMENSIONS{d_model, 1, NULL, NULL});
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::EncoderLayerNormalization(): Failed to initialize gamma -> ") + cc_tokenizer::String<char>(e.what()));
            }
        
            try
            {
                /*
                    Initialize beta (shifting parameter) to zeros
                    Even though it starts as zeros, it gets updated during training in EncoderLayerNormalization::backward()
                 */
                beta = Numcy::zeros<t>(DIMENSIONS{d_model, 1, NULL, NULL});
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::EncoderLayerNormalization(): Failed to initialize beta -> ") + cc_tokenizer::String<char>(e.what()));
            }
        }

        /*
            The usual learning rate depends on the optimizer and the model being trained. Here are common values used in practice:
            Typical Learning Rates
            ----------------------
            Optimizer                                       Common Learning Rate
            ---------                                       --------------------
            SGD (Stochastic Gradient Descent)               0.01 - 0.1
            SGD with Momentum                               0.001 - 0.1
            Adam (Adaptive Moment Estimation)               0.0001 - 0.001 (default: 0.001)
            RMSprop                                         0.0001 - 0.01 (default: 0.001)
            Adagrad                                         0.001 - 0.01

            Choosing a Learning Rate
            ------------------------
            - Start small (0.001) for deep networks to avoid instability.
            - Higher (0.01 to 0.1) for simpler networks like logistic regression or shallow MLPs.
            - Use a scheduler (e.g., decay over time) to adapt the learning rate dynamically.

            For Layer Normalization, using SGD, suggestion is  to start with 0.01.
            For using Adam, 0.001 is a solid default
         */

        /*
            The function's job is to use @incoming_gradient and compute...
            - gradient for gamma(the scale parameter) ∂L/∂γ 
            - gradient for beta(the shift parameter) ∂L/∂β
            - gradient for inputs to propagate further ∂L/∂x
            
            @incoming_gradient, in neural networks, backpropagation works by computing gradients layer by layer,
                                starting from the final loss and moving backward through the network.                                
                                This received gradient is called the incoming gradient because it is coming from the 
                                previous layer (downstream) during backpropagation
                                (it represents the gradient of the loss with respect to the output of the forward pass
                                of the same layer to which this backward propagation function belongs).                                
                                Each layer (upstream, like EncoderLayerNormalization) receives a gradient from the layer
                                ahead of it (downstream, closer to the final loss, like EncoderLayer),
                                which tells it how changes in its outputs (the receiving layer’s activations, i.e., EncoderLayerNormalization)
                                affect the final loss, e.g., EncoderLayerNormalization receives it from EncoderLayer
         */        
        Collective<t> backward(Collective<t>& incoming_gradient, t learning_rate = EncoderLayerNormalization::ENCODER_LAYER_NORMALIZATION_DEFAULT_LEARNING_RATE) throw (ala_exception)
        {
            /*
                Backpropagation for Layer Normalization:
        
                Given:
                x = input
                y = output
                γ = gamma
                β = beta
                L = loss
                - y(output) = γ(gamma) * (x - μ) / √(σ² + ε) + β(beta)
                - ∂L/∂y-(small y) (incoming_gradient) is the incoming gradient
        
                We need to compute:
                1. ∂L/∂γ-(capital Y) = sum(∂L/∂y * (x - μ)/√(σ² + ε))
                2. ∂L/∂β = sum(∂L/∂y)
                3. ∂L/∂x = (∂L/∂y * γ)/√(σ² + ε) + 
                           (∂L/∂σ² * 2(x - μ)/N) + 
                           (∂L/∂μ * 1/N)
             */

            /*
                - The notation ∂L/∂y = (chnage/gradient in/of L)/(change/gradient in/of y-(small y)) represents the gradient of the loss function L 
                  with respect to y(output), meaning how much the loss changes when y(output) changes.
                  This is commonly referred to as the incoming gradient because it is passed from the next layer during backpropagation
             */ 
 
            Collective<t> input_gradient;
            
            try 
            {
                Collective<t> temp1, temp2;

                // Reuse values computed in forward(), retrieve those saved values from forward pass
                Collective<t> x_minus_mean = this->input_dispersion;  // (x - mean)
                Collective<t> standard_deviation = Numcy::sqrt(this->input_variance_stabilized);  // sqrt(variance + epsilon)
                cc_tokenizer::string_character_traits<char>::size_type N = this->input.getShape().getN();  // Normalization dimension size

                // 1. Gradient for gamma (∂L/∂γ), compute gradients for gamma
                Collective<t> gamma_gradient = Numcy::sum(incoming_gradient * this->input_normalized);

                // 2. Gradient for beta (∂L/∂β), compute gradients for beta
                Collective<t> beta_gradient = Numcy::sum(incoming_gradient);

                // 3. Gradient for input (∂L/∂x)
                // Part 1: Direct gradient from normalized output
                Collective<t> dxhat = incoming_gradient * this->gamma;  // Direct contribution
                // Part 2: Gradient through variance (∂L/∂σ²)
                temp1 = ((dxhat * x_minus_mean) * (t)-0.5);
                temp2 = Numcy::pow<t>(standard_deviation, (t)-3);
                Collective<t> dvar = Numcy::sum<t>(temp1 * temp2);
                // Part 3: Gradient through mean (∂L/∂μ)   
                temp1 = Numcy::mean(x_minus_mean * (t)-2.0);
                temp2 =  dvar * temp1;                
                Collective<t> dmean = Numcy::sum((dxhat * (t)-1.0) / standard_deviation) + temp2;

                // Compute input gradient (∂L/∂x)
                temp1 = dvar * (t)2.0;
                temp1 = (x_minus_mean * temp1);
                temp2 = Numcy::zeros<t>(DIMENSIONS{1, 1, 0, 0});
                temp2[0] = (t)N;
                temp1 = temp1 / temp2; 
                // Combine all components
                // input_gradient = (dxhat / std_dev) + (dvar * 2.0f * x_minus_mean / N) + (dmean / N)
                input_gradient = (dxhat / standard_deviation) + temp1;
                temp1 = dmean / temp2;
                input_gradient = input_gradient + temp1;
                
                /*
                    Parameter Update (Temporary for Stress Testing)

                    These lines update the learnable parameters gamma and beta using the computed gradients.
                    In a proper deep learning framework, parameter updates should be handled by an optimizer
                    (such as SGD, Adam, or RMSprop) within the training loop, rather than inside the backward() function.

                    Why these lines are here?
                    - We temporarily place them at the end of backward() to stress test the implementation.
                    - This helps verify that gradients are computed correctly and that updates affect the model as expected.

                    Where should they actually be?
                    - In the training loop, where gradients are used to update parameters after each backward pass.
                    - Example (Pseudo-Code):                
                        for each training step:
                            1. Forward pass: output = model.forward(input)
                            2. Compute loss: loss = loss_function(output, target)
                            3. Backward pass: gradients = model.backward(loss_gradient)
                            4. Update parameters using an optimizer (e.g., Adam, SGD)
                
                    In a real scenario, an optimizer would control learning rate scheduling, weight decay, and momentum.
                 */
                // Update parameters (in practice, this would be done by an optimizer)
                // this->gamma -= learning_rate * gamma_gradient;
                // this->beta -= learning_rate * beta_gradient;

                temp1 = gamma_gradient * learning_rate;
                temp2 = beta_gradient * learning_rate;
                this->gamma = this->gamma - temp1;
                this->beta = this->beta - temp2;
            }
            catch(ala_exception& e) 
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::backward() -> ") + cc_tokenizer::String<char>(e.what()));
            }
            
            return input_gradient;
        }
                
        /**
         * @brief Performs the forward pass of Layer Normalization.
         * 
         * Layer Normalization normalizes the input across the feature dimension, 
         * ensuring that each input sample has zero mean and unit variance. 
         * This helps stabilize training and improve generalization.
         * 
         * Normalization Formula:
         *     y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
         * 
         * Where:
         *     - x        : Input tensor
         *     - mean     : Mean of input tensor across the normalization axis
         *     - variance : Variance of input tensor across the normalization axis
         *     - epsilon  : Small constant added for numerical stability (default: 1e-6)
         *     - gamma    : Learnable scaling parameter (initialized as ones)
         *     - beta     : Learnable shifting parameter (initialized as zeros)
         *     - y        : Normalized output tensor
         * 
         * @param input A Collective<t> tensor representing the input data.
         * @return A Collective<t> tensor containing the normalized and scaled output.
         * 
         * @throws ala_exception If any error occurs during computation.
         * 
         * @note The computed mean, variance, and normalized values are stored for use in backpropagation.
         */
        Collective<t> forward(Collective<t>& input) throw (ala_exception)
        {                        
            Collective<t> output;

            try
            {            
                this->input = input; // Store input tensor for backward pass
                                 
                // Compute mean of the input tensor
                this->input_mean = Numcy::mean(input);                
                // Compute variance of the input tensor
                this->input_variance = Numcy::variance(input, this->input_mean);
                // Variance stabilization
                this->input_variance_stabilized = this->input_variance + std::max((this->input_variance)[0], epsilon);
                                
                // We will need to normalize input: input_dispersion / sqrt(variance + epsilon)
                this->input_dispersion = input - this->input_mean;                                                                
                // Precompute standard deviation
                Collective<t> standard_deviation = Numcy::sqrt(this->input_variance_stabilized);
                this->input_normalized = this->input_dispersion / standard_deviation;
                                                
                output = gamma * this->input_normalized + beta;
            }
            catch(ala_exception& e)
            {                
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }
                       
            return output;
        }
};

#endif    