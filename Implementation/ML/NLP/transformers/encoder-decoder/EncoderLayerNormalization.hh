/*   
    ML/NLP/transformers/encoder-decoder/EncoderLayerNormalization.hh
    Q@khaa.pk 
 */

 /*
    Core Algorithm/Formula: output = gamma * (input - mean) / sqrt(variance + epsilon) + beta
  */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH

// Epsilon, arbitrarily small positive quantity(small or close to zero)
#define ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE (1e-6)

template <typename t = double>
class EncoderLayerNormalization
{
    t epsilon /* To avoid division by zero */ ; 
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel;    
    // Trainable Parameters: current implementation, gamma and beta are initialized but not yet properly set up as trainable parameters
    Collective<t> gamma, beta;

    Collective<t> input, input_mean, input_variance, input_normalized;
    Collective<t> input_dispersion, input_variance_stabilized;

    public:
        EncoderLayerNormalization(cc_tokenizer::string_character_traits<char>::size_type d_model, t eps = ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE) throw (ala_exception)
        {
            dimensionsOfTheModel = d_model;
            epsilon = eps;

            try
            {            
                /*
                    Initialize gamma (scaling parameter) to ones and beta (shifting parameter) to zeros
                    Even though they start as ones and zeros, they update during training. 
                    Even though they start as ones amd zeros,
                    gamma (scaling) and beta (shifting) are trainable parameters in layer normalization.
                 */
                gamma = Numcy::ones<t>(DIMENSIONS{d_model, 1, NULL, NULL});
                beta = Numcy::zeros<t>(DIMENSIONS{d_model, 1, NULL, NULL});
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::EncoderLayerNormalization() -> ") + cc_tokenizer::String<char>(e.what()));
            }
        }

        /*
            The function's job is to use @incoming_gradient and compute...
            - gradient for gamma(the scale parameter)
            - gradient for beta(the shift parameter) 
            - gradient for inputs to propagate further
            
            @incoming_gradient, in neural networks, backpropagation works by computing gradients layer by layer,
                                starting from the final loss and moving backward through the network.                                
                                This received gradient is called the incoming gradient because it is coming from the 
                                previous layer (downstream) during backpropagation
                                (it represents the gradient of the loss with respect to the output of the forward pass
                                of the same layer to which this backward propagation function belongs).                                
                                Each layer (upstream, like EncoderLayerNormalization) receives a gradient from the layer
                                ahead of it (downstream, closer to the final loss, like EncoderLayer),
                                which tells it how changes in its outputs (the receiving layer’s activations, i.e., EncoderLayerNormalization)
                                affect the final loss, e.g., EncoderLayerNormalization receives it from EncoderLayer.
         */
        Collective<t> backward(Collective<t>& incoming_gradient) throw (ala_exception)
        {
            /*
                Backpropagation for Layer Normalization:
        
                Given:
                - y = γ * (x - μ) / √(σ² + ε) + β
                - ∂L/∂y-(small y) (incoming_gradient) is the incoming gradient
        
                We need to compute:
                1. ∂L/∂γ-(capital Y) = sum(∂L/∂y * (x - μ)/√(σ² + ε))
                2. ∂L/∂β = sum(∂L/∂y)
                3. ∂L/∂x = (∂L/∂y * γ)/√(σ² + ε) + 
                           (∂L/∂σ² * 2(x - μ)/N) + 
                           (∂L/∂μ * 1/N)
             */

            /*
                - The notation (chnage/gradient in/of L)/(change/gradient in/of y-(small y)) represents the gradient of the loss function L 
                  with respect to y, meaning how much the loss changes when y changes.
                  This is commonly referred to as the incoming gradient because it is passed from the next layer during backpropagation
             */ 
            
            Collective<t> input_gradient;
            
            try 
            {
                Collective<t> temp1, temp2;
                // Retrieve saved values from forward pass
                Collective<t> x_minus_mean = this->input_dispersion;  // (x - mean)
                Collective<t> standard_deviation = Numcy::sqrt<t>(this->input_variance_stabilized);  // squareroot(varience + epsilon)

                /*std::cout<< "incoming_gradient (columns) = " << incoming_gradient.getShape().getNumberOfColumns() << ", " << incoming_gradient.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl; 
                std::cout<< "input_normalized (columns) = " << input_normalized.getShape().getNumberOfColumns() << ", " << input_normalized.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/ 

                // 1. Gradient for gamma (∂L/∂γ)
                Collective<t> gamma_gradient = Numcy::sum<t>(incoming_gradient * input_normalized);

                // 2. Gradient for beta (∂L/∂β)
                Collective<t> beta_gradient = Numcy::sum<t>(incoming_gradient);
            
                // 3. Gradient for input (∂L/∂x)

                cc_tokenizer::string_character_traits<char>::size_type N = this->input.getShape().getN();  // Total elements in normalization dimension
                // Part 1: Direct gradient from normalized output
                Collective<t> dxhat = incoming_gradient * this->gamma;
                // Part 2: Gradient through variance (∂L/∂σ²)
                temp1 = ((dxhat * x_minus_mean) * (t)-0.5);
                temp2 = Numcy::pow<t>(standard_deviation, (t)-3);
                Collective<t> dvar = Numcy::sum<t>(temp1 * temp2);
                // Part 3: Gradient through mean (∂L/∂μ)                
                temp1 = Numcy::mean(x_minus_mean * (t)-2.0);
                temp2 =  dvar * temp1;                
                Collective<t> dmean = Numcy::sum((dxhat * (t)-1.0) / standard_deviation) + temp2;

                // Combine all components
                temp1 = dvar * (t)2.0;
                //temp1 =  temp1 * x_minus_mean; 
                temp1 = (x_minus_mean * temp1);
                temp2 = Numcy::zeros<t>(DIMENSIONS{1, 1, 0, 0});
                temp2[0] = (t)N;
                temp1 = temp1 / temp2;                
                /*input_gradient = (dxhat / std_dev) + 
                        (dvar * 2.0f * x_minus_mean / N) + 
                        (dmean / N);*/
                input_gradient = (dxhat / standard_deviation) + temp1;
                temp1 = dmean / temp2;
                input_gradient = input_gradient + temp1;                      
                
                /*std::cout<< "temp1 (columns) = " << temp1.getShape().getNumberOfColumns() << ", " << temp1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                std::cout<< "x_minus_mean (columns) = " << x_minus_mean.getShape().getNumberOfColumns() << ", " << x_minus_mean.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/

                // Update parameters (in practice, this would be done by an optimizer)
                // this->gamma -= learning_rate * gamma_gradient;
                // this->beta -= learning_rate * beta_gradient;
            }
            catch(ala_exception& e) 
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::backward() -> ") + cc_tokenizer::String<char>(e.what()));
            }
            
            return input_gradient;
        }

        // Forward propagation
        Collective<t> forward(Collective<t>& input) throw (ala_exception)
        {
            /*
                Layer Normalization Formula:
    
                y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta

                where:
                    - x        : Input tensor
                    - mean     : Mean of input
                    - variance : Variance of input
                    - epsilon  : Small constant for numerical stability (e.g., 1e-6)
                    - gamma    : Learnable scaling parameter (initialized as ones)
                    - beta     : Learnable shifting parameter (initialized as zeros)
                    - y        : Normalized output
             */

            Collective<t> input_mean, input_variance, input_normalized;
            Collective<t> input_dispersion, input_variance_stabilized;

            Collective<t> output;

            try
            {
                this->input = input;                
                /*std::cout<< input.getShape().getN() << std::endl;
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < input.getShape().getN(); i++)
                {
                    input[i] = i;
                }*/
                
                /*std::cout<< " -------- -- -- -- - -- - - - -- - - - " << std::endl;
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < input.getShape().getN(); i++)
                {
                    std::cout<< input[i] << ", ";
                }    
                std::cout<< "\n ---- - -- - - - - - - -- - - - - - - -- -  ----- ----  " << std::endl;*/
                 
                // Compute mean of the input tensor
                input_mean = Numcy::mean(input);
                this->input_mean = input_mean;

                // Compute variance of the input tensor
                input_variance = Numcy::variance(input, input_mean);
                this->input_variance = input_variance;

                /*std::cout<< "Input mean = " << input_mean[0] << std::endl;
                std::cout<< "Input variance = " << input_variance[0] << std::endl;*/

                // Normalize input: (input - mean) / sqrt(variance + epsilon)
                input_dispersion = input - input_mean;
                this->input_dispersion = input_dispersion;

                /*std::cout<< " -------- -- -- -- - -- - - - -- - - - " << std::endl;
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < input_dispersion.getShape().getN(); i++)
                {
                    std::cout<< input_dispersion[i] << ", ";
                }    
                std::cout<< "\n ---- - -- - - - - - - -- - - - - - - -- -  ----- ----  " << std::endl;*/

                input_variance_stabilized = input_variance + /*(t)(1e-6)*/ std::max(input_variance[0], epsilon);
                this->input_variance_stabilized = input_variance_stabilized;

                input_normalized = input_dispersion / Numcy::sqrt(input_variance_stabilized);
                this->input_normalized = input_normalized;

                /*std::cout<< "\n ---- - -- - - - - - - -- - - - - - - -- -  ----- ----  " << std::endl;
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < input_normalized.getShape().getN(); i++)
                {
                    std::cout<< input_normalized[i] << ", ";
                }
                std::cout<< "\n ---- - -- - - - - - - -- - - - - - - -- -  ----- ----  " << std::endl;*/    


                /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < gamma.getShape().getN(); i++)
                {
                    gamma[i] = 2.5;
                }

                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < beta.getShape().getN(); i++)
                {
                    beta[i] = -2.5;
                }*/

                // Apply learned scale (gamma) and shift (beta)
                /*std::cout<< "gamma (columns) = " << gamma.getShape().getNumberOfColumns() << ", " << gamma.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl; 
                std::cout<< "input_normalized (columns) = " << input_normalized.getShape().getNumberOfColumns() << ", " << input_normalized.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;
                
                std::cout<< "input (columns) = " << input.getShape().getNumberOfColumns() << ", " << input.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/ 
                
                output = gamma * input_normalized + beta;
            }
            catch(ala_exception& e)
            {                
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }
            
            // ALL OF THIS CODE IS TO DEBUG Numcy::mean() and Numcy::variance() function
           
            /*t* ptr = cc_tokenizer::allocator<t>().allocate(10);

            ptr[0] = 1;
            ptr[1] = 2;
            ptr[2] = 3;
            ptr[3] = 4;
            ptr[4] = 5;
            ptr[5] = 6;
            ptr[6] = 7;
            ptr[7] = 8;
            ptr[8] = 9;
            ptr[9] = 10;

            Collective<t> obj = Collective<t>{ptr, DIMENSIONS{5, 2, NULL, NULL}};*/

            /*Collective<t> ret = Numcy::mean<t>(obj); 
            
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < ret.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
            {
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < ret.getShape().getNumberOfColumns(); j++)
                {
                    std::cout<< ret[i*ret.getShape().getNumberOfColumns() + j] <<", ";
                }

                std::cout<< std::endl;
            }*/

            /*std::cout<< "*******" << std::endl;

            Collective<t> ret;

            ret = Numcy::mean<t>(obj, AXIS_COLUMN); 

            std::cout<< "MEAN = ";
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < ret.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
            {
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < ret.getShape().getNumberOfColumns(); j++)
                {
                    std::cout<< ret[i*ret.getShape().getNumberOfColumns() + j] <<", ";
                }

                std::cout<< std::endl;
            } 

            ret = Numcy::variance(obj, ret, AXIS_COLUMN);

            std::cout<< "Variance = ";
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < ret.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
            {
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < ret.getShape().getNumberOfColumns(); j++)
                {
                    std::cout<< ret[i*ret.getShape().getNumberOfColumns() + j] <<", ";
                }

                std::cout<< std::endl;
            } 
            
            std::cout<< std::endl;*/

            return output;
        }
};

#endif    