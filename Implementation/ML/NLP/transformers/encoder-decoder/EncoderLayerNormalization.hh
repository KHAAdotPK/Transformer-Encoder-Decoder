/*   
    ML/NLP/transformers/encoder-decoder/EncoderLayerNormalization.hh
    Q@khaa.pk 
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH

// Epsilon, arbitrarily small positive quantity(small or close to zero)
#define ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE (1e-6)

template <typename t = double>
class EncoderLayerNormalization
{
    t epsilon;
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel;
    Collective<t> gamma, beta;

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
                // Compute variance of the input tensor
                input_variance = Numcy::variance(input, input_mean);

                /*std::cout<< "Input mean = " << input_mean[0] << std::endl;
                std::cout<< "Input variance = " << input_variance[0] << std::endl;*/

                // Normalize input: (input - mean) / sqrt(variance + epsilon)
                input_dispersion = input - input_mean;

                /*std::cout<< " -------- -- -- -- - -- - - - -- - - - " << std::endl;
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < input_dispersion.getShape().getN(); i++)
                {
                    std::cout<< input_dispersion[i] << ", ";
                }    
                std::cout<< "\n ---- - -- - - - - - - -- - - - - - - -- -  ----- ----  " << std::endl;*/

                input_variance_stabilized = input_variance + /*(t)(1e-6)*/ std::max(input_variance[0], epsilon);                
                input_normalized = input_dispersion / Numcy::sqrt(input_variance_stabilized);

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