/*   
    ML/NLP/transformers/encoder-decoder/EncoderLayerNormalization.hh
    Q@khaa.pk 
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_NORMALIZATION_HH

template <typename t = double>
class EncoderLayerNormalization
{
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel;
    Collective<t> gamma, beta;

    public:
        EncoderLayerNormalization(cc_tokenizer::string_character_traits<char>::size_type d_model) throw (ala_exception)
        {
            dimensionsOfTheModel = d_model;

            try
            {            
                // Initialize gamma (scaling parameter) to ones and beta (shifting parameter) to zeros
                gamma = Numcy::ones<t>(DIMENSIONS{d_model, 1, NULL, NULL});
                beta = Numcy::zeros<t>(DIMENSIONS{d_model, 1, NULL, NULL});
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::EncoderLayerNormalization() -> ") + cc_tokenizer::String<char>(e.what()));
            }
        }

        void forward(Collective<t>& input) throw (ala_exception)
        {
            /*try
            {
                // Compute mean of the input tensor
                Collective<t> mean = Numcy::mean<t>(input, AXIS_COLUMN);

                // Compute variance of the input tensor
                Collective<t> variance = Numcy::variance<t>(input, AXIS_COLUMN);

                // Normalize the input tensor
                input = (input - mean) / Numcy::sqrt<t>(variance + (t)1e-6);

                // Apply scaling and shifting
                input = gamma * input + beta;
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayerNormalization::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }*/

            Collective<t> input_mean, input_variance;
            Collective<t> input_changed, input_variance_changed;


            try
            {                
                // Compute mean of the input tensor
                input_mean = Numcy::mean(input);
                // Compute variance of the input tensor
                input_variance = Numcy::variance(input, input_mean);

                input_changed = input - input_mean;
                input_variance_changed = input_variance + (t)(1e-6);
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
        }
};

#endif    