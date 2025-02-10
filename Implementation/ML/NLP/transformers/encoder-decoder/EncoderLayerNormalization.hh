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
        EncoderLayerNormalization(cc_tokenizer::string_character_traits<char>::size_type d_model) : dimensionsOfTheModel(d_model)
        {
            /*gamma = Numcy::Ones<t>({d_model, 1});
            beta = Numcy::Zeros<t>({d_model, 1});*/
        }

        void forward(Collective<t>& input)
        {
            /*t mean = input.mean();
            t variance = input.var();
            input = gamma * (input - mean) / sqrt(variance + 1e-6) + beta;*/
        }
};

#endif    