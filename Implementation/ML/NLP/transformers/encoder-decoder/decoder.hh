/*
    ML/NLP/transformers/encoder-decoder/decoder.hh
    Q@khaa.pk
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_HH

template <typename t = double>
class Decoder 
{
    private:
        cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfLayers, numberOfAttentionHeads;
    
        // Instead of one decoder we have multiple decoder layers
        DecoderLayerList<t>* decoderLayerListHead;
    
        t dropOutRate;

    public:

        // Constructors (similar to your Encoder pattern)
        // Forward method (but decoder needs TWO inputs: decoder_input AND encoder_output)
        // Destructor

        Decoder(void) : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfLayers(DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), decoderLayerListHead(NULL), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER)
        {

        }

        Decoder(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_layers, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : dimensionsOfTheModel(d_model), numberOfLayers(num_layers), numberOfAttentionHeads(num_heads), decoderLayerListHead(NULL), dropOutRate(dropout_rate)
        {
        }
        
        Collective<t> forward(Collective<t>& decoder_input, Collective<t>& encoder_output, Collective<t>& decoder_mask, Collective<t>& encoder_mask) const        
        {
            // Implement the forward pass logic here
            // This is a placeholder implementation
            return Collective<t>(NULL, DIMENSIONS{0, 0, NULL, NULL});
        }        
};

#endif
// End of file