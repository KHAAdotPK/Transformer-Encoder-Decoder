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
        
        //Collective<t> forward(Collective<t>& decoder_input, Collective<t>& encoder_output, Collective<t>& decoder_mask, Collective<t>& encoder_mask) const
        Collective<t> forward(void) const
        {
            // Implement the forward pass logic here
            // This is a placeholder implementation
            return Collective<t>(NULL, DIMENSIONS{0, 0, NULL, NULL});
        }        
};

#endif
// End of file