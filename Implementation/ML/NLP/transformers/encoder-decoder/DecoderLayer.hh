/*
    ML/NLP/transformers/encoder-decoder/DecoderLayer.hh 
    Q@khaa.pk
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_LAYER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_LAYER_HH

template <typename t = double>
class DecoderLayer /*: public Layer<t>*/
{
private:
    // 1. Masked Self-Attention (looks at previous tokens only)
    Attention<t> masked_self_attention;
    
    // 2. Cross-Attention (attends to encoder output)
    Attention<t> cross_attention; 
    
    // 3. Feed-Forward Network (same as encoder)
    //DecoderFeedForwardNetwork<t> ffn;
    
    // 4. Three Layer Normalizations (one for each sublayer)
    //DecoderLayerNormalization<t> self_attn_norm;
    //DecoderLayerNormalization<t> cross_attn_norm; 
    //DecoderLayerNormalization<t> ffn_norm;

public:
    // Constructor and forward method (we'll implement these next)
    DecoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate)
        : masked_self_attention(d_model, num_heads, dropout_rate), cross_attention(d_model, num_heads, dropout_rate)
        //, ffn(d_model, dropout_rate), self_attn_norm(d_model), cross_attn_norm(d_model), ffn_norm(d_model)
    {
        // Initialization logic if needed
    }

};

#endif