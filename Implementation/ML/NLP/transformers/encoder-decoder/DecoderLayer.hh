/*
    ML/NLP/transformers/encoder-decoder/DecoderLayer.hh 
    Q@khaa.pk
 */

#include "./header.hh"

template <typename t = double>
class DecoderLayer
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
};