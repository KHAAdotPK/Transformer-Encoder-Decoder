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
    //DecoderLayerNormalization<t> self_attn_norm; // self_attention_normal
    //DecoderLayerNormalization<t> cross_attn_norm; // cross_attention_normal  
    //DecoderLayerNormalization<t> ffn_norm; // ffn_normal

public:
     
    // Default construtor
    DecoderLayer() : masked_self_attention(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), cross_attention(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER)
                     //ffn(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER),
                     //self_attn_norm(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER),
                     //cross_attn_norm(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER),
                     //ffn_norm(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER)              
    {
        /*
            Layer Normalization Epsilon Call: The following line doesn’t actually do anything
         */
        EncoderLayerNormalization::ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE; // Replaces macro, type-safe
    }      

    // Parametarized constructor 
    DecoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : masked_self_attention(d_model, num_heads), cross_attention(d_model, num_heads)
                  //ffn(d_model, dropout_rate)
                  //self_attn_norm(d_model) 
                  //cross_attn_norm(d_model)
                  //ffn_norm(d_model)
    {
        // Initialization logic if needed
    }

    // Forward pass method and other details would go here...
};

#endif