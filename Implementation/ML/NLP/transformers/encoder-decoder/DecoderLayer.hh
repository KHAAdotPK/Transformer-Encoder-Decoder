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

    t dropOutRate;
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
    DecoderLayer() : masked_self_attention(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), cross_attention(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER, DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER)
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
    DecoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : masked_self_attention(d_model, num_heads), cross_attention(d_model, num_heads), dropOutRate(dropout_rate)
                  //ffn(d_model, dropout_rate)
                  //self_attn_norm(d_model) 
                  //cross_attn_norm(d_model)
                  //ffn_norm(d_model)
    {
        // Initialization logic if needed
    }

    // Forward pass method and other details would go here...
    /*
        @decoder_input: This is the main data flowing through the decoder stack. For the first layer, it's the target sequence embeddings; for subsequent layers, it's the output of the previous DecoderLayer. This is used as the initial Query, Key, and Value for the masked self-attention sub-layer
        @encoder_output: This is the final output from the entire encoder stack. It's the crucial context from the source sentence. It serves as the Key and Value for the cross-attention sub-layer, while the Query comes from the output of the decoder's self-attention sub-layer
        @decoder_mask: This is the "look-ahead" mask. It's applied during the masked self-attention step to prevent any token from attending to future tokens in the target sequence, which is essential during training
        @encoder_mask: This is typically the padding mask from the source sentence. It's used in the cross-attention sub-layer to ensure the decoder doesn't pay attention to padding tokens in the encoder's output
     */    
    Collective<t> forward(Collective<t>& decoder_input, Collective<t>& encoder_output, Collective<t>& decoder_mask, Collective<t>& encoder_mask)
    {

        // The responsibility for creating a correctly shaped mask lies outside the Attention class
        /*for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < decoder_mask.getShape().getNumberOfRows(); i++)
        {

        }*/

        //masked_self_attention.forward(decoder_input, decoder_input, decoder_input, decoder_mask/*, encoder_mask*/);


        Collective<t> attention_output = masked_self_attention.forward(decoder_input, decoder_input, decoder_input, decoder_mask);

        decoder_input = decoder_input + attention_output;

        return Collective<t>(NULL, DIMENSIONS{0, 0, NULL, NULL});
    }
};

#endif