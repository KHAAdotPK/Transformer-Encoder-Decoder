/*
    ML/NLP/transformers/encoder-decoder/EncoderLayer.hh
    Q@khaa.pk
 */

/*
    Transformer, has "encoder layers" (instead of one encoder we have few). Each "encoder layer" (or just an encoder) is very similar to other or all encoders have same architecture. 
    Each encoder or "encoder layer" consists of two layers: Self-attention and a feed Forward Neural Network. 

    As is the case in NLP applications in general, we begin by turning each input word into a vector.
    After embedding the words in input sequence, each of them flows through each of the two layers of the encoder.   
    The embedding only happens in the bottom most encoder, but in other encoders, it would be the output of the encoder that is directly below.

    Each encoder layer in a transformer consists of:
    - Multi-head self-attention(attention.hh).
    - A feedforward network (FFN, EncoderFeedForwardNetwork.hh).
    - Layer normalization and dropout rate.    
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH

template <typename t>
class EncoderLayer;

template <typename t = double>
struct EncoderLayerList
{
    /*
        Transformer encoder layer
     */
    class EncoderLayer<t>* ptr; 

    struct EncoderLayerList<t>* next;
    struct EncoderLayerList<t>* previous;
};

/*
    The encoder consists of many encoder layers.
 */
template <typename t = double>
class EncoderLayer
{       
    Attention<t> attention;
    EncoderFeedForwardNetwork<t> ffn; // Forward Feed Network
    EncoderLayerNormalization<t> norm1, norm2; // Layer Normalization
    
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfAttentionHeads;
    t dropOutRate;

    public:
        //EncoderLayer() : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER), attention(), ffn(dimensionsOfTheModel, dropOutRate), norm1(dimensionsOfTheModel), norm2(dimensionsOfTheModel)
        EncoderLayer() :  attention(), ffn(dimensionsOfTheModel, DEFAULT_DROP_OUT_RATE_HYPERPARAMETER), norm1(dimensionsOfTheModel), norm2(dimensionsOfTheModel), dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER) 
        {  
            EncoderLayerNormalization::ENCODER_LAYER_NORMALIZATION_EPSILON_VALUE; // Replaces macro, type-safe
        }
        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads. 
            @dropout_rate, Dropout rate for regularization. The dropout_rate in the Transformer model is a regularization technique to prevent overfitting.
         */
        //EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : dimensionsOfTheModel(d_model), dropOutRate(dropout_rate), attention(d_model, num_heads), ffn(d_model, dropout_rate), norm1(d_model), norm2(d_model)
        EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : attention(d_model, num_heads), ffn(d_model, dropout_rate), norm1(d_model), norm2(d_model), dimensionsOfTheModel(d_model), numberOfAttentionHeads(num_heads), dropOutRate(dropout_rate)
        {                        
        }
        
        Collective<t> forward(Collective<t>& ei, Collective<t>& mask)
        {
            /*
                The output of MULTIHEADATTENTION::forward() is typically a transformed representation of the input sequence, where each token's representation has been updated based on attention over all tokens in the sequence. In a Transformer encoder, this output is usually processed further in the following steps:

                What to Do with the Output of MULTIHEADATTENTION::forward()?
                1. Pass it to the Feed-Forward Network (FFN)
                   - The multi-head self-attention output is fed into a position-wise feed-forward network (FFN).
                   - In this implementation it corespondes to passing it to EncoderFeedForwardNetwork::forward():
                     EncoderFeedForwardNetwork::forward(output);

                2. Apply Layer Normalization
                   - After adding a residual connection (if applicable), the output is normalized.
                   - In this implementation it correspondes to EncoderLayerNormalization::forward():
                     EncoderLayerNormlization::forward(output);
                
                3. Add Residual Connection (Optional, but Important in Transformer)
                   - If this is a standard Transformer encoder implementation, then add a residual connection before applying layer normalization.
                   output = input + output;  // Residual connection
                   EncoderLayerNormalization::forward(output);
                
                4. Use it as Input for the Next Encoder Layer (If Using Stacked Encoders)
                   - If Transformer encoder implementation has multiple layers, the output of this layer becomes the input to the next encoder layer.
                
                5. Pass to the Decoder (If Implementing an Encoder-Decoder Model)
                   - If this is part of an encoder-decoder Transformer (e.g., for machine translation), the final encoder output will be used as input to the decoder.   
             */
            Collective<t> output;

            try 
            {
               /*
                    The forward method of the ENCODERLAYER class call the forward method of the MULTIHEADATTENTION class with the same argument ei(encoder input) passed three times for query, key, and value.
                    This might seem redundant at first glance, but there's a specific reason for it.

                    While it may seem like a repetition, using the same argument for query, key, and value in the MultiHeadAttention call 
                    enables self-attention, a fundamental mechanism for Transformers to understand the relationships within a sequence.

                    Read more about in the comment section of MULTIHEADATTENTION::forward()
                */                           
                output = attention.forward(ei, ei, ei, mask); // Residual connection around attention
                output = ei + output; // Residual connection around attention

                output = norm1.forward(ei); // Layer normalization                
                /*
                    The encoder layer should only call backward() when running in training mode and,
                    during training, gradients will flow in the reverse order
                 */                
                norm1.backward(output);

                // ************************************************************************************ //
                //  The following commented statement has been replaced with the two statements below.  //
                //  This change was made due to the reason explained in the comment block of the        //
                //  operator+ overloaded method in the Numcy::Collective parameter class.               //
                // ************************************************************************************ //
                /*
                    output = output + ffn.forward(output); // Residual connection around FFN                    
                 */
                // The output of the feed-forward network is added to the input of the feed-forward network (output) to create a residual connection.
                // This residual connection helps the model learn better by allowing gradients to flow more easily during backpropagation
                // and helps mitigate the vanishing gradient problem.         
                Collective<t> residual = ffn.forward(output); // Feed-forward network output
                output = output + residual; // Residual connection around FFN
                
                // The encoder layer should only call backward() when running in training mode
                // The output of the feed-forward network is then passed through layer normalization to stabilize the training process.                
                output = norm2.forward(output); // Layer normalization
                /*
                    The encoder layer should only call backward() when running in training mode and,
                    during training, gradients will flow in the reverse order
                 */ 
                norm2.backward(output);
            }
            catch(ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("EncoderLayer::forward() -> ") + cc_tokenizer::String<char>(e.what()));
            }

            return output;
        }

        ~EncoderLayer()
        {            
        }

};

#endif