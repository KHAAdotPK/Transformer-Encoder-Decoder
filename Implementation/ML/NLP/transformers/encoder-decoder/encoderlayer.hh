/*
    lib/NLP/transformers/encoder-decoder/encoderlayer.hh
    Q@khaa.pk
 */

/*
    Transformer, has "encoder layers" (instead of one encoder we have few). Each "encoder layer" (or just an encoder) is very similar to other or all encoders have same architecture. 
    Each encoder or "encoder layer" consists of two layers: Self-attention and a feed Forward Neural Network. 

    As is the case in NLP applications in general, we begin by turning each input word into a vector.
    After embedding the words in input sequence, each of them flows through each of the two layers of the encoder.   
    The embedding only happens in the bottom most encoder, but in other encoders, it would be the output of the encoder that is directly below.
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_HH

/*
    The encoder consists of many encoder layers.
 */
template <typename t = double>
/*typedef*/ class EncoderLayer
{   
    //MULTIHEADATTENTION attention;  
    Attention<float> attention;

    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfAttentionHeads;
    float dropOutRate;

    public:
        EncoderLayer() : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER)
        {
            //attention = MULTIHEADATTENTION();  
            attention = Attention<float>();
        }
        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads. 
            @dropout_rate, Dropout rate for regularization. The dropout_rate in the Transformer model is a regularization technique to prevent overfitting.
         */
        EncoderLayer(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads, float dropout_rate) : dropOutRate(dropout_rate)
        {            
            //attention = MULTIHEADATTENTION(d_model, num_heads);   
            attention = Attention<float>(d_model, num_heads);
        }

        /*template <typename t = float>*/
        void forward(Collective<t>& ei)
        {
            /*
                The forward method of the ENCODERLAYER class call the forward method of the MULTIHEADATTENTION class with the same argument ei(encoder input) passed three times for query, key, and value.
                This might seem redundant at first glance, but there's a specific reason for it.

                While it may seem like a repetition, using the same argument for query, key, and value in the MultiHeadAttention call 
                enables self-attention, a fundamental mechanism for Transformers to understand the relationships within a sequence.

                Read more about in the comment section of MULTIHEADATTENTION::forward()
             */            
            attention.forward(ei, ei, ei);
        }

        ~EncoderLayer()
        {            
        }

} /*ENCODERLAYER*/;

//typedef ENCODERLAYER* ENCODERLAYER_PTR;

/*
    Transformer, has "encoder layers" (instead of one encoder we have few). 
 */

template <typename t = double>
/*typedef*/ struct EncoderLayerList
{
    /*
        Transformer encoder layer
     */
    class EncoderLayer<t>* ptr; 

    struct EncoderLayerList<t>* next;
    struct EncoderLayerList<t>* previous;
} /*ENCODERLAYERLIST*/;

//typedef struct EncoderLayerList<float> ENCODERLAYERLIST;
//typedef /*ENCODERLAYERLIST**/  EncoderLayerList<float>* ENCODERLAYERLIST_PTR;

#endif