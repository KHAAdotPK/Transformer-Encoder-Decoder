/*
    ML/NLP/transformers/encoder-decoder/MultiHeadAttentionList.hh
    Q@khaa.pk
 */

#include "./header.hh" 

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_MULTI_HEAD_ATTENTION_LIST_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_MULTI_HEAD_ATTENTION_LIST_HH

template <typename t /*= double*/> // Uncomment the default assignment of type and at compile time: warning C4348: 'DecoderLayer': redefinition of default parameter: parameter 1
class Attention;

template <typename t = double>
struct MultiHeadAttentionList
{
    /*
        Transformer Attention layer
     */
    class Attention<t>* ptr; 

    struct MultiHeadAttentionList<t>* next;
    struct MultiHeadAttentionList<t>* previous;
};

#endif
