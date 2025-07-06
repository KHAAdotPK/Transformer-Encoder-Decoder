/*
    ML/NLP/transformers/encoder-decoder/DecoderLayerList.hh 
    Q@khaa.pk
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_LAYER_LIST_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_LAYER_LIST_HH

// Forward declaration of DecoderLayer
template <typename t /*= double*/> // Uncomment the default assignment of type and at compile time: warning C4348: 'DecoderLayer': redefinition of default parameter: parameter 1
class DecoderLayer;

template <typename t = double>
struct DecoderLayerList
{
    DecoderLayer<t>* ptr; 

    struct DecoderLayerList<t>* next;
    struct DecoderLayerList<t>* previous;
};
// This is a doubly linked list of decoder layers

#endif
