/*
    ML/NLP/transformers/encoder-decoder/EncoderLayerList.hh 
    Q@khaa.pk
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_LAYER_LIST_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_DECODER_LAYER_LIST_HH

template <typename t = double>
class DecoderLayer; // Forward declaration of DecoderLayer

template <typename t = double>
struct DecoderLayerList
{
    DecoderLayer<t>* ptr; 

    struct DecoderLayerList<t>* next;
    struct DecoderLayerList<t>* previous;
};
// This is a doubly linked list of decoder layers

#endif
