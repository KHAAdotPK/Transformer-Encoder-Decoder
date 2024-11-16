/*
    lib/NLP/transformers/encoder-decoder/encoder.hh
    Q@khaa.pk
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_HH

template <typename t = double>
class Encoder 
{
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel, numberOfLayers, numberOfAttentionHeads;

    /*
        Instead of one encoder we have few. 
     */
    /*ENCODERLAYERLIST_PTR*/ EncoderLayerList<t>* encoderLayerListHead;
    float dropOutRate;

    public:
        Encoder(void) : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfLayers(DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER), encoderLayerListHead(NULL)
        {
            /*ENCODERLAYERLIST_PTR*/ EncoderLayerList<t>* current = NULL;
                                    
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfLayers; i++)
            {                                
                if (current == NULL)
                {                    
                    current = reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    encoderLayerListHead = current;
                    current->previous = NULL;                    
                }
                else
                {                 
                    current->next = reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    current->next->previous = current;
                    current = current->next;
                }
                
                current->next = NULL;    
                current->ptr = new EncoderLayer<t>(dimensionsOfTheModel, numberOfAttentionHeads, dropOutRate);
            }                       
        }
        
        /*
            All arguments are "hyperparameters". Learn more about them in DOCUMENTS/hyperparameters.md
            @d_model, the dimension of the embedding space. Like Weights of NN determines the capacity and expressive power of the model.
            @num_layers, number of encoders. In the original paper about Transformers "Attention is all we need", six encoders were used.
            @num_heads, the number of atention heads. 
            @dropout_rate, it represents the probability of randomly "dropping out" or deactivating units (neurons) in a layer/encoder. Typically set between 0.1 and 0.5.
         */
        Encoder(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_layers, cc_tokenizer::string_character_traits<char>::size_type num_heads, float dropout_rate) : dimensionsOfTheModel(d_model), numberOfLayers(num_layers), numberOfAttentionHeads(num_heads), dropOutRate(dropout_rate), encoderLayerListHead(NULL)
        {
            /*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>* current = NULL;  

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfLayers; i++)
            {                                
                if (current == NULL)
                {                    
                    current = reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    encoderLayerListHead = current;
                    current->previous = NULL;                    
                }
                else
                {                 
                    current->next = reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    current->next->previous = current;
                    current = current->next;
                }
                
                current->next = NULL;    
                current->ptr = new EncoderLayer<t>(dimensionsOfTheModel, numberOfAttentionHeads, dropOutRate);
            }                       
        }

        /*template <typename t = float>*/
        void forward(Collective<t>& ei)
        {
            /*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>* current = encoderLayerListHead;

            while (current != NULL)
            {
                current->ptr->forward(ei);

                current = current->next;                    
            }           
        }

        ~Encoder(void)        
        {
            if (encoderLayerListHead != NULL)
            {
                /*ENCODERLAYERLIST_PTR*/ EncoderLayerList<t>* current = encoderLayerListHead;
                /*ENCODERLAYERLIST_PTR*/ EncoderLayerList<t>* next;
                                
                while (current != NULL)
                {
                    next = current->next;

                    /*
                        The `delete[]` operator is used for deleting arrays allocated with `new[]`.
                        *(current->ptr) is a single object (not an array), you should use `delete` instead of `delete[]`.
                     */
                    delete current->ptr;
                    current->ptr = NULL;

                    cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current));
                    current = next;                    
                }                                
            }                           
        }
};
#endif