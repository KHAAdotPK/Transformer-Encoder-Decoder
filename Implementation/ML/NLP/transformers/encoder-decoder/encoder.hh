/*
    ML/NLP/transformers/encoder-decoder/encoder.hh
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
    EncoderLayerList<t>* encoderLayerListHead;

    t dropOutRate;

    public:

        // Default constructor
        Encoder(void) : dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfLayers(DEFAULT_NUMBER_OF_LAYERS_FOR_ENCODER_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER), encoderLayerListHead(NULL), dropOutRate(DEFAULT_DROP_OUT_RATE_HYPERPARAMETER)
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
                /*
                    In a Transformer-based encoder (such as in BERT or GPT-like models), each encoder layer consists of multiple sublayers, typically:
                    1. Self-Attention Layer
                    2. Feedforward Layer
                    3. Layer Normalization (before or after these)
                 */    
                current->ptr = new EncoderLayer<t>(dimensionsOfTheModel, numberOfAttentionHeads, dropOutRate);
            }                       
        }
        
        // Parameterized Constructor
        /*
            All arguments are "hyperparameters". Learn more about them in DOCUMENTS/hyperparameters.md
            @d_model, the dimension of the embedding space. Like Weights of NN determines the capacity and expressive power of the model.
            @num_layers, number of encoders. In the original paper about Transformers "Attention is all we need", six encoders were used.
            @num_heads, the number of atention heads. 
            @dropout_rate, it represents the probability of randomly "dropping out" or deactivating units (neurons) in a layer/encoder. Typically set between 0.1 and 0.5.
         */
        Encoder(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_layers, cc_tokenizer::string_character_traits<char>::size_type num_heads, t dropout_rate) : dimensionsOfTheModel(d_model), numberOfLayers(num_layers), numberOfAttentionHeads(num_heads), encoderLayerListHead(NULL), dropOutRate(dropout_rate)
        {
            if (dropout_rate < 0.0 || dropout_rate > 1.0)
            {
                dropout_rate = DEFAULT_DROP_OUT_RATE_HYPERPARAMETER;
                
                std::cerr << "Encoder::Encoder() Warning: Invalid dropout_rate provided (" << dropout_rate << "). " << "The dropout_rate must be between 0.0 and 1.0. " << "Using default value: " << DEFAULT_DROP_OUT_RATE_HYPERPARAMETER << "." << std::endl;
            }

            /*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>* current = NULL; 
                        
            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < numberOfLayers; i++)
            {                                
                if (current == NULL)
                {                    
                    current = /*new EncoderLayerList<t>();*/ reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    encoderLayerListHead = current;
                    current->previous = NULL;                    
                }
                else
                {                 
                    current->next = /*new EncoderLayerList<t>();*/ reinterpret_cast</*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>*>(cc_tokenizer::allocator<char>().allocate(sizeof(/*ENCODERLAYERLIST*/EncoderLayerList<t>)));
                    current->next->previous = current;
                    current = current->next;
                }
                
                current->next = NULL; 
                /*
                    In a Transformer-based encoder (such as in BERT or GPT-like models), each encoder layer consists of multiple sublayers, typically:
                    1. Self-Attention Layer
                    2. Feedforward Layer
                    3. Layer Normalization (before or after these)
                 */     
                current->ptr = new EncoderLayer<t>(dimensionsOfTheModel, numberOfAttentionHeads, dropOutRate);                
            }                       
        }

        /*
            Forward Pass
            ---------------
            When training the model, the forward pass computes outputs, 
            this forward pass propagates the input(encoder input) through all encoder layers
            @ei, encoder input
            @return, the output of the last encoder layer
         */
        /*template <typename t = float>*/
        Collective<t> forward(Collective<t>& ei, Collective<t>& mask, Collective<t>& attentionMaskInputSequence) const
        {
            /*ENCODERLAYERLIST_PTR*/EncoderLayerList<t>* current = encoderLayerListHead;

            Collective<t> output = ei;  
            
            //AD_HOC_DEBUG_MACRO(output);
            
            while (current != NULL)
            {
                /*
                    How the Forward Pass Works (EncoderLayer::forward()):
                    1. The input goes through Self-Attention → produces some transformed output
                    2. The output is then passed through Layer Normalization (EncoderLayerNormalization)
                    3. The final output of the encoder layer is then sent to the next encoder layer (or decoder)
                 */
                output = current->ptr->forward(output, mask, attentionMaskInputSequence);

                current = current->next;                    
            }

            /*
                Add explicit zeroing of padded rows after all encoder layers,
                right before returning the final encoder output.
                This is a foolproof way to guarantee padding doesn't leak through
             */
            /**
             * CRITICAL PADDING HANDLING
             * =========================
             * 
             * WHY THIS IS NECESSARY:
             * ----------------------
             * 1. Theoretical Ideal:
             *    - Attention masking (-inf) should prevent padded tokens from affecting outputs
             *    - LayerNorm and residual connections can still propagate small non-zero values
             * 
             * 2. Practical Reality:
             *    - Floating-point arithmetic introduces numerical noise
             *    - Residual connections may carry forward padding zeros imperfectly
             *    - LayerNorm re-scales all values, including padding positions
             * 
             * 3. Safety Guarantee:
             *    - Downstream layers (especially decoder cross-attention) assume padding is truly inert
             *    - Zeroing ensures mathematical correctness regardless of architecture changes
             * 
             * HOW TO IMPLEMENT:
             * ----------------
             * 1. Add this AFTER all encoder layers but BEFORE returning output:
             * 
             *    for (size_t i = 0; i < mask.size(); ++i) {
             *        if (mask[i] == 0) {
             *            std::fill_n(&output[i * d_model], d_model, 0.0);
             *        }
             *    }
             * 
             * 2. Requirements:
             *    - Must operate on final encoder output
             *    - Must use original padding mask (1=real, 0=padding)
             *    - Should be the LAST operation before return
             * 
             * DEBUGGING CONFIRMATION:
             * ----------------------
             * Verify padded rows show EXACT zeros:
             * 
             *    [ [ 0.12, -0.45, ...],  // Real token
             *      [ 0.00,  0.00, ...],  // Padded
             *      [ 0.00,  0.00, ...] ] // Padded
             * 
             * PERFORMANCE NOTE:
             * ----------------
             * - Adds O(N·d_model) time complexity (trivial vs attention's O(N²·d_model))
             * - No memory overhead
             * - Critical for correctness worth the minor cost
             */
            /*
                Forward Pass
               ---------------
               When training the model, the forward pass computes outputs, 
               this forward pass propagates the input(encoder input) through all encoder layers
               @ei, encoder input
               @return, the output of the last encoder layer
    
               NOTE: PADDING CLEANUP HANDLED EXTERNALLY
               =======================================
               This method returns raw encoder output which may contain non-zero values 
               in padded positions due to:
               - FFN bias terms, layer normalization, residual connections
               - Floating-point precision artifacts
    
               IMPORTANT: The training loop applies ADHOC_IMPLEMENTATION_OF_MASK_QUERY() 
               after this method to zero out padded rows before passing to decoder.
               This external cleanup allows easy debugging by commenting out the masking 
               to inspect raw encoder outputs.
    
               See training loop for padding enforcement implementation.
             */            
            return output;
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

                    cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(current), sizeof(EncoderLayerList<t>));
                    current = next;                    
                }                                
            }                           
        }
};
#endif