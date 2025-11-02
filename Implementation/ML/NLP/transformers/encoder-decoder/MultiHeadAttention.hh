/*
    ML/NLP/transformers/encoder-decoder/MultiHeadAttention.hh
    Q@khaa.pk
 */

#include "./header.hh" 

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_MULTI_HEAD_ATTENTION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_MULTI_HEAD_ATTENTION_HH

template <typename t = double>
struct MultiHeadAttention
{
    static Collective<t> worker(Collective<t>& Q, Collective<t>& K, Collective<t>& V, Collective<t>& w_q_slice, Collective<t>& w_k_slice, Collective<t>& w_v_slice) throw (ala_exception)
    {
        /*std::cout<< Q.getShape().getNumberOfColumns() << ", " << Q.getShape().getNumberOfRows() << std::endl;*
        /*std::cout<< w_q_slice.getShape().getNumberOfColumns() << ", " << w_q_slice.getShape().getNumberOfRows() << std::endl;*/
        
        Collective<t> Q_projected, K_projected, V_projected; 
     
        Collective<t> attention_scores, attention_weights, context_vector;
        // Scale factor for attention scores as defined in "Attention Is All You Need"
        // The scale factor 1/sqrt(d_k) prevents softmax saturation in the attention mechanism
        // when key dimension d_k is large, ensuring stable gradients during training
        t scaleFactor = 0;

        try 
        {
            // Transform these three inputs (Q, K, V) through linear projection for multi-head attention
            // 
            // This operation applies a learned (important aspect) linear transformation to the inputs (Q, K, V) 
            // using the attention head-specific weights w_q_slice, w_k_slice, w_v_slice. Each attention 
            // head learns distinct projection patterns that allow the model to focus on 
            // different aspects of the input sequence simultaneously
            //
            // Mathematically: Q_projected = Q · W<sub>i</sub><sup>Q</sup>
            //                 K_projected = K . W<sub>i</sub><sup>K</sup> 
            //                 V_projected = V . W<sub>i</sub><sup>V</sup>
            // Where:
            // - Q, K: input query and key matrices having shape = (sequence_length × d_model)
            // - V: input value matrix it may or may not have the same number of features as the Q, V matrices.  
            // - W<sub>i</sub><sup>Q/K</sup>: head-specific query and key weights having same shape = (d_model × <d_q, d_k>)
            // - W<sub>i</sub><sup>V</sup>: head-specific value weights having same shape = (number of featues × d_v) 
            // - Q_projected: transformed queries for head h (sequence_length × d_k)
            //
            // The projections enables each attention head to learn specialized query, key , value 
            // representations that capture different types of relationships and dependencies
            // within the sequences, a key mechanism behind the multi-head attention's
            // ability to process information in parallel and concurrently from multiple representation subspaces
            Q_projected = Numcy::dot(Q, w_q_slice);
            K_projected = Numcy::dot(K, w_k_slice);
            V_projected = Numcy::dot(V, w_v_slice);

            // We use the projected key dimension (number of features) because attention scores are computed as:
            // attention_scores = (Q_projected · trabspose(K_projected)) / sqrt(d_k)
            // where d_k is the dimension (number of features) of the projected key vectors
            scaleFactor =  1 / std::sqrt(static_cast<t>(K_projected.getShape().getNumberOfColumns()));

            // Compute scaled dot-product attention scores
            // 
            // 1. Calculate raw attention scores: Q_projected · transpose(K_projected)
            //    - Measures compatibility between each query and key pair
            //    - Results in matrix of shape (sequence_length × sequence_length)
            //
            // 2. Apply scale factor: Multiply by 1/sqrt((d_k))
            //    - Prevents softmax saturation when key dimension (number of features) d_k is large
            //    - Ensures stable gradients and effective training
            //    - As defined in "Attention Is All You Need" paper
            //
            // Result: Scaled attention scores ready for softmax normalization
            attention_scores = Numcy::dot(Q_projected, Numcy::transpose(K_projected));
            attention_scores = attention_scores * scaleFactor;

            attention_weights = Numcy::softmax(attention_scores);
            
            /*
             * Compute context vector: weighted sum of values based on attention distribution
             *
             * This operation creates a weighted sum of value vectors based on attention weights.
             * Each output position contains context gathered from all relevant input positions.
             * It represents the "context" that each query position attends to
             */
            context_vector = Numcy::dot(attention_weights, V_projected);
        }
        catch (ala_exception& e)
        {
            throw ala_exception(cc_tokenizer::String<char>("MultiHeadAttention::worker(Collective<&>) -> ") + e.what());
        }

        //
        return context_vector;
    }
};

#endif