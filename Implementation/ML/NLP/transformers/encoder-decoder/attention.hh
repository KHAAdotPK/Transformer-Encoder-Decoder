/*
    lib/NLP/transformers/encoder-decoder/attention.hh
    Q@khaa.pk
 */

#include "./header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ATTENTION_HH

/*
    Multi head attention.
 */
template <typename t = double>
/*typedef*/ class Attention // is all you need.
{
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfAttentionHead, dimensionsOfTheModel, numberOfAttentionHeads;
    Collective<t> queryWeights, keyWeights, valueWeights, outputWeights;

    public:
        Attention(void) : dimensionsOfAttentionHead(floor((double)(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER/DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER))), dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER), numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER)
        {   
            /*DIMENSIONS dim3 = {10, 3, NULL, NULL};
            DIMENSIONS dim2 = {0, 10, &dim3, NULL};
            dim3.prev = &dim2;
            DIMENSIONS dim1 = {0, 78, &dim2, NULL};
            dim2.prev = &dim1;
            DIMENSIONS dim = {0, 9, &dim1, NULL};
            dim1.prev = &dim;               
            Numcy::Random::randn(dim);*/
           //Numcy::Random::randn(DIMENSIONS{0, 0, NULL, NULL});
        }

        /*
            @d_model, name from the paper "Attention is all we need" we call it "dimensionsOfTheModel". 
            @num_heads, Number of attention heads.            
         */
        Attention(cc_tokenizer::string_character_traits<char>::size_type d_model, cc_tokenizer::string_character_traits<char>::size_type num_heads) : dimensionsOfAttentionHead((double)(d_model/num_heads)), dimensionsOfTheModel(d_model), numberOfAttentionHeads(num_heads)
        { 
            /*DIMENSIONS dim3 = DIMENSIONS{10, 3, NULL, NULL};
            DIMENSIONS dim2 = DIMENSIONS{0, 10, &dim3, NULL};
            dim3.prev = &dim2;
            DIMENSIONS dim1 = DIMENSIONS{0, 78, &dim2, NULL};
            dim2.prev = &dim1;
            DIMENSIONS dim = DIMENSIONS{0, 9, &dim1, NULL};
            dim1.prev = &dim;*/

            //DIMENSIONS dim3(DIMENSIONS{10, 3, NULL, NULL});
            //DIMENSIONS dim2(DIMENSIONS{0, 10, &dim3, NULL});

            //cc_tokenizer::string_character_traits<char>::size_type *ptr = cc_tokenizer::allocator<cc_tokenizer::string_character_traits<char>::size_type>().allocate(5);            
            //cc_tokenizer::string_character_traits<char>::size_type *ptr = reinterpret_cast<cc_tokenizer::string_character_traits<char>::size_type*>(cc_tokenizer::allocator<unsigned int>().allocate(5));
            DIMENSIONS dim = DIMENSIONS{d_model, d_model, NULL, NULL};
            queryWeights = Numcy::Random::randn<t>(dim);
            keyWeights = Numcy::Random::randn<t>(dim);
            valueWeights = Numcy::Random::randn<t>(dim);

            outputWeights = Numcy::Random::randn<t>(dim);
        }

        /*
            In the context of Transformers, self-attention is a mechanism where the model attends to different positions within the 
            same sequence to capture relationships and dependencies between words. Essentially, it's like asking questions about 
            different parts of the input sequence and using the answers to understand the overall meaning.

            Here's how the self-attention works with the same argument for query, key, and value:
            1. Each element in input sequence(@ei) acts as a query, key, and value:- 
               Every element (word) in the input sequence(@ei) is used as a query 
               to ask a question about the relationship with other words. Simultaneously, it also acts as a key for other elements to 
               compare against and as a value that holds information potentially relevant to the queries. 
            2. Attention scores:- 
               The Multi-Head Attention module calculates attention scores by comparing each query with all the keys.
               These scores indicate how relevant each value is to the specific query,
               considering both the content of the words and their positions.
            3. Weighted sum:-
               The attention scores are used to compute a weighted sum of the values.
               Words with higher attention scores contribute more to the final output, 
               effectively focusing on the most relevant parts of the sequence for each query.

            By using the same encoder input sequence(@ei) for query, key, and value, the self-attention mechanism allows the 
            model to learn relationships between any two words within the sequence, regardless of their order. 
            This is crucial for capturing long-range dependencies and the overall context of the input.         
         */

        /*            
            @ei_query, 
            - Represents the sequence of queries on which attention is being computed.
            - Each query is a vector that's compared to keys(@ei_key) to determine relevant values.
            - It's like asking a question to gat a focus on specific parts of the input.            
            @ei_key,
            - Contains the keys used to compute attention scores for each query.
            - Each key corresponds to a value in the @ei_value tensor.
            - Keys(@ei_keys) are used to compare with queries(@ei_query) to find relevant information.
            @ei_value,
            - Holds the values that will be selectively attended to based on the attention scores.
            - The values are the actual information that the attention mechanism extracts. 

            Key Points:
            1 Attention scores are computed using the dot product between queries(@ei_query) and keys(@ei_key), scaled by the square root of the head_dim().
            2 The model learns to assign higher attention scores to key-value pairs that are more relevant to the given query.
            - The return value...
            3 The final output of the forward method is the weighted sum of the values, where the weights are the normalized attention scores.
         */
        //template <typename t = float>
        void forward(Collective<t>& ei_query, Collective<t> &ei_key, Collective<t> &ei_value)
        { 
            Numcy::Random::randn<t>(DIMENSIONS{0, 0, NULL, NULL});
        }

        ~Attention()
        {                        
        }


} /*ATTENTION, MULTIHEADATTENTION*/;

#endif