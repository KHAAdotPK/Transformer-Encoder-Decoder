/*
    ML/NLP/transformers/encoder-decoder/EncoderFeedForwardNetwork.hh
    Q@khaa.pk
 */

#include "header.hh"

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_FEED_FORWARD_NETWORK_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_FEED_FORWARD_NETWORK_HH

/*
    The FeedForwardNetwork class implements a two-layer feedforward neural network with a ReLU activation function, as described in the "Attention is All You Need" paper.
    It is designed to process the output of the multi-head attention mechanism and apply a non-linear transformation to it.
 */

template <typename t = double>
class EncoderFeedForwardNetwork 
{
    cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModel;
    t dropOutRate;
    Collective<t> weights1, weights2, bias1, bias2;

    public:
                
        EncoderFeedForwardNetwork(cc_tokenizer::string_character_traits<char>::size_type d_model, t dropout_rate) : dimensionsOfTheModel(d_model), dropOutRate(dropout_rate)
        {
            DIMENSIONS dim1 = DIMENSIONS{d_model, 4 * d_model, NULL, NULL};  // Expand
            DIMENSIONS dim2 = DIMENSIONS{4 * d_model, d_model, NULL, NULL};  // Reduce back

            weights1 = Numcy::Random::randn<t>(dim1);
            weights2 = Numcy::Random::randn<t>(dim2);
            bias1 = Numcy::Random::randn<t>(dim1);
            bias2 = Numcy::Random::randn<t>(dim2);
        }

        void forward(Collective<t>& input)
        {
            /*input = relu(matmul(input, weights1) + bias1);
            input = matmul(input, weights2) + bias2;*/
            // Apply dropout
        }
};

#endif
