/*
    ML/NLP/transformers/encoder-decoder/Layer.hh
    Q@khaa.pk
 */

 template <typename t = double>
 class Layer 
 {
     public:
         virtual Collective<t> forward(Collective<t>& input, bool is_training = true) = 0;
         virtual ~Layer() = default;
 };