/*
    ML/NLP/transformers/encoder-decoder/EncoderLayerList.hh
    Q@khaa.pk
 */

#include "./header.hh" 

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_LIST_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_ENCODER_LAYER_LIST_HH

/*
    EncoderLayerList represents a doubly linked list of encoder layers in a Transformer encoder stack.

    ==================== Purpose ====================
    In the Transformer architecture (as described in "Attention is All You Need"), the encoder is not a single unit but 
    a stack of *identical* encoder layers (usually 6 or 12). Each layer performs self-attention, followed by a feed-forward
    network, and each is wrapped with residual connections and layer normalization.

    Rather than using an array/vector of EncoderLayer objects, this implementation uses a doubly linked list to represent 
    a flexible and extendable chain of encoder layers.

    ==================== Structure ====================
    struct EncoderLayerList {
        EncoderLayer<t>* ptr;               // Pointer to the actual encoder layer (self-attention + FFN)
        EncoderLayerList<t>* next;          // Pointer to the next encoder layer in the stack
        EncoderLayerList<t>* previous;      // Pointer to the previous encoder layer in the stack
    };

    ==================== Benefits of Linked Structure ====================
    - **Flexibility in Construction**: Layers can be added or removed dynamically without resizing a contiguous array.
    - **Clear Navigation**: The 'previous' pointer allows easy backtracking (useful in training/debugging).
    - **Memory Management**: Decouples layer allocation from model management—layers can be individually allocated/freed.

    ==================== Usage Flow ====================
    1. Each node holds an `EncoderLayer` that implements its own `forward()` logic.
    2. During forward propagation through the encoder stack:
        - The input (a sequence of vectors) is passed to the first layer.
        - Its output is passed to the `next` EncoderLayerList node, and so on.
    3. This continues recursively or iteratively until the final encoder layer is reached.
    4. The output from the last EncoderLayer becomes the final encoded representation.

    ==================== Design Considerations ====================
    - Although vector-based storage might be more cache-friendly and efficient in real-time applications,
      the linked list structure reflects educational clarity and manual control, suitable for low-level
      or custom framework implementations like this one.
    - It is assumed that a separate managing class or model will hold a reference to the head of this list
      and control iteration through it.
    - In a training context, gradients can propagate backward using the `previous` pointers (manual backward pass).

    ==================== In Context ====================
    This list-based encoder stack directly reflects the architecture of the Transformer model's encoder block,
    where each EncoderLayer transforms the input to a richer representation by capturing dependencies between tokens.

    This structure is essential in building encoder-decoder models for tasks such as:
        - Machine translation
        - Text summarization
        - Question answering
        - Language modeling

    See EncoderLayer.hh for internal layer logic and attention mechanisms.    
 */
template <typename t>
class EncoderLayer;

template <typename t = double>
struct EncoderLayerList
{
    /*
        Transformer encoder layer
     */
    class EncoderLayer<t>* ptr; 

    struct EncoderLayerList<t>* next;
    struct EncoderLayerList<t>* previous;
};

#endif