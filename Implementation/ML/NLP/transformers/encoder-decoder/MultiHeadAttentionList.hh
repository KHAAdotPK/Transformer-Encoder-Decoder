/*
    ML/NLP/transformers/encoder-decoder/MultiHeadAttentionList.hh
    Q@khaa.pk
 */

#include "./header.hh" 

#ifndef NLP_ENCODER_DECODER_TRANSFORMER_MODEL_MULTI_HEAD_ATTENTION_LIST_HH
#define NLP_ENCODER_DECODER_TRANSFORMER_MODEL_MULTI_HEAD_ATTENTION_LIST_HH

/*
    MultiHeadAttentionList represents a doubly linked list of attention heads in a Transformer’s multi-head attention (MHA) mechanism.

    ==================== Purpose ====================
    In the Transformer architecture (as described in "Attention is All You Need"), multi-head attention (MHA) is a core component of each encoder and decoder layer. MHA consists of multiple attention heads that operate in parallel, each processing a subspace of the input to capture diverse relationships within the sequence. The `MultiHeadAttentionList` structure organizes these attention heads as a doubly linked list, allowing flexible management of multiple heads within a single MHA layer.

    This structure is used to store and iterate over multiple instances of the `Attention` class, each representing one attention head with its own query, key, and value projection matrices. The outputs of these heads are later concatenated to produce the final MHA output.

    ==================== Structure ====================
    struct MultiHeadAttentionList {
        Attention<t>* ptr;                  // Pointer to an individual attention head (scaled dot-product attention)
        MultiHeadAttentionList<t>* next;    // Pointer to the next attention head in the list
        MultiHeadAttentionList<t>* previous; // Pointer to the previous attention head in the list
    };

    ==================== Benefits of Linked Structure ====================
    - **Flexibility in Head Management**: The linked list allows dynamic addition or removal of attention heads, making it easy to experiment with different numbers of heads without resizing fixed-size containers like arrays or vectors.
    - **Clear Navigation**: The `previous` and `next` pointers enable easy traversal of heads, which is useful during forward and backward passes or for debugging individual head behaviors.
    - **Decoupled Allocation**: Each attention head can be independently allocated and deallocated, providing fine-grained control over memory usage, which is particularly useful in custom or low-level implementations.
    - **Educational Clarity**: The linked list structure mirrors the conceptual modularity of MHA, making it easier to understand and teach how multiple heads contribute to the overall attention mechanism.

    ==================== Usage Flow ====================
    1. Each node in the `MultiHeadAttentionList` holds an `Attention` object that implements scaled dot-product attention for a specific subspace of the input (dimension `d_k = d_model / num_heads`).
    2. During the forward pass of the multi-head attention mechanism:
        - The input sequence (e.g., embedded tokens of shape `[seq_len, d_model]`) is passed to each attention head in the list.
        - Each head computes its own attention output (shape `[seq_len, d_k]`) using its unique projection matrices.
        - The outputs from all heads are collected (via iteration over the list) and concatenated along the feature dimension to produce a single tensor (shape `[seq_len, d_model]`).
        - A final linear projection is applied to the concatenated output to produce the MHA output.
    3. During the backward pass:
        - The incoming gradient is split across heads (based on their respective subspaces).
        - Each head’s `backward` method is called to compute gradients for its projection matrices and input.
        - Gradients are summed across heads to compute the total gradient with respect to the input.
    4. The `next` and `previous` pointers facilitate iteration and gradient flow, ensuring all heads are processed in a consistent order.

    ==================== Design Considerations ====================
    - **Alternative Storage**: While a `std::vector<Attention<t>>` might offer better cache locality and simpler indexing, the doubly linked list provides a flexible, educational structure that emphasizes the modularity of attention heads. This is particularly valuable for low-level implementations or when teaching the Transformer architecture.
    - **Memory Overhead**: The linked list introduces pointer overhead (`next` and `previous`), which may be less efficient than a contiguous array for large numbers of heads. However, this trade-off is acceptable in educational or experimental contexts where flexibility is prioritized.
    - **Head Independence**: Each attention head operates independently, and the linked list structure reinforces this by decoupling heads from one another, allowing for potential modifications (e.g., adding specialized heads or pruning heads dynamically).
    - **Integration with EncoderLayer**: The `MultiHeadAttentionList` is typically managed by an `EncoderLayer` (or `DecoderLayer`), which orchestrates the forward and backward passes across all heads and handles the concatenation and final projection.

    ==================== In Context ====================
    The `MultiHeadAttentionList` is a critical component of the Transformer’s multi-head attention mechanism, which is used in both encoder and decoder layers. By allowing multiple attention heads to process the same input in parallel, it enables the model to capture diverse linguistic patterns (e.g., syntactic, semantic, or positional relationships) within a homogeneous input sequence, such as text embeddings.

    This structure is essential for building Transformer models for tasks such as:
        - Machine translation (e.g., English to French)
        - Text summarization
        - Question answering
        - Language modeling

    The `MultiHeadAttentionList` integrates with the `EncoderLayer` by providing the mechanism to compute multi-head attention, where each head’s output contributes to the final representation of the input sequence. See `Attention.hh` for the implementation of individual attention heads and `EncoderLayer.hh` for how this list is used within the broader encoder architecture.
*/
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
