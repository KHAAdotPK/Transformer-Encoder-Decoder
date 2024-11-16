### `Input sequence` and `Target sequence`.
---
When training a transformer model, a typical dataset consists of pairs of `input sequence`s and corresponding `target sequence`s.
In a trained Transformer model, the `input sequence` serves as the `prompt` or starting point for generating a response, influencing the content and direction of the model's `output swquence`(that is why, the encoder decoder model is also called `sequence-to-sequence`).
* Purpose of `input sequence` and/or `target sequence`:
    * Serves as training data during model training.
    For instance, an `input sequence` could be a sentence, and its corresponding `target sequence` would indicate whether the sentence is negative or positive.
        * __Example__, during training of model written to analyze sentiment: 
            * `Input sequence`: A sentence like "The weather is great!"
            * `Target sequence`: A label indicating sentiment (e.g., `positive`)

    * Acts as the query or question when interacting with a trained model.
        * __Example__, when interacting with trained model to analyze sentiment:
            * `Input sequence`: A declarative sentence "The weather is beautiful today."
            * `Output selection`/sequence: The model's response, such as "It is indeed! What are your plans for enjoying it?"

The `target sequence` acts as a guide during training hence shaping the model's `internal representation and internal parameters`. The decoder solely relies on its `knowledge/parameters` (the weights and biases or internal representation and internal parameters). The model's `knowledge/parameters` (weights and biases) are adjusted through `backpropagation` based on the `calculated loss`. This process aims to minimize the difference between the generated output (`output sequence/selection`) and the `target sequence`.

__In our implementation__:
Defined as `Collective` parameterized by a data type (e.g., `float`).
This data type (the `float`) should match the requirements of the specific model and training data.
    
```C++ 
struct Collective<float> inputSequence;
struct Collective<float> targetSequence;
```
__Important considerations__:
* Data type (the `float` in our case) should match the requirements of the specific model and training data.
* Preprocessing might be required for the input sequence, such as tokenization.
* The format of the input sequence (e.g., list of tokens, padded tensor) depends on the specific model implementation.

---
__Building and Processing Input and Target Sequences for Transformer Training with Line Batch Size__.

---
The following `C++` macros and training loop code are part of a program implementing a training loop for a Transformer model with an encoder-decoder architecture. The code includes macros for building input and target sequences from CSV parsers and vocabularies, and a training loop that iterates over epochs and training instances. The ellipsis (...) in the training loop represents the part of the code where the actual training operations take place, such as forward and backward passes, optimization, and updating model parameters.

__Key Components__:
1. Input Sequence Macro:
    * `BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE`: Constructs an input sequence from a CSV parser and vocabulary.
```C++
/*
    @is, input sequence
    @v, vocabulary, it is input vocabulary
    @icp, input csv parser
    @t, type
 */
#define BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, v, p, t) {\
t* ptr = cc_tokenizer::allocator<t>().allocate(p.get_total_number_of_tokens());\
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < p.get_total_number_of_tokens(); i++)\
{\
    CORPUS_PTR ret = v(p.get_token_by_number(i + 1));\
    ptr[i] = ret->index;\
}\
\
is = Collective<t>(ptr, {static_cast<cc_tokenizer::string_character_traits<char>::size_type>(p.get_total_number_of_tokens()), 1, NULL, NULL});\
}\
```
2. Target Sequence Macro:
    * `BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE`: Constructs a target sequence from a CSV parser and target vocabulary.
```C++
/*
    @ts, target sequence
    @v, vocabulary, it is target vocabulary
    @tcp, target csv parser
    @t, type
 */
#define BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, v, p, t) {\
t* ptr = reinterpret_cast<t*>(cc_tokenizer::allocator<char>().allocate((p.get_total_number_of_tokens())*sizeof(t)));\
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < p.get_total_number_of_tokens(); i++)\
{\
    CORPUS_PTR ret = v(p.get_token_by_number(i + 1));\
    ptr[i] = ret->index;\
}\
\
ts = Collective<t>(ptr, {static_cast<cc_tokenizer::string_character_traits<char>::size_type>(p.get_total_number_of_tokens()), 1, NULL, NULL});\
}\
```
3. Training Loop Macro:
    * `TRAINING_LOOP_LINE_BATCH_SIZE`: Iterates over epochs and training instances, loading input and target sequences using the previously defined macros. The ellipsis (...) represents the core training operations.
```C++
/*
    @icp, input csv parser
    @tcp, target csv parser
    @ei, encoder input
    @di, decoder input
    @dm, dimensions of the model(d_model)
    @es, epochs, 
    @iv, input sequence vocabulary
    @tv, target sequence vocabulary
    @p, position
    @dt, division term
    @pe, position encoding
    @is, input sequence
    @ts, target sequence
    @t, type
    @v, be verbose when true
 */
#define TRAINING_LOOP_LINE_BATCH_SIZE(icp, tcp, ei, di, dm, es, iv, tv, p, dt, pe, is, ts, t, v)\
{\
    for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < es; i++)\
    {\
        if (v == true)\
        {\
        }\
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.get_number_of_lines(); j++)\
        {\
            icp.get_line_by_number(j + 1);\
            tcp.get_line_by_number(j + 1);\
            if (v == true)\
            {\
            }\
            BUILD_INPUT_SEQUENCE_FOR_LINE_BATCH_SIZE(is, iv, icp, t);\
            BUILD_TARGET_SEQUENCE_FOR_LINE_BATCH_SIZE(ts, tv, tcp, t);\
            
            ---------------------------------------
            -----------------------------------------------
            ----------------------------------------------------
        }\
        --------------------------------------
        -----------------------------------------------
        ----------------------------------------------------
    }\
    --------------------------------------
    -----------------------------------------------
    ----------------------------------------------------
```
