still working on this paper...


---
## ___Positional Encoding___
### `How Transformers See Words`.
---
### `d_model`
`d_model` is a crucial hyperparameter in Transformers: It sets the size of word embeddings and affects both model capacity and computational cost.
1.  `d_model` in the Original Paper:
    - The Transformer architecture was first introduced in the 2017 paper "Attention Is All You Need."
    - The paper indeed uses the term `d_model` to refer to the hyperparameter that determines the dimensionality of word embeddings and internal representations within the model.
2. Common Values of `d_model`:
    - The typical range for `d_model` values is 128 to 1024, with 256 being a common choice.
        - Balancing expressiveness and efficiency: 
            - Higher dimensions can capture more intricate patterns but require more resources.
            - Lower dimensions are computationally lighter but might limit model capabilities.
    - The optimal value can vary depending on factors like:
        - Task complexity
        - Dataset size
        - Available computational resources
        - Desired performance

It's often a hyperparameter that's tuned during model training to find the best setting for a specific use case.
```C++
#define COMMAND "\n\
                 \n\
                 \n\
                 \n\
                 dmodel --dmodel (Its a hperparameter, the dimension of the model)\n\
                 \n\
                 \n\
                 sfc --sfc (Scaling factor constant)\n\
                 \n\
                 \n\
                 "
ARG arg_dmodel;
cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
cc_tokenizer::string_character_traits<char>::size_type dimensionsOfTheModelHyperparameter;
FIND_ARG(argv, argc, argsv_parser, "--dmodel", arg_dmodel);
if (arg_dmodel.argc) 
{
    dimensionsOfTheModelHyperparameter = std::atoi(argv[arg_dmodel.i + 1]);   
}
else
{
    dimensionsOfTheModelHyperparameter = DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER;
}
```
### ```Positional Encoding```
The `positional encoding`(___Collective<float> positionEncoding___)  is crucial for the Transformer model, which lacks the inherent order awareness of recurrent neural networks. This encoding injects information about word order into the model's internal representation by combining 'position' data with a `division term`(___Collective<float> divisionTerm___). This enriched representation, not strictly a hidden state, then flows through both encoding and decoding stages, enabling the model to understand relationships between words even when they're far apart.

```C++
/*
 In the Transformer model, the "position Collective" is a special container that stores the positions of each word in a sentence. It's like a map that tells the model where each word is located. This information is crucial because it helps the model understand the order of words, which is essential for understanding meaning.
 */
struct Collective<float> position;
/*
    @p, position an instance of Collective composite
    @is, input sequence
    @dt, division term
    @dm, dimensions of the model(d_model)
    @pe, position encoding
    @t, type
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t) {\
p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], (t)1.0, {1, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), {1, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}};\
-----------------------------------------------------------\
-----------------------------------------------------------\
-----------------------------------------------------------\
-----------------------------------------------------------\
-----------------------------------------------------------\
-----------------------------------------------------------\
}\
```
```C++
/*
    To make this positional information usable for the model, it's transformed into a special kind of code/tensor called "positional encoding." It's like a secret language that tells the model about word order in a way it can understand. The model creates this code/tensor by doing a math trick: it multiplies the position Collective with another Collective called "divisionTerm." This multiplication creates the positional encoding, which is then used by the model to capture the relationships between words in a sentence, even if they're far apart.
 */
struct Collective<float> positionEncoding; 
```
```C++
/*
    In transformers, a common technique for incorporating sequence information is by adding positional encodings to the input embeddings.
    The positional encodings are generated using sine and cosine functions of different frequencies.
    The expression wrapped in the following macro is used to scale the frequency of the sinusoidal functions used to generate positional encodings. 

    div_term
    ----------    
    This expression is used in initializing "div_term".

    The expression is multiplyed by -1
    ------------------------------------
    The resulting "div_term" array contains values that can be used as divisors when computing the sine and cosine values for positional encodings.
    
    Later on "div_term" and "positions" are used with sin() cos() functions to generate those sinusoidal positional encodings.
    The idea is that by increasing the frequency linearly with the position,
    the model can learn to make fine-grained distinctions for smaller positions and coarser distinctions for larger positions.
    
    @sfc, Scaling Factor Constant.
    @d_model, Dimensions of the transformer model.
 */
#define SCALING_FACTOR(sfc, d_model) -1*(log(sfc)/d_model)
```
```C++
/*
    This tensor is a part of the positional encoding calculation used to provide the model with information about the positions of tokens in the input sequence.
    The purpose of "div_term" is to scale the frequencies of the sinusoidal functions. It does that by working as divisor when computing the sine and cosine values for positional encodings.
    Think of it like a tuning fork: div_term helps the model adjust the frequencies of certain mathematical waves, called sine and cosine waves, to match the positions of words in a sentence. This tuning process is crucial because it allows the model to capture relationships between words that are far apart, making it better at understanding long and complex sentences.
 */
struct Collective<float> divisionTerm;

/*
    @p, position an instance of Collective composite
    @is, input sequence
    @dt, division term
    @dm, dimensions of the model(d_model)
    @pe, position encoding
    @t, type
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t) {\
-----------------------------------------------------------\
dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, {dm/2, 1, NULL, NULL}), dm/2), {dm/2, 1, NULL, NULL}};\
dt = dt * SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm);\
-----------------------------------------------------------\
-----------------------------------------------------------\
-----------------------------------------------------------\
-----------------------------------------------------------\
}\
```
```C++
/*
    @p, position an instance of Collective composite
    @is, input sequence
    @dt, division term
    @dm, dimensions of the model(d_model)
    @pe, position encoding
    @t, type
 */
#define BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t) {\
p = Collective<t>{Numcy::arange<t, t>((t)0.0, (t)is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], (t)1.0, {1, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}), {1, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL}};\
dt = Collective<t>{Numcy::exp<t>(Numcy::arange<t, t>((t)0.0, (t)dm, (t)2.0, {dm/2, 1, NULL, NULL}), dm/2), {dm/2, 1, NULL, NULL}};\
dt = dt * SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm);\
pe = Numcy::zeros<t>({dm, is.shape[NUMCY_DIMENSIONS_SHAPE_COLUMNS], NULL, NULL});\
FILL_EVEN_INDICES_OF_POSITION_ENCODING(pe, Numcy::sin<t>(p * dt));\
FILL_ODD_INDICES_OF_POSITION_ENCODING(pe, Numcy::cos<t>(p * dt));\
}\

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
           \
        }\
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < iv.get_number_of_lines(); j++)\
        {\
            icp.get_line_by_number(j + 1);\
            tcp.get_line_by_number(j + 1);\
            if (v == true)\
            {\
                \
            }\
            ---------------------------------------\
            -----------------------------------------------\
            BUILD_POSITION_ENCODING_FOR_LINE_BATCH_SIZE(p, is, dt, dm, pe, t);\
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