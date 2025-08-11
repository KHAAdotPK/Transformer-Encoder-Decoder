# Transformer Input Sequence Analysis

#### Written by, Sohail Qayum Malik

---

## Project Overview

This series of documents provide a comprehensive analysis of a custom C++ transformer implementation, focusing on the complete pipeline from input sequence processing through encoder input/output preparation, decoder input/output preparation. The implementation represents a complete from-scratch build of the transformer architecture, including custom word embeddings, novel position encoding, and sophisticated build system architecture.

## What is Input Sequencing?

Input Sequencing is the process of preparing text data for a transformer model to understand. Think of it like translating human language into numbers that a computer can work with.

## The Big Picture

Our transformer model needs to:
1. Take sentences (like "Hello world")
2. Convert each word into numbers (`word embeddings, custom-trained word embeddings using Skip-gram/CBOW algorithms`)
3. Arrange these numbers so the model can process them
4. Tell the model which parts are real words and which parts are just padding

## How We Build Input Sequences

### Step 1: Convert Words to Numbers

Each word in our vocabulary gets converted to a n-dimensional vector (`a list of n in our case n is 16 numbers`). These vectors come from pre-trained word embeddings that capture the meaning of words.

**Example from our output:**
- Token 1 becomes: `0.729382, -0.020946, -0.0216489, ...` (16 numbers total)
- Token 2 becomes: `2.24245, -0.0792378, -0.0660413, ...` (16 numbers total)

### Step 2: Create Fixed-Length Sequences

All sentences must be the same length for efficient processing. In our example:
- **Maximum sequence length**: 3 positions
- **Actual tokens**: 2 words
- **Padding needed**: 1 position filled with zeros

This creates a matrix:

```text
Position 1: [0.729382, -0.020946, -0.0216489, ...] <- Real word
Position 2: [2.24245, -0.0792378, -0.0660413, ...]  <- Real word  
Position 3: [0, 0, 0, 0, 0, 0, 0, ...]             <- Padding
```
### Step 3: Create an Attention Mask

The model needs to know which positions contain real words and which are just padding. We create a mask:

```text
Mask: [1, 1, 0]
```
- `1` means "this is a real word, pay attention to it"
- `0` means "this is padding, ignore it"

## Understanding the following Debug Output

```text
::: DEBUG DATA -: Model::buildInputSequence() :- :::
Number of tokens in this line: 2
is(Input Sequence), Columns: 16, Rows: 3
0.729382 -0.020946 -0.0216489 -0.0505344 0.0730361 0.013116 0.155757 0.0192252 -0.129759 0.0439584 -0.0528336 0.028011 0.0216742 -0.110869 0.0733035 -0.0746424 
2.24245 -0.0792378 -0.0660413 -0.00121151 -0.0352882 0.0374772 -0.0400047 0.0446142 0.0542433 0.0296386 0.066942 0.0646408 0.0355952 0.0190345 -0.0222506 0.0231328 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
mask, Columns: 3, Rows: 1
1 1 0 
```
Let's break down what the code produced:

### Input Data
```
Number of tokens in this line: 2
```
This sentence has 2 real words.

### The Input Sequence Matrix
```
is(Input Sequence), Columns: 16, Rows: 3
0.729382 -0.020946 -0.0216489 ... (Row 1 - First word)
2.24245 -0.0792378 -0.0660413 ... (Row 2 - Second word)
0 0 0 0 0 0 0 0 ...              (Row 3 - Padding)
```

- **3 rows**: One for each position in our sequence
- **16 columns**: Each word embedding has 16 dimensions
- **First 2 rows**: Real word embeddings
- **Last row**: All zeros (padding)

### The Attention Mask
```
mask, Columns: 3, Rows: 1
1 1 0
```

This tells the transformer:
- Position 1: Real word ✓
- Position 2: Real word ✓ 
- Position 3: Padding, ignore ✗

## Summary

The input sequence builder transforms raw text into a structured format that transformers can process efficiently. By converting words to embeddings, handling variable lengths with padding, and providing attention masks, we create the essential foundation for all subsequent transformer operations.

The debug output shows this process working correctly: 2 real tokens converted to embeddings, 1 padding position, and a proper attention mask to guide the model's focus.

## Conclusion

This C++ transformer implementation demonstrates a thorough, educational approach to understanding transformer architecture. The input sequence processing shows proper implementation of fundamental concepts including embedding handling, sequence padding, and attention masking. The sophisticated build system and debug infrastructure indicate a well-engineered approach to complex system development.

The decision to implement custom word embeddings and build from scratch, while challenging, provides deep understanding of each component's role in the overall architecture. The current input sequence implementation serves as a solid foundation for the more complex attention mechanisms to follow.

---

*Document generated from debug output analysis and implementation discussion.*

## Build System Architecture

The implementation features a sophisticated conditional compilation system:

### Debug Flag
- `verbose_is`: Input Sequence verbose debugging

### Build Configuration
```batch
msbuild project.xml /p:BuildInputSequenceVerbose=yes

usage/RUN.cmd build verbose_is
```

```C++
/*
 * --------------------------------------------
 * | BUILD INPUT SEQUENCE FOR ANY BATCH SIZE |
 * --------------------------------------------
 */   
/*
 * Builds an input sequence for a batch of tokens from a line of text.
 * 
 * This macro allocates memory and processes tokens to create an input sequence
 * using pre-trained word embeddings.
 * 
 * Parameters:
 * @is    - An output parameter of type Collective<t> that will store the final input sequence.
 * @v     - Vocabulary object that maps tokens to indices
 * @icp   - Input CSV parser object representing the input corpus, which provides token-related information.
 * @mntpl_input - Each input sequence is padded to ensure uniform length across variable-length sequences per line. 
 *          The value of maximum number of tokens/sequences per line (mntpl_input) determines the size of all input sequences. 
 *          If an input line has fewer tokens, padding is added to match the required length.              
 * @mask  - Padding tokens should not receive valid position encodings because they do not contribute to the model’s 
 *          understanding of sequence structure(padding tokens are added to make all input sequences 
 *          uniform in length). 
 *          Since positional encodings influence attention weights, allowing padding tokens to have meaningful encodings
 *          might lead to misleading attention patterns.
 *          You need a mask that differentiates real tokens from padding tokens. The mask should have:
 *          Value (DEFAULT_VALID_WORD_VECTOR_MASK_VALUE) for real tokens (to keep their positional encoding).
 *          Value (DEFAULT_PADDING_WORD_VECTOR_VALUE) for padding tokens (to zero out their positional encoding).
 * @t     - The data type of embeddings (e.g., float, double).
 * @w1    - Matrix of pre-trained word embeddings where each row represents a word vector.
 * @redundancy - Optional parameter that allows for multiple occurrences of the same token in the vocabulary.
 * @v    - Optional parameter that enables verbose output for debugging purposes.
 * 
 * Implementation:
 * 1. Allocates memory for all tokens * embedding dimension in the current line
 * 2. For each token in the line:
 *    - Looks up the token's index in the vocabulary
 *    - If found, copies the corresponding word embedding from w1
 *    - Each word embedding is a row vector from the w1 matrix
 * 3. Mask Setting in this Macro:
 *    - Memory is allocated for the mask (`ptr_mask`), with all values initially set to `DEFAULT_PADDING_WORD_VECTOR_VALUE`.
 *    - Inside the loop, when a valid token is found in the vocabulary, its corresponding mask index is set to `DEFAULT_VALID_WORD_VECTOR_MASK` (typically 1).
 *    - Padding tokens remain with their initial value (`DEFAULT_PADDING_WORD_VECTOR_VALUE`, typically 0), ensuring they are ignored in position encoding calculations.
 *    - Finally, the mask is wrapped in a `Collective<t>` object for use in downstream processing.
 *    (PLEASE NOTE:-  Implementation can ignore trailing padding, example: If all sequences are mntpl_input=10, but only the first 7 positions contain valid tokens, you can use sequence_length=7 instead of a mask.) 
 * 
 * Error Handling:
 * - Handles memory allocation failures (bad_alloc)
 * - Handles length errors 
 * - Handles custom ala_exceptions
 * - All errors are propagated with additional context
 *  
 * Note: The Vocabulary object uses internal indexing that starts at INDEX_ORIGINATES_AT_VALUE.
 *       In contrast, word embeddings use zero-based indexing (starting at 0).
 */    
void buildInputSequence(cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char>& icp, CORPUS& iv, Collective<t>& is, Collective<t>& mask, Collective<t>& W1, bool redundancy = ALLOW_REDUNDANCY, bool v = false) throw (ala_exception)
{                                                                                                                                     
    try
    {                           
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < icp.get_total_number_of_tokens(); i++)
        {
            /* Get the index of the token in the vocabulary. These indices originate at INDEX_ORIGINATE_AT_VALUE */
            cc_tokenizer::string_character_traits<char>::size_type index = iv(icp.get_token_by_number(i + 1), icp.get_current_line_number(), i + 1, redundancy);
                    
            /* If this condition is false, we are no longer strictly using post-padding; instead, padding tokens may appear */
            /* between valid tokens, leading to mixed padding. */
            /* TODO: Investigate whether the following statement can ever evaluate to be false, because under that circumstances */
            /* mixed padding might occur. */
            if (index != INDEX_NOT_FOUND_AT_VALUE)
            {
                /* Marking real tokens with a valid mask value */
                mask[i] = DEFAULT_VALID_WORD_VECTOR_MASK_VALUE;
            
                /* we, Word Embedding */
                Collective<t> we = W1.slice((index - INDEX_ORIGINATES_AT_VALUE /*TOKENIZER_PARSER_TOKEN_NUMBER_ORIGINATES_AT_INDEX*/ /*Indices in parser originate at 1 in parser*/)*W1.getShape().getNumberOfColumns(), W1.getShape().getNumberOfColumns());
                for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < we.getShape().getN(); j++)
                {
                    is[i*we.getShape().getN() + j] = we[j];
                }
            }
            else
            {
                /* Handling Vocabulary Lookup Failure: */
                /* ----------------------------------- */
                /* If the token is not found in the vocabulary (`index == INDEX_NOT_FOUND_AT_VALUE`), we must halt processing immediately and raise an exception. */
                /* It prevents mixed-padding: If we continue processing after encountering an unknown token, padding tokens may be inserted between valid tokens instead of at the end, violating  the post-padding strategy. */
                throw ala_exception("Model::buildInputSequence() Error: Encountered a token that is not present in the vocabulary. This should never happen if the inputs are within the expected range. Potential cause: Vocabulary is incomplete or incorrectly loaded.");
            }                
        }
#ifdef MAKE_THIS_MODEL_VERBOSE                     
#endif                
    }
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("Model::buildInputSequence() -> ") + cc_tokenizer::String<char>(e.what()));   
    }
}
```