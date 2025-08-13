# Transformer Position Encoding Analysis

#### Written by, Sohail Qayum Malik

(PLEASE NOTE:- THERE ARE ERRORS INN THIS DOCUMENT)
(IMPLEMENTATON IS CREATING POSITION ENCODINGS CORRECTLY)

---

## Project Overview

This series of documents provide a comprehensive analysis of a custom C++ transformer implementation, focusing on the complete pipeline from input sequence processing through encoder input/output preparation, decoder input/output preparation. The implementation represents a complete from-scratch build of the transformer architecture, including custom word embeddings, novel position encoding, and sophisticated build system architecture.

The following detail allows transformer to process variable-length sentences/batches while maintaining perfect positional awareness!

## What is Position Encoding?

Position encoding tells the transformer model **where each word sits in a sentence**. Think of it like giving each word a unique "address" - just like houses on a street have numbers so you know their location.

Without position encoding, the transformer would see the words "cat chased mouse" and "mouse chased cat" as identical, since it processes all words simultaneously rather than one by one.

## The Challenge: Variable-Length Sentences

Real sentences have different lengths, but our model needs fixed-size inputs(`fixed number of words/token in each sentence of a training corpus`). In our corpus, every sentence has a **maximum of say 3 words**. But some sentences are shorter, so we need to add padding to make them all the same length(`3 words/tokens per sentence`).

For example:
- A sentence like "Hello world" (2 words) becomes "Hello world [PAD]" (3 positions)
- A sentence like "AI is amazing" (3 words) stays "AI is amazing" (3 positions)

The following debug output shows **one single sentence** that has only **2 real words** and **1 padding token**.

```text
sentence: "Hello World [PAD]"
mask: 1 1 0  // 1 = real word, 0 = padding
Shape of mask is (1 row, 3 columns)
```

## Step-by-Step Breakdown

### ~~Step 1: Create Position Numbers~~

```text
p: [1, 2, 3] // Because our max sentence length is 3, and according to our mask that third word is represented by padding byte (look at the mask)
Shape of p is (3 rows, 1 column)
```

### Step 2: Get product of p and mask (Apply Masking)
```C++
// Perform element-wise multiplication between the position matrix `p` and the mask matrix `mask`
for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < mntpl_input; i++)
{             
    p[i] = p[i] * mask[i];                 
}

// Because our numeric library has not yet provided the broadcasting feature
p = p * mask;
```
```text
p (m, n) , mask (n, p) there for p is now (m, p) so p is now (3, 3)

1            1.1 1.1 1.0   1 1 0
2 * 1 1 0 =  2.1 2.1 2.0 = 2 2 0 
0            0.1 0.1 0.0   0 0 0 
```

### Step 3: Create Frequency Scales/Patterns (dt, division terms)

The next step is to create different "frequencies" for each of the n (for this implementation default is 64) embedding dimensions. Think of it like tuning forks - each dimension vibrates at a different frequency.

```text
frequency = 1 / (10000^(dimension_pair/64)) // dimension pairing, the duplicate values for different trignometric functions
```

From the debug output: 
```text
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 0.0705842 0.0705842 0.00498213 0.00498213 0.00035166 0.00035166 2.48216e-05 2.48216e-05 1.75201e-06 1.75201e-06 1.23664e-07 1.23664e-07 8.72875e-09 8.72875e-09 
2 2 0.141168 0.141168 0.00996426 0.00996426 0.000703319 0.000703319 4.96432e-05 4.96432e-05 3.50403e-06 3.50403e-06 2.47329e-07 2.47329e-07 1.74575e-08 1.74575e-08 
```

**Key insight**: Values/Dimensions come in pairs (`this is the dimension pairing in action`) because:
- Dimensions 0&1 share frequency, 2&3 share frequency, 4&5 share frequency, etc.
- Each pair shares the same base frequency but uses different trigonometric functions(sine for first member of pair and cosine for second member of the pair).
- Frequency is scaled and it uses exponential decay with dimension pairing so frequencies get smaller for higher dimensions (like 1 → 0.749894 → 0.562341...).
- When 0-based indexing is used for tokens, the vector 1 is vectors of all zeros. 
- The vector for pos=2 should be roughly double the vector for pos=1 before the sin/cos functions are applied. 

### Step 4: Apply Sine and Cosine Functions

We create the final position encoding by applying:

- **Sine function** to even-indexed dimensions (0, 2, 4, ...)
- **Cosine function** to odd-indexed dimensions (1, 3, 5, ...)
Hence the **Pattern**: sine, cosine, sine, cosine, sine, cosine... across all n (`in our case 64`) dimensions

This creates a unique "fingerprint" for each position (`out of n many positions of embedding dimensions for each word/token of a sentence`).

## The Magic: Why This Works

Each position gets a unique n-dimensional vector that has special mathematical properties:

1. **Uniqueness**: No two positions (`out of n many positions of embedding dimensions for each word/token of a sentence`) have the same encoding
2. **Relative positioning**: The model can learn relationships between positions
3. **Scalability**: Works for any sequence length (`up to our maximum number of word/tokens per sentence or batch`)
4. **Smooth patterns**: Similar positions have similar encodings (`between two or more sentences/batches`)

## Real-World Analogy

Think of it like a music box with 64 different tuning forks:
- Each position in your sentence strikes these tuning forks with different strengths
- Position 1 creates one unique "chord" across all 64 forks
- Position 2 creates a different "chord" 
- Position 3 (if it exists) creates yet another "chord"
- Padding positions create silence (all zeros)

## Debug Output Interpretation

Looking at our actual output:

```text
Position 1 encoding: [0.909297, -0.416147, 0.99748, 0.0709483, ...] // For word/token 1 
Position 2 encoding: [-0.756802, -0.653644, 0.141539, -0.989933, ...] // For word/token 2
Position 3 encoding: [0, 0, 0, 0, ...]  ← All zeros because it's padding! // Becuase the sentence/batch has only two words/tokens
```

**Key Observations**: `Values vary naturally`, the sine and cosine functions produce different ranges of values (both positive and negative) depending on the input. `Different patterns per position`, each position creates a unique mathematical signature. `Perfect padding masking`, position 3 is all zeros because it represents padding as this sentence/batch only had two words/tokens.  

## Summary

Position encoding transforms simple position numbers into rich, mathematical representations that help transformers understand word order. Our implementation (Model::buildPositionEncodings()):

✅ Handles variable-length sentences with padding  
✅ Uses proven sinusoidal encoding mathematics  
✅ Properly masks padding tokens  
✅ Creates unique fingerprints for each position

This foundation enables the transformer to process language with full awareness of word positioning and relationships.

---

*Document generated from comprehensive analysis of position encoding implementation which is part of encoder input transformation pipeline.*

## Build System Architecture

The implementation features a sophisticated conditional compilation system:

### Debug Flag
- `verbose_pe`: Position Encoding verbose debugging

### Build Configuration
```batch
msbuild project.xml /p:BuildPositionEncodingVerbose=yes

usage/RUN.cmd build verbose_pe
```

```C++
/**
 * @brief Constructs position encoding for a batch of input sequences using sinusoidal encoding.
 *
 * This function generates position encoding vectors that will be used in 
 * transformer-based models to retain positional information. It implements 
 * the standard sinusoidal position encoding scheme where:
 * - Even dimensions use sine function: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
 * - Odd dimensions use cosine function: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * The function handles padding tokens by applying a mask to ensure they receive 
 * zero position encodings, maintaining the semantic meaning that padding tokens 
 * do not contribute to positional understanding.
 *
 * @param p  An output parameter of type `Collective<t>` representing position indices.
 *           Shape: [1 x mntpl_input]. Contains sequential position values from 
 *           POSITIONAL_ENCODING_START_VALUE to sequence length, masked for padding.
 * 
 * @param pe An output parameter of type `Collective<t>` that stores the final position encodings.
 *           Shape: [sequence_length x dm]. Contains alternating sine/cosine values 
 *           for each position and embedding dimension.
 * 
 * @param dt An output parameter of type `Collective<t>` representing the division/scaling term.
 *           Shape: [sequence_length x dm]. Contains the exponential scaling factors 
 *           computed as exp(-log(10000) * 2*dim_pair / dm) for frequency modulation.
 * 
 * @param dm The model's embedding dimension. Must be even for proper sine/cosine pairing.
 * 
 * @param is An input tensor representing the input sequence batch (currently unused in implementation).
 *  
 * @param mask A mask tensor of shape [1 x mntpl_input] that differentiates real tokens (1) 
 *        from padding tokens (0). Padding tokens should not receive valid position 
 *        encodings because they do not contribute to the model's understanding of 
 *        sequence structure. Padding tokens are added to make all input sequences 
 *        uniform in length.
 * 
 * @param mntpl_input Maximum number of tokens per line (sequence length). Each input 
 *        sequence is padded to ensure uniform length across variable-length sequences. 
 *        If an input line has fewer tokens, padding is added to match this required length.
 * 
 * @param sin_transformed_product An output parameter storing sine-transformed position*frequency products.
 *        Used as intermediate storage for sine calculations before final assignment.
 * 
 * @param cos_transformed_product An output parameter storing cosine-transformed position*frequency products.
 *        Used as intermediate storage for cosine calculations before final assignment.
 *
 * @throws ala_exception Thrown on memory allocation errors, length errors, or other exceptions
 *         during position encoding computation.
 *
 * Algorithm Steps:
 * 1. Generate position indices using arange() from start value to sequence length
 * 2. Apply mask to zero out positions corresponding to padding tokens
 * 3. Compute scaling factors (dt) using exponential decay based on dimension pairs
 * 4. Calculate position*frequency products for all positions and dimensions
 * 5. Apply sine transformation for even indices, cosine for odd indices
 * 6. Fill final position encoding matrix with masked sine/cosine values
 *
 * Mathematical Foundation:
 * - Frequency decreases exponentially with dimension: freq = 1/10000^(2i/d_model)
 * - Dimension pairing: dimensions (2i, 2i+1) share the same base frequency
 * - Masking ensures padding positions contribute zero to attention computations
 *
 * Memory Layout Notes:
 * - All matrices use row-major ordering for element access
 * - Position encodings are computed element-wise without broadcasting
 * - Final PE matrix has alternating sine/cosine pattern across dimensions
 */
/*
    m    n
    p = mntpl_input x 1
    mask = 1 x mntpl_input
    n    p           
    m x p                 
    p * mask   
 */
void buildPositionEncoding(Collective<t>& p, Collective<t>& pe, Collective<t>& dt, cc_tokenizer::string_character_traits<char>::size_type dm, Collective<t>& is, Collective<t>& mask, cc_tokenizer::string_character_traits<char>::size_type mntpl_input, Collective<t>& sin_transformed_product, Collective<t>& cos_transformed_product) throw (ala_exception)
{            
    try
    {   /*
            Generate position indices: range from POSITIONAL_ENCODING_START_VALUE(inclusive) to input sequence-length(exclusive), sequence-length is the number of tokens in a line.        
         */
        p = Collective<t>{Numcy::arange<t, t>((t)POSITIONAL_ENCODING_START_VALUE, (t)mntpl_input + (t)(t)POSITIONAL_ENCODING_START_VALUE, (t)1.0, DIMENSIONS{1, mntpl_input, NULL, NULL}), DIMENSIONS{1, mntpl_input, NULL, NULL}};                
 
        /*
         * Perform element-wise multiplication between the position matrix `p` and the mask matrix `mask`.
         *
         * Why this is done:
         * - The position matrix `p` contains positional encodings for each token in the sequence.
         * - The mask matrix `mask` indicates which tokens are valid (1) and which are padding (0).
         * - By multiplying `p` and `mask` element-wise, we ensure that positional encodings for invalid tokens
         *   (padding) are zeroed out, while valid tokens retain their positional values.
         *
         * How this works:
         * - Both `p` and `mask` have the same number of elements, but they may have different shapes.
         * - The `[]` operator is overloaded to access elements linearly, regardless of the shapes of `p` and `mask`.
         * - The loop iterates over each element of `p` and `mask`, multiplying them together and storing the result in `p`.
         *
         * Example:
         * - If `p` is [1, 2, 3] and `mask` is [1, 1, 0], the result will be [1, 2, 0].
         * - This ensures that the positional encoding for the third token (padding) is zeroed out.
         *
         * Note:
         * - This is NOT broadcasting. Broadcasting would automatically expand the smaller array to match the shape
         *   of the larger array, but here we are explicitly iterating over the elements and performing the
         *   multiplication manually.
         */                
         for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < mntpl_input; i++)
         {             
            p[i] = p[i] * mask[i];                 
         }                
        /*
            // n in p must equal to m in mask
            // p mow becomes matrix of colmns from mask and rows from p
         */
        p = p * mask;
                                
        //p = Numcy::transpose<t>(p);

        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < dt.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); i++)
        {
            /*t value = (t)POSITIONAL_ENCODING_START_VALUE;*/

            for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < dm; j++)
            {                        
                /* 
                    // Standard positional encoding:
                    // 1.0 / std::pow(10000.0, value / (t)dm); 
                */

                t dimension_pair = (t)0;

                // For even j: use j/2, for odd j: use (j-1)/2 
                // This ensures pairs of dimensions have the same base frequency 
                if ( (j % 2) == 0 ) // Even
                {
                    dimension_pair = (t)(j / 2); // Integer division
                } 
                else // Odd 
                {
                    dimension_pair = (t)((j - 1) / 2); // Integer division
                }
                        
                t exponent = -std::log(10000.0) * (2.0 * dimension_pair) / (t)dm;
                dt[i * dt.getShape().getNumberOfColumns() + j] = std::exp(exponent);

                /*t exponent = -std::log(10000.0) * (2.0 * j) / (t)dm;
                dt[i*dt.getShape().getNumberOfColumns() + j] = std::exp(exponent);*/

                /*dt[i*dt.getShape().getNumberOfColumns() + j] = std::exp(value * (t)(SCALING_FACTOR(SCALING_FACTOR_CONSTANT, dm)));*/
 
                /*value = value + (t)(2*i);  // Increments by 2*/
            }              
        }                
 
        Collective<t> p_to_dt = p * dt;

        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < p_to_dt.getShape().getN(); i++)
        {
            sin_transformed_product[i] = std::sin(p_to_dt[i]);
            cos_transformed_product[i] = std::cos(p_to_dt[i]);
        }                          
#ifdef MAKE_THIS_MODEL_VERBOSE_FOR_POSITION_ENCODING                
        /*std::cout<< "sin_transformed_product, Columns: " << sin_transformed_product.getShape().getNumberOfColumns() << ", Rows: " << sin_transformed_product.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << std::endl;*/
#endif                                
        /* Fill even and odd indices separately */
        for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < pe.getShape().getN(); i+=2)
        {                       
            pe[i] = sin_transformed_product[i] * mask[i/pe.getShape().getNumberOfColumns()];
        }
        for (cc_tokenizer::string_character_traits<char>::size_type i = 1; i < pe.getShape().getN(); i+=2)
        {
            pe[i] = cos_transformed_product[i] * mask[i/pe.getShape().getNumberOfColumns()];
        }

        // 64 rows, 3 columns                         3 rows and 1 column   
        //Numcy::transpose(sin_transformed_product) * Numcy::transpose(mask);
    }
    catch (std::bad_alloc& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding() Error: ") + cc_tokenizer::String<char>(e.what()));
    }
    catch (std::length_error& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding() Error: ") + cc_tokenizer::String<char>(e.what()));
    }            
    catch (ala_exception& e)
    {
        throw ala_exception(cc_tokenizer::String<char>("Model::buildPositionEncoding() -> ") + cc_tokenizer::String<char>(e.what()));
    }
}
```