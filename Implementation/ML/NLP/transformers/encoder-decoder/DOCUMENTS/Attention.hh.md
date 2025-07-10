## Key Components Analysis of Attention Class (of file attention.hh)
---

This document describes the implementation of the Multi-Head Attention (MHA) mechanism, a core component of Transformer models.

What is Multi-Head Attention?
Multi-Head Attention is the core mechanism that allows Transformer models to "pay attention" to different parts of the input sequence simultaneously. Think of it as having multiple "eyes" that can focus on different relationships in the data at the same time.

### Basic Concepts

#### The Three Key Components
- **Query (Q)**: "What am I looking for?" - represents the current position asking questions
- **Key (K)**: "What information do I have?" - represents all positions that can be attended to
- **Value (V)**: "What actual information do I contain?" - the actual content that gets extracted

### The Magic Formula
```
Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V
```

This formula means:
1. Compare queries with keys (Q·K^T)
2. Scale the results for stability (÷√d_k)
3. Apply softmax to get attention weights
4. Use these weights to combine values

### 1. Class Architecture Overview

The `Attention` class implements the scaled dot-product attention with multiple heads as described in "Attention Is All You Need". It enables the model to jointly attend to information from different representation subspaces at different positions.

### Template-Based Design
```cpp
template <typename t = double>
class Attention // Is all you need.
```

### Constructor Parameters

- `d_model`: Dimension of the model (default: 512)
  - Size of input/output vectors
- `num_heads`: Number of attention heads (default: 8)
  - Enables parallel attention mechanisms

- Derived parameter: `dimensionsOfAttentionHead = d_model / num_heads` // Size of each head	
  - **Mathematical Relationship**: `d_k = d_model / num_heads` // Size of each head

## 2. Weight Matrices and Parameters

### Learnable Parameters (Internal Weights)
```cpp
// Projection matrices for Q, K, V (Separate Projections)
Collective<t> queryWeights,    // W^Q projection matrix, transforms input to queries
              keyWeights,      // W^K projection matrix, transforms input to keys   
              valueWeights;    // W^V projection matrix, transforms input to values

Collective<t> outputWeights;   // W^O output projection matrix (Final projection matrix, final linear transformation after attention computation)

// All weights initialized randomly using normal distribution
```

### Scaling Factor
```cpp
t scaleFactor;  // 1/√d_k for numerical stability
```
- **Purpose**: Prevents attention scores from becoming too large
- **Formula**: `scaleFactor = 1.0 / √(dimensionsOfAttentionHead)`

## 3. Input Caching System

### Forward Pass Input Storage
```cpp
// Original inputs
Collective<t> X_ei_query,    // Cached input for Q projection
              X_ei_key,      // Cached input for K projection  
              X_ei_value;    // Cached input for V projection
```
- **Gradient Computation**: Required for computing gradients w.r.t. projection weights
- **Memory Trade-off**: Stores inputs to avoid recomputation during backpropagation

## 4. Computation Graph Emulation

### Intermediate Value Caching (cache variables for Backpropagation, the intermediate results to compute gradients later)
```cpp
// Manual computation graph tracking
Collective<t> masked_cached_query,             // Q after masking (store transformed Q)
              masked_cached_key,               // K after masking (store transformed K)
              masked_cached_value;             // V after masking (store transformed V)
Collective<t> cached_attention_weights;        // Softmax output (A) (stores attention scores after softmax)
Collective<t> cached_output_before_projection; // Context vectors (O) (stores attention output before final projection)
```

**Purpose**: Emulates automatic differentiation frameworks by manually storing intermediate values needed for gradient computation.

## 5. Forward Pass Implementation

### Linear Projections with Scaling
```cpp
// 1: Scale during projection (linear projections)
query = Numcy::matmul<t>(ei_query, queryWeights) * scaleFactor; // Scale Q for stability, (Q = X·W^Q / √d_k) or (query = input × queryWeights × scaleFactor) 
key = Numcy::matmul<t>(ei_key, keyWeights) * scaleFactor;	// Scale K for stability, (K = X·W^K / √d_k) or (key = input × keyWeights × scaleFactor)
value = Numcy::matmul<t>(ei_value, valueWeights); // No scaling, V doesn't need scaling. (V = X·W^V) or (value = input × valueWeights)

// 2: Scale after dot product (commented alternative)
// scores = query · key^T / sqrt(d_k);

// 2: Apply Masking

### Masking Strategy
```cpp
// Zero out padded positions in Q, K, V
for (size_t k = 0; k < value.getShape().getDimensionsOfArray().getNumberOfInnerArrays(); k++) {
    if (mask[k] == 0) {
        for (size_t l = 0; l < value.getShape().getNumberOfColumns(); l++) {
            query[k*value.getShape().getNumberOfColumns() + l] = 0;
            key[k*value.getShape().getNumberOfColumns() + l] = 0;
            value[k*value.getShape().getNumberOfColumns() + l] = 0;
        }
    }
}
```

### Hence the key Features are:

- Implements scaled dot-product attention
- Properly handles padding masks
- Stores intermediate values for backpropagation
- Includes numerical stability measures

### Attention Computation Pipeline

```cpp
// 1. Compute attention scores 
scores = Numcy::matmul<t>(query, Numcy::transpose(key)); // scores = Q·K^T
// Scale after dot product (commented alternative)
// scores = query · key^T / sqrt(d_k);

// 2. Apply attention mask (sets padded positions to -inf). This ensures they become 0 after softmax
ADHOC_IMPLEMENTATION_OF_MASK_QUERY(scores, mask, false); // Apply mask to scores
ADHOC_IMPLEMENTATION_OF_MASK_KEY(scores, mask, false);   // Apply mask to scores

// 3. Apply softmax to get attention weights
attention_weights = softmax<t>(scores);

// 4. Compute context vectors (apply attention to values)
output = Numcy::matmul<t>(attention_weights, value); // output = attention_weights × value

// 5. Final projection. Transform output through final linear layer
output = Numcy::matmul<t>(output, outputWeights); // outputWeights is a.k.a W^O
```

### 6. Backward Pass Implementation

How Backward Pass Works?
The backward pass computes gradients by working backwards through the forward pass operations

### Gradient Computation Chain (the gradient flow or the gradient flow steps)

#### The backward pass follows the chain rule through these steps

#### Step 1: Computes gradients for Output Projection
```cpp
// dL/dW^O = O^T * dL/dY or gradient_output_weights = cached_output^T × incoming_gradient
Collective<t> gradient_output_weights = Numcy::matmul<t>(Numcy::transpose(cached_output_before_projection), incoming_gradient); 
```

#### Step 2: Compute gradient for Context Vector (Attention Output Gradient) 
```cpp
// dL/dO = dL/dY * W^O^T or gradient_attention_output = incoming_gradient × outputWeights^T
Collective<t> gradient_attention_output = Numcy::matmul<t>(incoming_gradient, Numcy::transpose(this->outputWeights));
```

#### Step 3: Computes gradients for Attention Weights
```cpp
// dL/dA = dL/dO * V^T or gradient_attention_weights = gradient_attention_output × cached_value^T
Collective<t> gradient_attention_weights = Numcy::matmul<t>(gradient_attention_output, Numcy::transpose(this->masked_cached_value));
```

#### Step 4: Computes gradients for Value
```cpp
// dL/dV = A^T * dL/dO or gradient_value = cached_attention_weights^T × gradient_attention_output
Collective<t> gradient_value = Numcy::matmul<t>(Numcy::transpose(this->cached_attention_weights), gradient_attention_output);
```

#### Step 5: Softmax Backward
```cpp
// dL/dS = softmax_backward(dL/dA, A)
Collective<t> gradient_attention_scores = softmax_backward(gradient_attention_weights, this->cached_attention_weights);
```

#### Step 6-7: Computes gradients for Query and Key
```cpp
// dL/dK = (dL/dS)^T * Q * scaleFactor or gradient_key = gradient_attention_scores^T × cached_query × scaleFactor
Collective<t> gradient_key = Numcy::matmul<t>(Numcy::transpose(gradient_attention_scores), this->masked_cached_query);
gradient_key = gradient_key * scaleFactor;

// dL/dQ = (dL/dS)^T * K * scaleFactor or gradient_query = gradient_attention_scores^T × cached_key × scaleFactor
Collective<t> gradient_query = Numcy::matmul<t>(Numcy::transpose(gradient_attention_scores), this->masked_cached_key);
gradient_query = gradient_query * scaleFactor;
```

#### Step 8-10: Computes gradients for respective projection Weight Matrices (W^V, W^K, W^Q)
```cpp
// dL/dW^Q = X^T * dL/dQ or gradient_query_weights = original_input^T × gradient_query
Collective<t> gradient_query_weights = Numcy::matmul<t>(Numcy::transpose(this->X_ei_query), gradient_query);

// dL/dW^K = X^T * dL/dK or gradient_key_weights = original_input^T × gradient_key
Collective<t> gradient_key_weights = Numcy::matmul<t>(Numcy::transpose(this->X_ei_key), gradient_key);

// dL/dW^V = X^T * dL/dV or gradient_value_weights = original_input^T × gradient_value
Collective<t> gradient_value_weights = Numcy::matmul<t>(Numcy::transpose(this->X_ei_value), gradient_value);
```

### (Gradient Flow) Weight Updates Using Learning Rate 
```cpp
// Apply learning rate and update weights
gradient_query_weights = gradient_query_weights * learning_rate;
this->queryWeights = this->queryWeights - gradient_query_weights;

gradient_key_weights = gradient_key_weights * learning_rate;
this->keyWeights = this->keyWeights - gradient_key_weights;

gradient_value_weights = gradient_value_weights * learning_rate;
this->valueWeights = this->valueWeights - gradient_value_weights;
```

### Input Gradient Computation
```cpp
// Gradients flow back through three projection paths
Collective<t> input_gradient_from_query = Numcy::matmul(gradient_query, Numcy::transpose(this->queryWeights));
Collective<t> input_gradient_from_key = Numcy::matmul(gradient_key, Numcy::transpose(this->keyWeights));
Collective<t> input_gradient_from_value = Numcy::matmul(gradient_value, Numcy::transpose(this->valueWeights));

// Sum all contributions
input_gradient = input_gradient_from_query + input_gradient_from_key + input_gradient_from_value;
```

### Constructor Implementations

#### Default Constructor
```cpp
Attention(void) : 
    dimensionsOfAttentionHead(floor((t)(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER/DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER))),
    dimensionsOfTheModel(DEFAULT_DIMENTIONS_OF_THE_TRANSFORMER_MODEL_HYPERPARAMETER),
    numberOfAttentionHeads(DEFAULT_NUMBER_OF_ATTENTION_HEADS_HYPERPARAMETER)
{
    scaleFactor = (1.0 / std::sqrt(dimensionsOfAttentionHead));
}
```

### Parameterized Constructor
```cpp
Attention(cc_tokenizer::string_character_traits<char>::size_type d_model, 
         cc_tokenizer::string_character_traits<char>::size_type num_heads) :
    dimensionsOfAttentionHead(floor((t)(d_model/num_heads))),
    dimensionsOfTheModel(d_model),
    numberOfAttentionHeads(num_heads)
{
    DIMENSIONS dim = DIMENSIONS{d_model, d_model, NULL, NULL};
    
    try {
        queryWeights = Numcy::Random::randn<t>(dim);
        keyWeights = Numcy::Random::randn<t>(dim);
        valueWeights = Numcy::Random::randn<t>(dim);
        outputWeights = Numcy::Random::randn<t>(dim);
    } catch (ala_exception& e) {
        throw ala_exception(cc_tokenizer::String<char>("Attention::Attention() -> ") + e.what());
    }
    
    scaleFactor = (1.0 / std::sqrt(dimensionsOfAttentionHead));
}
```

