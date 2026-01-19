/*
    MLM.md
    Written by, Sohail Qayum Malik
 */

`"Readers should be aware that this article represents an ongoing project. The information and code contained herein are preliminary and will be expanded upon in future revisions."`

## Masked Language Model

BERT uses Bidirectional Attention (every word can see every other word in the sentence simultaneously) to capture dependencies between words in a sentence. This is why BERT is so much better at understanding the "context" of a word. 
- **The MLM Task:**
    1. You take a full sentence: [I, love, programming]
    2. You randomly replace 15% of the words with a [MASK] token: [I, [MASK], programming]
    3. You ask the Encoder to predict what the hidden word was.

For a **BERT** like system you would use the same encoder layer (for example, a sequence to sequence Transformer model) but different decoder (a linear layer of **MLM** Head) to predict the hidden word (In order to build a BERT from scratch, you need to add the MLM Head on top of your current encoder output. This is the part that translates "internal numbers" into "English words.")

**The MLM Head:**

The MLM task is like a puzzle where you have to fill in the blanks. 

**The Setup:**
- **Encoder Output:** A [**batch_size**, **sequence_length**, **d_model**] matrix for example [25000, 7, 8], where 25000 is the numbers of lines in the batch, 7 is the number of tokens in each line, and 8 is the dimension of the embedding.
- **Your Vocabulary Size:** Let's say you have **28** words in your dictionary.
- **The Weight Matrix:** You need a new matrix (lets call it $W_{MLM}$) of size [**d_model**, **vocab_size**] e.g. [8, 28].

1. **The Bridge:** From Encoder Output to Word Prediction
Your encoder output is a tensor of shape [batch_size, sequence_length, d_model] (for example, [25000, 7, 8], 7 tokens wide, 8 numbers deep; where 25000 is the numbers of lines in the batch). To predict a masked word, you need to "project" that d_model vector back into the size of your Vocabulary (e.g. [sequence_length, d_model] x $W_{MLM}$ -> [sequence_length, vocab_size]).

**The MLM Head Logic (The Final Layer):**

1. **Linear Transformation:** You multiply the encoder output by a weight matrix $W_{MLM}$ of size [d_model, vocab_size] (take the [MASK] token as input -> [1, d_model] which is the first token in the sequence, and project it into the size of your Vocabulary -> [1, vocab_size]).

       $[1, d_model] x W_{MLM}$ -> [1, vocab_size] , when $W_{MLM}$ is [d_model, vocab_size]

2. **Softmax:** You turn those raw scores (logits of the [MASK] token) into probabilities (0 to 1) (use a Softmax to get the probability of each word in the Vocabulary).

    $P_(word_i)$ = $e^{logits(word_i)} / \sum_{j=1}^{vocab_size} e^{logits(word_j)}$

    **-OR-**

    $P_(word_i)$ = $e^{scores_i} / \sum_{j=1}^{vocab_size} e^{scores_j}$

    **-OR-**

    $P_(word_i)$ = softmax($scores_i$)
    
3. **The Result:** A probability for every word in your dictionary. The word with the highest probability is your "prediction" for the [MASK].

**The Cross-Entropy Loss (The "Correction" Part):** 

This is how the model "learns" from its mistakes.

**Steps Involved:**

1. **Forward Pass:**
    - The model processes the input sequence and generates a prediction for each token.
    - The prediction is a probability distribution over the vocabulary.
    - The model compares its prediction with the actual target (ground truth) for each token.
    - The comparison is done using the Cross-Entropy Loss function.

2. **Backward Pass:**
    - The loss is backpropagated through the model.
    - The gradients are computed for each parameter in the model.
    - The gradients are used to update the model's parameters.

**Forward + Backward Pass:** What does it all mean?  

- **Look at the target:** We know the hidden word was actually the word with highest probability.
- **Look at the prediction:** The Forward Pass gave target word a 10% probability.
- **Calculate the "Pain" (Loss):** The loss is $-\log(0.10)$.
    - If the probability is low (0.10), the loss is high (2.3).
    - If the probability is high (0.90), the loss is low (0.1).
- **Backpropagate:** Use that "Pain" value to slightly nudge those 8 numbers in your weight matrix so that next time, the probability for the target word is higher.

**Implementing the "Masking" in C++:** Here is a simplified way to structure BERT Masking 
Strategy in C++ training loop
```C++
// RODO,
// Logic for C++ Training Loop

for (int i = 0; i < sequence_length; ++i) {
    if (should_mask(0.15)) { // 15% probability
        // 1. Store the original token ID as the 'Target' (Label)
        int original_word = input_sequence[i];
        
        // 2. Replace the word in the input with the [MASK] ID
        input_sequence[i] = VOCAB_MASK_ID; 
        
        // 3. After the Forward Pass, calculate loss ONLY on these positions
        // Loss = CrossEntropy(prediction_at_i, original_word)
    }
}
```


