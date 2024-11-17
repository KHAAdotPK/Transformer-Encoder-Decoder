 Drawing from Jay Alammar’s **Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)**.
---

- **"**The word, however, needs to be represented by a vector. To transform a word into a vector, we turn to the class of methods called “word embedding” algorithms.**"**
    
    - The above statement refers to the process of converting words into high-dimensional        vectors using algorithms like Word2Vec. Word embeddings are a core element of natural language processing because they represent words in a way that captures semantic relationships and contextual similarity. For example, words like "sunshine," "kisses," "cheeks," and "birds" are each represented as a vector, a list of real numbers that encodes aspects of each word's meaning and relationships to other words.

```
sunshine -0.837496 -1.055320 1.098407 -0.875249 0.951608 -0.904695 2.015359 -0.769061 1.338858 0.967960 -0.858001 0.013812 -0.026567 1.422336 1.577360 2.043583 -1.098003 -0.968427 1.265099 0.584877 0.250852 0.062136 0.363300 1.104700 -0.353904 -1.276656 0.194506 -0.526952 -0.878275 -0.327096 -0.460842 -0.103160 -0.356279 0.167415 -2.428486 -0.570614 -0.768021 -0.166090 -0.006392 -1.441865 -0.007248 -0.517012 -0.634928 0.857430 -1.004510 -1.277651 0.081099 -0.759937 0.279635 0.040915

kisses -0.561253 0.087849 0.540067 1.773597 -0.199710 -1.311770 -0.399599 2.156286 0.114282 -0.543003 0.130212 0.967182 -0.717343 0.884294 0.564800 -2.326968 -0.665600 1.005081 0.161274 -0.543652 -1.101190 -0.217401 -0.381640 0.559697 0.232606 0.120287 0.308728 0.245665 -0.347646 -1.059290 0.576970 0.177871 -2.125266 -0.167447 2.005500 -0.493680 0.313577 -0.559682 0.159446 -0.208280 0.780383 0.985279 -0.694880 -0.674929 1.155425 -0.218686 -0.768610 -0.438125 -1.385112 -0.991019

cheeks -0.909332 -0.460400 -0.079960 -0.089470 -0.693628 1.224485 0.874309 0.552563 0.269663 -0.967204 0.773489 0.986966 0.416275 -0.476604 1.656933 -1.764315 0.822849 -0.267776 1.257140 -0.514210 -2.350793 0.810076 -0.102020 0.101556 -1.318519 0.892773 1.506314 -1.819429 -0.237838 0.499354 0.544531 -0.693131 -0.109434 -0.403910 -1.663165 0.975767 1.947294 0.153396 -0.048958 1.049197 -0.987959 -0.299658 1.529233 -0.861486 -1.079875 0.072373 1.316198 -1.538929 -0.538378 -0.948837

birds -0.467162 -0.036999 -0.040974 0.019772 0.885775 0.527786 1.059645 1.215737 -0.587111 -0.251593 1.273645 0.139854 -0.948526 -0.732949 -0.197380 -0.882619 2.282855 -0.004618 -1.949643 -0.799179 -0.618504 -0.258405 0.879007 0.181021 0.936570 -1.048642 -0.138906 -0.544141 -0.359967 -1.284610 0.503619 0.693182 -1.666434 -0.480470 1.760978 -0.527589 -1.091076 -0.840732 -0.062099 -0.246962 -1.595210 0.522291 -0.419830 -1.111785 -0.497824 0.545630 -1.536440 0.622187 -0.168387 1.315373
```

- **"**By design, a RNN takes two inputs at each time step: an input (in the case of the encoder, one word from the input sentence), and a hidden state.**"** 

    - The hidden state represents the output of the hidden layer in an RNN at a specific time step. It encapsulates information from the input sequence up to that point and is passed along to subsequent time steps. Hence, RNNs process input sequences one step at a time, and at the final step, the hidden state is the result of sequential updates applied at each time step, **influenced by one word (or input unit) at a time from the input sequence and the hidden states from earlier steps**."

    - In a sentence like "The cat sat on the...", the hidden state at the word "on" includes the cumulative **context vector** from "The cat sat on".

- **"**The **context vector** turned out to be a bottleneck for these types of models.**"** 

    - **The Word (or Token) as Context**.
        These fixed-size embeddings can fail to capture the full complexity of a word's meaning, especially in context. For instance: The word "bank" in "river bank" vs. "financial bank" may be mapped to the same vector, leading to loss of contextual meaning.

    - **Bottleneck from Hidden States**.
        The **hidden state** in RNNs summarizes the entire sequence up to a given time step into a fixed-size vector. For long sequences, the **hidden state** struggles to retain information about early inputs, leading to problems like the vanishing gradient problem during training.

      

