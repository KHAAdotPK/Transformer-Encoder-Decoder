# Encoder-Decoder Transformer Model Implementation

This repository contains an implementation of the **Transformer encoder-decoder model** from scratch in C++. The goal is to build a sequence-to-sequence model that leverages pre-trained word embeddings (from Skip-gram and CBOW) and incorporates the core components of the Transformer architecture.

---

## **Progress So Far**

### **1. Pre-trained Word Embeddings**
- Successfully implemented **Skip-gram** and **CBOW** models from scratch in C++.
- Trained word embeddings on a corpus and saved them for use in the Transformer model.
- The embeddings are used to represent input sequences in the encoder.

### **2. Input and Target Sequence Handling**
- Implemented functionality to build **input sequences** using pre-trained embeddings.
- Implemented functionality to build **target sequences** using token indices from the vocabulary.
- Both sequences are processed in batches, with support for dynamic batch sizes.

### **3. Positional Encoding**
- Implemented **sinusoidal positional encoding** to provide the model with information about the order of tokens.
- Positional encodings are added to the input embeddings before being passed to the encoder.

### **4. Encoder Implementation (In Progress)**
- Defined the structure for the **encoder**, including multi-head self-attention and feed-forward layers.
- Implemented **self-attention masks** to prevent attending to future tokens.
- Work in progress: Integrating residual connections and layer normalization.

### **5. Training Loop**
- Set up a basic training loop to process input and target sequences.
- Implemented functionality to handle multiple epochs and batches.
- Work in progress: Integrating the encoder and decoder into the training loop.

---

## **Future Implementation Goals**

### **1. Complete the Encoder**
- Finish implementing the **multi-head self-attention mechanism**.
- Add **residual connections** and **layer normalization** after each sub-layer.
- Implement the **feed-forward network** (FFN) with ReLU activation.

### **2. Implement the Decoder**
- Build the **decoder** with:
  - **Self-attention** (with masking to prevent attending to future tokens).
  - **Cross-attention** to attend to the encoder's output.
  - **Feed-forward network** (FFN).
- Add residual connections and layer normalization after each sub-layer.

### **3. Combine Encoder and Decoder**
- Pass the encoder's output to the decoder.
- Ensure the dimensions of all inputs and outputs match.

### **4. Loss Function and Optimization**
- Implement a **cross-entropy loss function** for sequence-to-sequence tasks.
- Add **gradient clipping** and **learning rate scheduling** for stable training.

### **5. Testing and Debugging**
- Test the model on simple tasks (e.g., copying or reversing a sequence) to validate the implementation.
- Gradually scale up to more complex tasks (e.g., machine translation).

### **6. Performance Optimization**
- Optimize memory usage and computation for large datasets.
- Add support for GPU acceleration (if applicable).

---

## **Code Structure**
```
Implementation/
├── lib/
│ ├── ala_wxcwption/
│ ├── allocator/
│ ├── argsv-cpp/
│ ├── corpus/
│ ├── csv/
│ ├── Numcy/
│ ├── pairs/
│ ├── parser/
│ ├── read_write_weights/
│ ├── String/
│ ├── sundry/
├── ML/
│ ├── NLP/
│ │ ├── transformers/
│ │ │ ├── encoder-decoder/
│ │ │ │ ├── attention.hh
│ │ │ │ ├── encoder.hh
│ │ │ │ ├── encoderlayer.hh
│ │ │ │ ├── header.hh
│ │ │ │ ├── model.hh
usage/
├── src/
│ ├── main.cpp
│ ├── main.hh
weights/
├── w1p.dat
├── w2p.dat
```

#### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.

