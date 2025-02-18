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
- Defined the structure for the **encoder**, including: 
  - **Multi-head self-attention** with masking support.
  - **Feed-forward layers** with `ReLU` activation.
  - **Layer normalization** after each sub-layer.
  - **Droupout** for regularization.
  - **Residual connections** for stable gradient flow.
- Implemented **self-attention masks** to prevent attending to future tokens(`padding tokens`).
- Work in progress: Testing attention outputs and integrating them with feed-forward layers.
### **5. Training Loop**
- Set up a basic training loop to process input and target sequences.
- Implemented functionality to handle multiple epochs and batches.
- Work in progress: Integrating the encoder and decoder into the training loop.

---

## **Future Implementation Goals**

### **1. Finalize the Encoder**
- Verify correct functionality of **multi-head self-attention**.
- Ensure residual connections and **layer normalization** are correctly applied.
- Optimize the **feed-forward network** (FFN) for computational efficiency.

### **2. Implement the Decoder**
- Build the **decoder** with:
  - **Masked self-attention** (with masking to prevent attending to future tokens).
  - **Cross-attention** to attend to the encoder's output.
  - **Feed-forward network** (FFN) with `ReLU` activation.
- Integrate residual connections and layer normalization after each sub-layer.

### **3. Integrate Encoder and Decoder**
- Pass the encoder's output to the decoder.
- Ensure attention mechanisms function correctly across both components.
- Validate that input-output dimensions remain consistent throughout processing.

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
│ ├── ala_exception/
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
│ │ │ │ ├── EncoderFeedForwardNetwork.hh
│ │ │ │ ├── encoderlayer.hh
│ │ │ │ ├── EncoderLayerNormalization.hh
│ │ │ │ ├── header.hh
│ │ │ │ ├── hyperparameters.hh
│ │ │ │ ├── model.hh
usage/
├── src/
│ ├── main.cpp
│ ├── main.hh
├── data/
│ ├── chat/
| │ ├── INPUT.txt
| │ ├── TARGET.txt
│ ├── weights/
| │ ├── w1p.dat
| │ ├── w2p.dat
```

### Build and Run this model
The batch script (`RUN.cmd`) manages both building and running the program with customizable parameters.

The script accepts the following command-line arguments:

- `verbose`: Enables verbose output during execution
- `e [number]`: Sets the number of epochs (default: 1)
- `w1 [filename]`: Specifies the path to the weights file (default: "./data/weights/w1p.dat")
- `build [verbose]`: Initiates the build process
  - Adding `verbose` enables conditional preprocessing

#### Examples

Build with verbose preprocessing:
```batch
cd usage
RUN.cmd build verbose
```

Run with custom parameters:
```batch
cd usage
RUN.cmd e 5 w1 ./custom/weights.dat verbose
```

#### Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request if you would like to improve the implementation.

#### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.

