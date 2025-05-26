# Transformer Encoder-Decoder Model — C++ Implementation

This repository contains an implementation of the **Transformer Encoder-Decoder model** from scratch in C++. The objective is to build a sequence-to-sequence model that leverages pre-trained word embeddings (from Skip-gram and CBOW) and incorporates all core components of the Transformer architecture.

> **For detailed insights into input processing, encoder behavior, and debugging output, refer to the [Debug Output Documentation](https://github.com/KHAAdotPK/Transformer-Encoder-Decoder/tree/main/Documents/summary-output-of-progress-so-far.md). This document provides a comprehensive breakdown of the input sequence, masking, positional encoding, and encoder input, along with observations and recommendations.**

---

## **Progress Overview**

### **1. Pre-trained Word Embeddings**
- Successfully implemented **Skip-gram** and **CBOW** models from scratch in C++.
- Trained embeddings on a corpus and saved them for direct use in the Transformer model.
- The encoder uses these embeddings to represent input sequences.

### **2. Input and Target Sequence Handling**
- Built functionality for preparing **input sequences** using pre-trained embeddings.
- Built functionality for preparing **target sequences** using token indices.
- Full batch processing support with configurable batch sizes.

### **3. Positional Encoding**
- Implemented **sinusoidal positional encoding** to inject token position information.
- Positional encodings are added to embeddings before feeding into the encoder.

### **4. Encoder Development (In Progress)**
- Designed the **encoder** with:
  - **Multi-head self-attention** supporting masking.
  - **Feed-forward networks** with ReLU activation.
  - **Layer normalization** after each sub-layer.
  - **Dropout** for regularization.
  - **Residual connections** for improved gradient flow.
- Implemented **self-attention masks** to prevent attention to padding tokens.
- Currently testing and integrating feed-forward layers.

### **5. Training Loop**
- Set up a basic training loop processing input and target batches.
- Supports multiple epochs.
- Ongoing work: integrating full encoder-decoder workflow.

---

## **Upcoming Milestones**

### **1. Complete the Encoder**
- Finalize multi-head self-attention mechanism.
- Confirm correct application of residuals and layer normalization.
- Optimize the feed-forward layers.

### **2. Build the Decoder**
- Implement the **decoder** featuring:
  - **Masked self-attention** (to prevent peeking ahead).
  - **Cross-attention** on encoder outputs.
  - **Feed-forward networks**.
- Add residuals, normalization, and dropout.

### **3. Connect Encoder and Decoder**
- Seamlessly pass encoder outputs to the decoder.
- Ensure dimensional consistency and correct attention behavior.

### **4. Loss Function and Optimization**
- Implement **cross-entropy loss** for training.
- Introduce **gradient clipping** and **learning rate scheduling** for stable convergence.

### **5. Validation and Debugging**
- Test on simple sequence-to-sequence tasks (e.g., copying, reversing).
- Gradually scale to more complex tasks like machine translation.

### **6. Performance Enhancements**
- Optimize memory and compute for larger datasets.
- Explore optional GPU acceleration.

---

## **Codebase Structure**
```
Implementation/
├── lib/
│   ├── ala_exception/
│   ├── allocator/
│   ├── argsv-cpp/
│   ├── corpus/
│   ├── csv/
│   ├── Numcy/
│   ├── pairs/
│   ├── parser/
│   ├── read_write_weights/
│   ├── String/
│   ├── sundry/
├── ML/
│   ├── NLP/
│   │   ├── transformers/
│   │   │   ├── encoder-decoder/
│   │   │   │   ├── attention.hh
│   │   │   │   ├── decoder.hh
│   │   │   │   ├── DecoderLayer.hh
│   │   │   │   ├── DecoderLayerList.hh
│   │   │   │   ├── encoder.hh
│   │   │   │   ├── EncoderFeedForwardNetwork.hh
│   │   │   │   ├── encoderlayer.hh
│   │   │   │   ├── EncoderLayerList.hh
│   │   │   │   ├── EncoderLayerNormalization.hh
│   │   │   │   ├── header.hh
│   │   │   │   ├── hyperparameters.hh
│   │   │   │   ├── Layer.hh
│   │   │   │   ├── model.hh
usage/
├── src/
│   ├── main.cpp
│   ├── main.hh
├── data/
│   ├── chat/
│   │   ├── INPUT.txt
│   │   ├── TARGET.txt
│   ├── weights/
│   │   ├── w1p.dat
│   │   ├── w2p.dat
```

---

## **Building and Running the Model**

Use the provided batch script (`RUN.cmd`) to build and execute the program with customizable parameters.

### **Available Command-line Arguments:**
- `verbose`: Enables verbose output.
- `e [number]`: Sets the number of epochs (default: 1).
- `w1 [filename]`: Specifies the pre-trained weights file (default: `./data/weights/w1p.dat`).
- `build [verbose]`: Initiates build; adding `verbose` triggers verbose compilation mode.

### **Examples**

**Build with verbose preprocessing:**
```batch
cd usage
RUN.cmd build verbose
```

**Run with custom parameters:**
```batch
cd usage
RUN.cmd e 5 w1 ./custom/weights.dat verbose
```

---

## **Contributing**

Contributions are welcome!  
Feel free to open an issue or submit a pull request if you'd like to improve or expand the project.

---

## **License**

This project is governed by a license, available in the accompanying `LICENSE` file.  
Please refer to it for complete licensing details.
