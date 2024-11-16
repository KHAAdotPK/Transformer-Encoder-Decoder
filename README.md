### Transformers encoder-decoder model implementation in C/C++

This project aims to implement `encoder-decoder` model of **Transformers** architecture. ~~However, development of the `encoder-decoder` portion is currently on hold~~.

~~___Reason for Pause___~~:
~~The implementation is waiting for the integration of **custom word embeddings** trained using the [Skip-gram](https://github.com/KHAAdotPK/skip-gram) algorithm written in C/C++. Once these **word embeddings** are available, they can be incorporated into the model~~.

___Current Progress___:
The project includes preliminary code for processing and preparing training data.
Placeholder functions exist for the **encoder** and **decoder**, awaiting the **word embeddings**.


___Next Steps___:
Word Embeddings: Develop or integrate C/C++ code for training word embeddings using the `Skip-gram` algorithm.
Encoder-Decoder Integration: Modify the **encoder** and **decoder** functions to utilize the trained **word embeddings**.
Model Training and Evaluation: Implement training routines and evaluate the model's performance on a sentiment analysis task.

___Note___:
This project prioritizes **CPU** efficiency due to resource constraints. The final model architecture will consider this limitation.

#### License
This project is governed by a license, the details of which can be located in the accompanying file named 'LICENSE.' Please refer to this file for comprehensive information.

~~Stay tuned for further updates as development resumes~~!