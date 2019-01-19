
Running Google BERT with Multilingual (104 languages) pretrained neural net locally or via Google Colab.

----------
Google BERT official page: https://github.com/google-research/bert

Keras BERT: https://github.com/CyberZHG/keras-bert

----------

I. Run BERT via Google Colab (the simplest way):
------

1. Open URL: http://colab.research.google.com/github/blade1780/bert/blob/master/BERT.ipynb
2. Menu Runtime -> Run All (or press Ctrl+F9)
3. Agree to reset all runtimes if needed
4. Wait for downloading model and all imports
5. Change input strings (**sentence**, **sentence_1** and **sentence_2**) and press Play button left side to recalculate only current cell (or press Ctrl+Enter)

If use mobile Chrome, it may be need to activate checkbox Full Version in browser settings.




II. Run BERT locally (you need GPU GTX 970 4Gb or higher):
------

1. Install TensorFlow from https://www.tensorflow.org/install (install CUDA Toolkit 9.0, cuDNN SDK 7.2 and run)

pip install tensorflow-gpu

2. Intall Keras

pip install keras

3. Install Keras BERT

pip install keras-bert

4. Clone this repository

git clone https://github.com/blade1780/bert

5. Download and extract pretrained BERT model to folder 'bert': https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip (632 Mb)

6. Navigate to 'bert' folder

cd bert

7. Run

python BERT.py

