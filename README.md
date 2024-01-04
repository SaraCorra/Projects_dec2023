# Projects_dec2023

This GitHub repository is designed to showcase three distinct projects:

1. [Named Entity Recognition](./Named%20Entity%20Recognition/)
2. [Image captioning](./Image%20Captioning/)
3. [Digit recognition](./Digit%20Recognition/)

## Named Entity Recognition 

This project, undertaken by a team of three - Bottino Manuel, Poetto Patrick, and Shaboian Goar, alongside myself - utilized the [GUM dataset](https://github.com/nluninja/nlp_datasets/tree/main/GUM). GUM, Georgetown University Multilayer corpus, offers annotated texts focusing on contemporary English. It employs the IOB2 format for named entity recognition, where words within and outside chunks are tagged with B, I, and O tags.

Here is a snippet from our training dataset:

![NER_example](Named%20Entity%20Recognition/images/ner.png)

The project compared Feedforward neural network, LSTM, BiLSTM, and fine-tuning the pretrained BERT model. LSTM captures long-term dependencies while BiLSTM processes sequences in both directions. A challenge we faced was the small training dataset (2495 examples). To tackle this, we employed data augmentation, generating new sentences while maintaining entity structure.

For robustness, we used k-fold cross-validation, partitioning the dataset into k subsets for multiple model training iterations. This combats data limitations and enhances model reliability. The BERT model, with bidirectional attention and contextual embeddings, demonstrated superior efficacy in Named Entity Recognition, showcasing higher precision, recall, and F1-score.

Here's the direct link to access the [Colab Notebook](https://colab.research.google.com/drive/1wG27yed191Gi_00GlLZFBIJHPQFCcSWm#scrollTo=L7tplwzLPWLH&uniqifier=1).

## Image captioning 

The project, conducted in collaboration with colleague Shaboian Goar, aims to develop a deep neural network for scientific figure captioning. We utilized the [SciCap dataset](https://drive.google.com/drive/folders/1gfnSxuG67eDzjFbwxVxNtAqoqxU9ch2r?usp=sharing), focusing on articles in Computer Sciences (cs) and Machine Learning (stat.ML) from 2010-2020 on [arXiv](https://arxiv.org/). 

Below, few examples from the training dataset are displayed:

<img src="Image%20Captioning/images/image_captions.png" alt="captions_example" width="65%">

Managing the large dataset prompted the use of subfolders to ease CUDA memory burden during training. The baseline model used was CNN + LSTM architecture. This allows to extract the features from the images using the convolutional network, which are then used in the LSTM decoder for word generation.
Since the dataset provides both visual and text data, it was possible to leverage both sources of data to build a comprehensive model.

The model architecture consists of two primary components: the encoder and the decoder. In the encoder, a comparison between ResNet and GoogLeNet is conducted, followed by a Rectified Linear Unit (ReLU) activation function and dropout for regularization. Moving to the decoder, word embeddings are employed to create vector representations for each word, followed by a dropout layer. Long Short-Term Memory (LSTM) units are utilized to generate words, and the process concludes with a final linear layer for classification.

The entire project utilized PyTorch for implementation.

Here's the direct link to access the [Colab Notebook](https://colab.research.google.com/drive/1gkVih9wGMpYsDLXR4Eo_EJbN-wmlYs9I#scrollTo=J-H6rGn2smMV) and the link for the  [Google drive folder](https://drive.google.com/drive/folders/1gfnSxuG67eDzjFbwxVxNtAqoqxU9ch2r?usp=drive_link) needed to run the Script.

## Digit recognition 

This exercise serves as a demonstration of the optimization process effectiveness using digit information from the well-known [MNIST dataset]( https://www.kaggle.com/datasets/hojjatk/mnist-dataset). It is a database of handwritten digits, widely employed for testing new machine learning algorithms. The primary objective here is to accurately classify the ten digits into their respective classes.

The model architecture includes two inner layers, both employing the hyperbolic tangent activation function. In the output layer, the softmax activation function facilitates classification based on the highest probability. Given the multi-classification nature of the problem, cross-entropy loss is employed.

The optimization technique involves the mini-batch stochastic gradient method with Adam diagonal scaling, implemented in a Python function. The model's weights minimizing the loss function are used for testing the dataset, resulting in a final accuracy of 95.85%.

Here's the direct link to access the [Colab Notebook](https://colab.research.google.com/drive/1TK-7WaA8z2IcNJuiF0ClE3FvFsqM4hsx#scrollTo=14fa7869).











