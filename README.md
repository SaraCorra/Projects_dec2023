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




