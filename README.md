# CS7641 Project Proposal - AITA Classifier

## Introduction

  

r/AmItheAsshole (AITA) is a community where users seek judgment on whether their actions were the “asshole” or not. Posts are classified as either negative (the Asshole) or positive (not the Asshole).

This project aligns with stance detection, which involves classifying text as expressing favor, opposition, or neutrality ([cite](https://dl.acm.org/doi/abs/10.1145/3369026?casa_token=wvtiQ-iP5sUAAAAA:aDJoFE18kV2vTxlaDlt-XHiDZxudL1lAIqdbehLSGbTY7fq4XbgdpXbQSuXcCE8S6tscfAAP0VAhHQ)) Stance detection has been studied in social media platforms like Twitter ([cite](https://dl.acm.org/doi/10.1145/3003433)). Additionally, morality classification ([cite](https://ieeexplore.ieee.org/abstract/document/9240031?casa_token=BiQfI7KeFuIAAAAA:dfSkwIxa3wLCHsdlb26X3oHtaReNk7u7U2wPoMq3NkuYCwoeuSUlUDLN6AByOdHxwrV8VLQEMA)), which involves assessing the ethical or moral stance of a text, is closely related to this work.

Our project falls in a similar category, which aims to build a morality classifier and generate explanations for why a post was classified as "the Asshole" or not. 

## Dataset

We have two datasets based on scraped contents of Reddit.

Dataset:

1.  [Reddit-AITA-2018-to-2022](https://huggingface.co/datasets/MattBoraske/Reddit-AITA-2018-to-2022/viewer/default/train?sort%5Bcolumn%5D=toxicity_label&sort%5Bdirection%5D=desc&p=2&row=9369)
    
2.  [AITA-2018-2019](https://github.com/iterative/aita_dataset) 
    

Relevant Features:

1.  Title \[text\]
    
2.  Body \[text\]
    
3.  Verdict \[binary\]
    

  

## Problem Definition

Problem: Develop ML model to classify posts from the r/AmItheAhole (AITA) subreddit, determining whether the author is considered "the A\*hole" or "Not the A\*hole" based on the post content. Generate text explaining the classification.

Motivation: This project aims to:

1.  Act as an automated tone checker for writing
    
2.  Explain why and how to adjust your writing to meet a tone
    

We are essentially making a “Grammarly” but for writing tone.

  

## Methods

### 3+ Data Preprocessing Methods Identified

1.  Text Cleaning: Removing special characters, URLs, excessive whitespace, and non-relevant data ensures the model processes only pertinent textual information, reducing noise.
    
2.  Data Labeling: Data in the HuggingFace dataset is unlabeled. We will use basic majority counts of keywords ‘nta’/‘yta’ found in comments to generate these labels.
    
3.  Data Dimensionality Reduction: Since we are only interested in title, body, comments, and label, we will discard all other columns from both datasets.
    

  

### 3+ ML Algorithms/Models Identified

1\. BERT 

BERT is a widely supported model with numerous pre-trained variants, proven to perform exceptionally well in text classification tasks. [(cite)](https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b)

2\. Sentence Transformers (all-MiniLM-L12-v2)  
The [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) model is a lightweight, sentence-transformer variant optimized for sentence embeddings, making it suitable for text classification tasks and ideal for clustering and similarity tasks as well.

3\. Large Language Models (LLMs) under 8B Parameters  
LLMs like [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Mistral-7B](http://mistralai/Mistral-7B-v0.1) are designed to leverage extensive pre-training data, enhancing their ability to generate coherent and contextually appropriate responses. These models will be used for both classification and explanation generation, shifting from a simple prediction model to one that provides rationale behind decisions, enhancing interpretability.

### Supervised Methods

*   Binary Text Classification: Train models in distinguishing between binary outcomes.
    
*   Text Generation Using Next Token Prediction: Design a prompt for the LLM for the text classification and explanation generation tasks. Do instruction fine-tuning on the dataset to improve accuracy.
    

### Unsupervised Methods:

*   Clustering Techniques: K-means or hierarchical clustering, for grouping similar posts or comments.
    
*   Dimensionality Reduction: Using algorithms like PCA (Principal Component Analysis) or t-SNE for visualizing high-dimensional text embeddings.
    
*   Topic Modeling: Techniques like LDA (Latent Dirichlet Allocation) to uncover hidden topics within the datasets, providing insights into the thematic structure of the text.
    

## (Potential) Results and Discussion 

We will employ standard performance metrics for binary classifier including Precision, Recall, and F1 score, along with confusion matrix for easier interpretation. When comparing performance between models, we may employ ROC-AUC score as well. We aim to achieve Precision, Recall, and F1 scores above 0.8.

  

Our first goal is to develop a binary classifier that given a post, accurately predict whether the author is considered "the A\*hole" or "Not the A\*hole". Secondly, we aim to make the model interpretable by identifying the words and phrases that contribute the most to the each prediction.

  
  
  

## References (All)

*   BERT Paper:
    

[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

*   Sentiment Analysis Paper/Transformers (AIAYN)
    

[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

*   Stance Detection: [https://dl.acm.org/doi/abs/10.1145/3369026?casa\_token=wvtiQ-iP5sUAAAAA:aDJoFE18kV2vTxlaDlt-XHiDZxudL1lAIqdbehLSGbTY7fq4XbgdpXbQSuXcCE8S6tscfAAP0VAhHQ](https://dl.acm.org/doi/abs/10.1145/3369026?casa_token=wvtiQ-iP5sUAAAAA:aDJoFE18kV2vTxlaDlt-XHiDZxudL1lAIqdbehLSGbTY7fq4XbgdpXbQSuXcCE8S6tscfAAP0VAhHQ)
    
*   Stance In Tweets:
    
*   [https://dl.acm.org/doi/10.1145/3003433](https://dl.acm.org/doi/10.1145/3003433)
    
*   Morality Classification: [https://ieeexplore.ieee.org/abstract/document/9240031?casa\_token=BiQfI7KeFuIAAAAA:dfSkwIxa3wLCHsdlb26X3oHtaReNk7u7U2wPoMq3NkuYCwoeuSUlUDLN6AByOdHxwrV8VLQEMA](https://ieeexplore.ieee.org/abstract/document/9240031?casa_token=BiQfI7KeFuIAAAAA:dfSkwIxa3wLCHsdlb26X3oHtaReNk7u7U2wPoMq3NkuYCwoeuSUlUDLN6AByOdHxwrV8VLQEMA) 
    
*   Morality Foundations Dictionary: [https://www.researchgate.net/publication/342137805\_The\_Extended\_Moral\_Foundations\_Dictionary\_eMFD\_Development\_and\_Applications\_of\_a\_Crowd-Sourced\_Approach\_to\_Extracting\_Moral\_Intuitions\_from\_Text](https://www.researchgate.net/publication/342137805_The_Extended_Moral_Foundations_Dictionary_eMFD_Development_and_Applications_of_a_Crowd-Sourced_Approach_to_Extracting_Moral_Intuitions_from_Text)
    
*   Text Classification using BERT: [https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b](https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b)
    
*   Sentence-Transformers Model [https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
