# CS7641 Project Proposal - AITA Classifier

## Introduction

`r/AmItheA*hole` (AITA) is a community where users seek judgment on whether their actions were the `A*hole` or not. Posts are classified as either negative (the `A*hole`) or positive (not the `A*hole`).

This project aligns with [stance detection](https://doi.org/10.1145/3369026), which involves classifying text as expressing favor, opposition, or neutrality. [Stance detection has been studied in social media platforms like Twitter](https://dl.acm.org/doi/10.1145/3003433). Additionally, [morality classification](https://ieeexplore.ieee.org/abstract/document/9240031?casa_token=IG_eXC9q7NkAAAAA:KCDtycMz9dtUCWgSbGHLirTB7oNwhPwApNIJWJNgyHom_rv8AnsJMLXGkDiqK72t6GZNM_ULsw), which involves assessing the ethical or moral stance of a text, is closely related to this work.

Our project falls in a similar category, which aims to build a morality classifier and generate explanations for why a post was classified as "the `A*hole`" or not.

### Dataset

We have two datasets based on scraped contents of Reddit

1. [Reddit-AITA-2018-to-2022](https://huggingface.co/datasets/MattBoraske/Reddit-AITA-2018-to-2022/viewer/default/train?sort%5Bcolumn%5D=toxicity_label&sort%5Bdirection%5D=desc&p=2&row=9369)
2. [AITA-2018-2019](https://github.com/iterative/aita_dataset)

Relevant features are:

1. `Title` \[text\]
2. `Body` \[text\]
3. `Verdict` \[binary\]

## Problem Definition

Develop ML model to classify posts from the AITA subreddit, determining whether the author is considered "the `A*hole`" or "Not the `A*hole`" based on the post content. Generate text explaining the classification.

This project aims to:

1. Act as an automated tone checker for writing
2. Explain why and how to adjust your writing to meet a tone

We are essentially making a “Grammarly” but for writing tone.

## Methods

### Data Preprocessing Preprocessing

1. Text Cleaning: Removing special characters, URLs, excessive whitespace, and non-relevant data ensures the model processes only pertinent textual information, reducing noise.
2. Data Labeling: Data in the HuggingFace dataset is unlabeled. We will use basic majority counts of keywords `nta` or `yta` found in comments to generate these labels.
3. Data Dimensionality Reduction: Since we are only interested in title, body, comments, and label, we will discard all other columns from both datasets.

### Algorithms and Models

1. BERT

    BERT is a widely supported model with numerous pre-trained variants, [proven to perform exceptionally well in text classification tasks](https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b).

2. Sentence Transformers (all-MiniLM-L12-v2)

    The [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) model is a lightweight, sentence-transformer variant optimized for sentence embeddings, making it suitable for text classification tasks and ideal for clustering and similarity tasks as well.

3. Large Language Models (LLMs) under 8B Parameters

    LLMs like [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Mistral-7B](http://mistralai/Mistral-7B-v0.1) are designed to leverage extensive pre-training data, enhancing their ability to generate coherent and contextually appropriate responses. These models will be used for both classification and explanation generation, shifting from a simple prediction model to one that provides rationale behind decisions, enhancing interpretability.

### Unsupervised Methods

* Clustering Techniques: K-means or hierarchical clustering, for grouping similar posts or comments.
* Dimensionality Reduction: Using algorithms like PCA (Principal Component Analysis) or t-SNE for visualizing high-dimensional text embeddings.
* Topic Modeling: Techniques like LDA (Latent Dirichlet Allocation) to uncover hidden topics within the datasets, providing insights into the thematic structure of the text.

### Supervised Methods

* Binary Text Classification: Train models in distinguishing between binary outcomes.
* Text Generation Using Next Token Prediction: Design a prompt for the LLM for the text classification and explanation generation tasks. Do instruction fine-tuning on the dataset to improve accuracy.

## (Potential) Results and Discussion

We will employ standard performance metrics for binary classifier including Precision, Recall, and F1 score, along with confusion matrix for easier interpretation. When comparing performance between models, we may employ ROC-AUC score as well. We aim to achieve Precision, Recall, and F1 scores above 0.8.

Our first goal is to develop a binary classifier that given a post, accurately predict whether the author is considered "the `A*hole`" or "Not the `A*hole`". Secondly, we aim to make the model interpretable by identifying the words and phrases that contribute the most to the each prediction.

## Contribution Table

| Name | Proposal Contributions |
|---|---|
| All | Idea brainstorming (3 per person) |
| All | Disscussion + finalization |
| All | References |
| Kyu Yeon | Introduction & Background |
| Lex | Problem Definition |
| Dingu/Ethan | Methods (Data preprocessing, models, methods) |
| Yuto | Potential Results & Discussion |
| Ethan | Video Recording |
| Dingu | Github Pages |
| Yuto | Gantt Chart, Contribution Table |

## Gantt Chart

[Link](https://docs.google.com/spreadsheets/d/18YNVB-EbJJxHQ7TgGCrHxtCeOkt0s-LmVG50PYV-BY0/edit?usp=sharing)

## References

* [1] D. Küçük and F. Can, “Stance Detection,” ACM Computing Surveys, vol. 53, no. 1, pp. 1–37, Feb. 2020, doi: <https://doi.org/10.1145/3369026>.
* [2] S. M. Mohammad, P. Sobhani, and S. Kiritchenko, “Stance and Sentiment in Tweets,” ACM Transactions on Internet Technology, vol. 17, no. 3, pp. 1–23, Jul. 2017, doi: <https://doi.org/10.1145/3003433>.
* [3] M. C. Pavan et al., "Morality Classification in Natural Language Text," in IEEE Transactions on Affective Computing, vol. 14, no. 1, pp. 857-863, 1 Jan.-March 2023, doi: 10.1109/TAFFC.2020.3034050.
* [4] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” arXiv.org, Oct. 11, 2018. <https://arxiv.org/abs/1810.04805>
* [5] A. Vaswani et al., “Attention Is All You Need,” arXiv.org, Jun. 12, 2017. <https://arxiv.org/abs/1706.03762>
* [6] K. Pham, “Text Classification with BERT,” Medium, May 09, 2023. <https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b>
* [7] “sentence-transformers/all-MiniLM-L12-v2 · Hugging Face,” huggingface.co, Jun. 08, 2023. <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>
‌

