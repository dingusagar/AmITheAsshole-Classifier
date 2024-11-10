Based on the following requirement: 
"""
2. GitHub Repository: Reuse the GitHub repository from the proposal, and add all relevant directories, files, and code. Include a README.md file explaining all relevant directories and files using the format below.

/dir/: Description of the directory
/dir/file.txt: Description of the file
"""
... we have added a README.md file to explain each directory / files.

# Code
All code can be found in `notebooks/`.

# Images
All generated images can be found in `img/`.

# Dataset
All datasets can be found in `data/`.

# Reports
All reports can be found in `reports/`

We now explain each folder.
-----------
## `reports/`
- `midterm.md`: This is our midterm report, following the requirements of the class.
- `proposal.md`: This is our proposal report, following the requirements of the class.

## `notebooks/`
- `visualizations.ipynb`: visualizations using PCA, TSNE, GMM. We tried to see if they produced useful decision boundaries, but unfortunately not so much.
- `top2vec.kpynb`: Here we used Top2Vec to see if we could find any patterns in the topics of our classifications. Some interesting topic clusterings, but did not help much in our classification problem.
- `requirements.txt`: Requirements file.
- `readme_conda.md`: Info on how to use conda with our repo.
- `lex_classifiers.ipynb`: Tests using Logistic Regression, SVC, KMeans to see precision, recall, and f1.
- `clustering_tests/`: Directory showing some clustering examples
- `clustering_tests/kmeans.py`: Example using kmeans to cluster our data.
- `pretrained_baseline/` Directory using an example pretrained model for sentiment classification
- `pretrained_baseline/bert_pretrained_baseline.ipynb`: We wanted to see if a pretrained bert model could solve our classification problem, but not so much.
- `topic_modeling/`: Here we use BERTTopic to do some topic modeling.
- `topic_modeling/topic-modeling-berttopic.ipynb`: Using BERTTopic to see what topics are generated.

## `img/`
- There are many images here. The title should tell you what they are. 

## `data/`
- `aita_balanced_downsampled.csv`: Our dataset was initialially imbalanced, so we made it balanced via downsampling. We will do tests using this dataset; if it is not so great, we may revert back to the original dataset.
- `clean_data.py`: For cleaning our raw dataset.
- `pull_data.sh`: For downloading the dataset.