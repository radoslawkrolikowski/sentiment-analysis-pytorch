# Sentiment Analysis with PyTorch

The repository will walk you through the process of building a complete Sentiment Analysis model, which will be able to predict a polarity of given review (whether the expressed opinion is positive or negative). The dataset on which the model is going to be trained is popular IMDb movie reviews dataset.

### Table of contents

* [Data preprocessing](https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/1_data_processing.ipynb) or view the notebook using nbviewer here: [data preprocessing nbviewer](https://nbviewer.jupyter.org/github/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/1_data_processing.ipynb)
 	
    The first notebook covers data loading from the raw dataset, feature extraction and analysis, also text preprocessing and train/val/test sets preparation.


* [Vocabulary and batch iterator](https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/2_vocabulary.ipynb) or view the notebook using nbviewer here: [vocabulary and batch iterator nbviewer](https://nbviewer.jupyter.org/github/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/2_vocabulary.ipynb)
	
    The second tutorial contains instructions on how to set up the vocabulary object that will be responsible for the following tasks:
	* Creating dataset's vocabulary.
	* Filtering dataset in terms of the rare words occurrence and sentences lengths.
	* Mapping words to their numerical representation (word2index) and reverse (index2word).
	* Enabling the use of pre-trained word vectors.

	Furthermore, we will build the BatchIterator class that could be used for:
	* Sorting dataset examples.
	* Generating batches.
	* Sequence padding.
	* Enabling BatchIterator instance to iterate through all batches.
 

* [biGRU model](https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/3_biGRU.ipynb) or view the notebook using nbviewer here: [biGRU model nbviewer](https://nbviewer.jupyter.org/github/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/3_biGRU.ipynb)
	
    In the third notebook, the bidirectional Gated Recurrent Unit model will be built. In our neural network we will implement and use the following architectures and techniques: bidirectional GRU, stacked (multi-layer) GRU, dropout/spatial dropout, max-pooling, avg-pooling. The hyperparameters fine-tuning process will be presented. After choosing the proper parameters set, we will train our model and determine the generalization error. 


* [biGRU with additional features](https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/4_biGRU_with_additional_features.ipynb) or view the notebook using nbviewer here: [biGRU with additional features nbviewer](https://nbviewer.jupyter.org/github/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/4_biGRU_with_additional_features.ipynb)

    In this notebook, we will implement the bidirectional Gated Recurrent Unit model that uses features extracted in the first tutorial.


* [biGRU with Glove vectors](https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/5_biGRU_with_Glove_vectors.ipynb) or view the notebook using nbviewer here: [biGRU with Glove vectors nbviewer](https://nbviewer.jupyter.org/github/radoslawkrolikowski/sentiment-analysis-pytorch/blob/master/4_biGRU_with_Glove_vectors.ipynb)

    This notebook covers the implementation of the bidirectional Gated Recurrent Unit model, which uses pre-trained Glove word embeddings.


### Dataset

Dataset is available under the following link:
<http://ai.stanford.edu/~amaas/data/sentiment/>

Unpack the downloaded *tar.gz* file using:

`tar -xzf acllmdb.tar.gz`

Rearrange the data to the following structure:

    dataset
      ├── test
      │     ├── positive
      │     ├── negative
      ├── train
            ├── positive
            └── negative


### Requirements

1. Create a virtual environment (conda, virtualenv etc.).

	`conda create -n <env_name> python=3.7`

2. Activate your environment.

	`conda activate <env_name>`

3. Create a new kernel.

	`pip install ipykernel`

	`python -m ipykernel install --user --name <env_name>`

4. Go to the directory: `.local/share/jupyter/kernels/<env_name>` and ensure that *kernel.json* file contains the path to your environment python interpreter (can be checked by `which python` command).

  ```
  {
   "argv": [
    "home/user/anaconda3/envs/<env_name>/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
   ],
   "display_name": "<env_name>",
   "language": "python"
  }
  ```
5. Install requirements.

	`pip install -r requirements.txt `


6. Restart your environment.

	`conda deactivate`
    
	`conda activate <env_name>`


### Usage

Inside your virtual environment launch the *jupyter notebook*, and open the notebook file (with *.ipynb* extension), then change the kernel to the one created in the preceding step (<env_name>). Now you are ready. Follow me through the tutorial.


### Model Performance

Model  | Test accuracy | Validation accuracy | Training accuracy 
------------- | :---: |:---: | :---:
biGRU  | 0.880 |0.878 | 0.908
biGRU with extra features | 0.882 | 0.881| 0.898
biGRU with Glove vectors | 0.862 | 0.862| 0.842


### References

* <https://pytorch.org/docs/stable/index.html>
* <https://arxiv.org/pdf/1801.06146.pdf>
* <https://arxiv.org/pdf/1705.02364.pdf>
* <https://en.wikipedia.org/wiki/Sentiment_analysis>
* <https://monkeylearn.com/sentiment-analysis/#sentiment-analysis-use-cases-and-applications>
* <https://www.kaggle.com/praveenkotha2/end-to-end-text-processing-for-beginners>
* <https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/>
* <https://spacy.io/api/annotation#pos-tagging>
* <https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.TextBlob.sentiment>
* <https://scikit-learn.org/stable/modules/manifold.html>
* <https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76>
* <https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130>
