import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class Vocab:

    """The Vocab class is responsible for:
    Creating dataset's vocabulary.
    Filtering dataset in terms of the rare words occurrence and sentences lengths.
    Mapping words to their numerical representation (word2index) and reverse (index2word).
    Enabling the use of pre-trained word vectors.


    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        Pandas or numpy dataset containing in the first column input strings to process and target non-string
        variable as last column.
    target_col: int, optional (default=None)
        Column index refering to targets strings to process.
    word2index: dict, optional (default=None)
        Specify the word2index mapping.
    sos_token: str, optional (default='<SOS>')
        Start of sentence token.
    eos_token: str, optional (default='<EOS>')
        End of sentence token.
    unk_token: str, optional (default='<UNK>')
        Token that represents unknown words.
    pad_token: str, optional (default='<PAD>')
        Token that represents padding.
    min_word_count: float, optional (default=5)
        Specify the minimum word count threshold to include a word in vocabulary if value > 1 was passed.
        If min_word_count <= 1 then keep all words whose count is greater than the quantile=min_word_count
        of the count distribution.
    max_vocab_size: int, optional (default=None)
        Maximum size of the vocabulary.
    max_seq_len: float, optional (default=0.8)
        Specify the maximum length of the sequence in the dataset, if max_seq_len > 1. If max_seq_len <= 1 then set
        the maximum length to value corresponding to quantile=max_seq_len of lengths distribution. Trimm all
        sequences whose lengths are greater than max_seq_len.
    use_pretrained_vectors: boolean, optional (default=False)
        Whether to use pre-trained Glove vectors.
    glove_path: str, optional (default='Glove/')
        Path to the directory that contains files with the Glove word vectors.
    glove_name: str, optional (default='glove.6B.100d.txt')
        Name of the Glove word vectors file. Available pretrained vectors:
        glove.6B.50d.txt
        glove.6B.100d.txt
        glove.6B.200d.txt
        glove.6B.300d.txt
        glove.twitter.27B.50d.txt
        To use different word vectors, load their file to the vectors directory (Glove/).
    weights_file_name: str, optional (default='Glove/weights.npy')
        The path and the name of the numpy file to which save weights vectors.

    Raises
    -------
    ValueError('Use min_word_count or max_vocab_size, not both!')
        If both: min_word_count and max_vocab_size are provided.
    FileNotFoundError
        If the glove file doesn't exists in the given directory.

    """


    def __init__(self, dataset, target_col=None, word2index=None, sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>',
             pad_token='<PAD>', min_word_count=5, max_vocab_size=None, max_seq_len=0.8,
             use_pretrained_vectors=False, glove_path='Glove/', glove_name='glove.6B.100d.txt',
             weights_file_name='Glove/weights.npy'):

        # Convert pandas dataframe to numpy.ndarray
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_numpy()

        self.dataset = dataset
        self.target_col = target_col

        if self.target_col:
            self.y_lengths = []

        self.x_lengths = []
        self.word2idx_mapping = word2index

        # Define word2idx and idx2word as empty dictionaries
        if self.word2idx_mapping:
            self.word2index = self.word2idx_mapping
        else:
            self.word2index = defaultdict(dict)
            self.index2word = defaultdict(dict)

        # Instantiate special tokens
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        # Instantiate min_word_count, max_vocab_size and max_seq_len
        self.min_word_count = min_word_count
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len

        self.use_pretrained_vectors = use_pretrained_vectors

        if self.use_pretrained_vectors:
            self.glove_path = glove_path
            self.glove_name = glove_name
            self.weights_file_name = weights_file_name

        self.build_vocab()


    def build_vocab(self):
        """Build the vocabulary, filter dataset sequences and create the weights matrix if specified.

        """
        # Create a dictionary that maps words to their count
        self.word_count = self.word2count()

        # Trim the vocabulary
        # Get rid of out-of-vocabulary words from the dataset
        if self.min_word_count or self.max_vocab_size:
            self.trimVocab()
            self.trimDatasetVocab()

        # Trim sequences in terms of length
        if self.max_seq_len:
            if self.x_lengths:
                self.trimSeqLen()

            else:
                # Calculate sequences lengths
                self.x_lengths = [len(seq.split()) for seq in self.dataset[:, 0]]

                if self.target_col:
                    self.y_lengths = [len(seq.split()) for seq in self.dataset[:, self.target_col]]

                self.trimSeqLen()


        # Map each tokens to index
        if not self.word2idx_mapping:
            self.mapWord2index()

        # Crate index2word mapping
        self.index2word = {index: word for word, index in self.word2index.items()}

        # Map dataset tokens to indices
        self.mapWords2indices()

        # Create weights matrix based on Glove vectors
        if self.use_pretrained_vectors:
            self.glove_vectors()


    def word2count(self):
        """Count the number of words occurrences.

        """
        # Instantiate the Counter object
        word_count = Counter()

        # Iterate through the dataset and count tokens
        for line in self.dataset[:, 0]:
            word_count.update(line.split())

            # Include strings from target column
            if self.target_col:
                for line in self.dataset[:, self.target_col]:
                    word_count.update(line.split())

        return word_count


    def trimVocab(self):
        """Trim the vocabulary in terms of the minimum word count or the vocabulary maximum size.

        """
        # Trim the vocabulary in terms of the minimum word count
        if self.min_word_count and not self.max_vocab_size:
            # If min_word_count <= 1, use the quantile approach
            if self.min_word_count <= 1:
                # Create the list of words count
                word_stat = [count for count in self.word_count.values()]
                # Calculate the quantile of words count
                quantile = int(np.quantile(word_stat, self.min_word_count))
                print('Trimmed vocabulary using as mininum count threashold: quantile({:3.2f}) = {}'.\
                      format(self.min_word_count, quantile))
                # Filter words using quantile threshold
                self.trimmed_word_count = {word: count for word, count in self.word_count.items() if count >= quantile}
            # If min_word_count > 1 use standard approach
            else:
                # Filter words using count threshold
                self.trimmed_word_count = {word: count for word, count in self.word_count.items()\
                                   if count >= self.min_word_count}
                print('Trimmed vocabulary using as minimum count threashold: count = {:3.2f}'.format(self.min_word_count))

        # Trim the vocabulary in terms of its maximum size
        elif self.max_vocab_size and not self.min_word_count:
            self.trimmed_word_count = {word: count for word, count in self.word_count.most_common(self.max_vocab_size)}
            print('Trimmed vocabulary using maximum size of: {}'.format(self.max_vocab_size))
        else:
            raise ValueError('Use min_word_count or max_vocab_size, not both!')

        print('{}/{} tokens has been retained'.format(len(self.trimmed_word_count.keys()),
                                                     len(self.word_count.keys())))


    def trimDatasetVocab(self):
        """Get rid of rare words from the dataset sequences.

        """
        for row in range(self.dataset.shape[0]):
            trimmed_x = [word for word in self.dataset[row, 0].split() if word in self.trimmed_word_count.keys()]
            self.x_lengths.append(len(trimmed_x))
            self.dataset[row, 0] = ' '.join(trimmed_x)
        print('Trimmed input strings vocabulary')

        if self.target_col:
            for row in range(self.dataset.shape[0]):
                trimmed_y = [word for word in self.dataset[row, self.target_col].split()\
                             if word in self.trimmed_word_count.keys()]
                self.y_lengths.append(len(trimmed_y))
                self.dataset[row, self.target_col] = ' '.join(trimmed_y)
            print('Trimmed target strings vocabulary')


    def trimSeqLen(self):
        """Trim dataset sequences in terms of the length.

        """
        if self.max_seq_len <= 1:
            x_threshold = int(np.quantile(self.x_lengths, self.max_seq_len))
            if self.target_col:
                y_threshold = int(np.quantile(self.y_lengths, self.max_seq_len))
        else:
            x_threshold = self.max_seq_len
            if self.target_col:
                y_threshold =  self.max_seq_len

        if self.target_col:
            for row in range(self.dataset.shape[0]):
                x_truncated = ' '.join(self.dataset[row, 0].split()[:x_threshold])\
                if self.x_lengths[row] > x_threshold else self.dataset[row, 0]

                # Add 1 if the EOS token is going to be added to the sequence
                self.x_lengths[row] = len(x_truncated.split()) if not self.eos_token else \
                                      len(x_truncated.split()) + 1

                self.dataset[row, 0] = x_truncated

                y_truncated = ' '.join(self.dataset[row, self.target_col].split()[:y_threshold])\
                if self.y_lengths[row] > y_threshold else self.dataset[row, self.target_col]

                # Add 1 or 2 to the length to inculde special tokens
                y_length = len(y_truncated.split())
                if self.sos_token and not self.eos_token:
                    y_length = len(y_truncated.split()) + 1
                elif self.eos_token and not self.sos_token:
                    y_length = len(y_truncated.split()) + 1
                elif self.sos_token and self.eos_token:
                    y_length = len(y_truncated.split()) + 2

                self.y_lengths[row] = y_length

                self.dataset[row, self.target_col] = y_truncated

            print('Trimmed input sequences lengths to the length of: {}'.format(x_threshold))
            print('Trimmed target sequences lengths to the length of: {}'.format(y_threshold))

        else:
            for row in range(self.dataset.shape[0]):

                x_truncated = ' '.join(self.dataset[row, 0].split()[:x_threshold])\
                if self.x_lengths[row] > x_threshold else self.dataset[row, 0]

                # Add 1 if the EOS token is going to be added to the sequence
                self.x_lengths[row] = len(x_truncated.split()) if not self.eos_token else \
                                      len(x_truncated.split()) + 1

                self.dataset[row, 0] = x_truncated

            print('Trimmed input sequences lengths to the length of: {}'.format(x_threshold))


    def mapWord2index(self):
        """Populate vocabulary word2index dictionary.

        """
        # Add special tokens as first elements in word2index dictionary
        token_count = 0
        for token in [self.pad_token, self.sos_token, self.eos_token, self.unk_token]:
            if token:
                self.word2index[token] = token_count
                token_count += 1

        # If vocabulary is trimmed, use trimmed_word_count
        if self.min_word_count or self.max_vocab_size:
            for key in self.trimmed_word_count.keys():
                self.word2index[key] = token_count
                token_count += 1

        # If vocabulary is not trimmed, iterate through dataset
        else:
            for line in self.dataset.iloc[:, 0]:
                for word in line.split():
                    if word not in self.word2index.keys():
                        self.word2index[word] = token_count
                        token_count += 1
            # Include strings from target column
            if self.target_col:
                for line in self.dataset.iloc[:, self.target_col]:
                    for word in line.split():
                        if word not in self.word2index.keys():
                            self.word2index[word] = token_count
                            token_count += 1

        self.word2index.default_factory = lambda: self.word2index[self.unk_token]


    def mapWords2indices(self):
        """Iterate through the dataset to map each word to its corresponding index.
        Use special tokens if specified.

        """
        for row in range(self.dataset.shape[0]):
            words2indices = []
            for word in self.dataset[row, 0].split():
                words2indices.append(self.word2index[word])

            # Append the end of the sentence token
            if self.eos_token:
                words2indices.append(self.word2index[self.eos_token])

            self.dataset[row, 0] = np.array(words2indices)

        # Map strings from target column
        if self.target_col:
            for row in range(self.dataset.shape[0]):
                words2indices = []

                # Insert the start of the sentence token
                if self.sos_token:
                    words2indices.append(self.word2index[self.sos_token])

                for word in self.dataset[row, self.target_col].split():
                    words2indices.append(self.word2index[word])


                # Append the end of the sentence token
                if self.eos_token:
                    words2indices.append(self.word2index[self.eos_token])

                self.dataset[row, self.target_col] = np.array(words2indices)

        print('Mapped words to indices')


    def glove_vectors(self):
        """ Read glove vectors from a file, create the matrix of weights mapping vocabulary tokens to vectors.
        Save the weights matrix to the numpy file.

        """
        # Load Glove word vectors to the pandas dataframe
        try:
            gloves = pd.read_csv(self.glove_path + self.glove_name, sep=" ", quoting=3, header=None, index_col=0)
        except FileNotFoundError:
            print('File: {} not found in: {} directory'.format(self.glove_name, self.glove_path))

        # Map Glove words to vectors
        print('Start creating glove_word2vector dictionary')
        self.glove_word2vector = gloves.T.to_dict(orient='list')

        # Extract embedding dimension
        emb_dim = int(re.findall('\d+' ,self.glove_name)[-1])
        # Length of the vocabulary
        matrix_len = len(self.word2index)
        # Initialize the weights matrix
        weights_matrix = np.zeros((matrix_len, emb_dim))
        words_found = 0

        # Populate the weights matrix
        for word, index in self.word2index.items():
            try:
                weights_matrix[index] = np.array(self.glove_word2vector[word])
                words_found += 1
            except KeyError:
                # If vector wasn't found in Glove, initialize random vector
                weights_matrix[index] = np.random.normal(scale=0.6, size=(emb_dim, ))

        # Save the weights matrix into numpy file
        np.save(self.weights_file_name, weights_matrix, allow_pickle=False)

        # Delete glove_word2vector variable to free the memory
        del self.glove_word2vector

        print('Extracted {}/{} of pre-trained word vectors.'.format(words_found, matrix_len))
        print('{} vectors initialized to random numbers'.format(matrix_len - words_found))
        print('Weights vectors saved into {}'.format(self.weights_file_name))
