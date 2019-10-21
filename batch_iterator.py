import numpy as np
import torch
from vocabulary import Vocab


class BatchIterator:

    """The BatchIterator class is responsible for:
    Sorting dataset examples.
    Generating batches.
    Sequence padding.
    Enabling BatchIterator instance to iterate through all batches.

    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        If vocab_created is False, pass Pandas or numpy dataset containing in the first column input strings
        to process and target non-string variable as last column. Otherwise pass vocab.dataset object.
    batch_size: int, optional (default=None)
        The size of the batch. By default use batch_size equal to the dataset length.
    vocab_created: boolean, optional (default=True)
        Whether the vocab object is already created.
    vocab: Vocab object, optional (default=None)
        Use if vocab_created = True, pass the vocab object.
    target_col: int, optional (default=None)
        Column index refering to targets strings to process.
    word2index: dict, optional (default=None)
        Specify the word2index mapping.
    sos_token: str, optional (default='<SOS>')
        Use if vocab_created = False. Start of sentence token.
    eos_token: str, optional (default='<EOS>')
        Use if vocab_created = False. End of sentence token.
    unk_token: str, optional (default='<UNK>')
        Use if vocab_created = False. Token that represents unknown words.
    pad_token: str, optional (default='<PAD>')
        Use if vocab_created = False. Token that represents padding.
    min_word_count: float, optional (default=5)
        Use if vocab_created = False. Specify the minimum word count threshold to include a word in vocabulary
        if value > 1 was passed. If min_word_count <= 1 then keep all words whose count is greater than the
        quantile=min_word_count of the count distribution.
    max_vocab_size: int, optional (default=None)
        Use if vocab_created = False. Maximum size of the vocabulary.
    max_seq_len: float, optional (default=0.8)
        Use if vocab_created = False. Specify the maximum length of the sequence in the dataset, if
        max_seq_len > 1. If max_seq_len <= 1 then set the maximum length to value corresponding to
        quantile=max_seq_len of lengths distribution. Trimm all sequences whose lengths are greater
        than max_seq_len.
    use_pretrained_vectors: boolean, optional (default=False)
        Use if vocab_created = False. Whether to use pre-trained Glove vectors.
    glove_path: str, optional (default='Glove/')
        Use if vocab_created = False. Path to the directory that contains files with the Glove word vectors.
    glove_name: str, optional (default='glove.6B.100d.txt')
        Use if vocab_created = False. Name of the Glove word vectors file. Available pretrained vectors:
        glove.6B.50d.txt
        glove.6B.100d.txt
        glove.6B.200d.txt
        glove.6B.300d.txt
        glove.twitter.27B.50d.txt
        To use different word vectors, load their file to the vectors directory (Glove/).
    weights_file_name: str, optional (default='Glove/weights.npy')
        Use if vocab_created = False. The path and the name of the numpy file to which save weights vectors.

    Raises
    -------
    ValueError('Use min_word_count or max_vocab_size, not both!')
        If both: min_word_count and max_vocab_size are provided.
    FileNotFoundError
        If the glove file doesn't exist in the given directory.
    TypeError('Cannot convert to Tensor. Data type not recognized')
        If the data type of the sequence cannot be converted to the Tensor.

    Yields
    ------
    dict
        Dictionary that contains variables batches.

    """


    def __init__(self, dataset, batch_size=None, vocab_created=False, vocab=None, target_col=None, word2index=None,
             sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>', pad_token='<PAD>', min_word_count=5,
             max_vocab_size=None, max_seq_len=0.8, use_pretrained_vectors=False, glove_path='Glove/',
             glove_name='glove.6B.100d.txt', weights_file_name='Glove/weights.npy'):

        # Create vocabulary object
        if not vocab_created:
            self.vocab = Vocab(dataset, target_col=target_col, word2index=word2index, sos_token=sos_token, eos_token=eos_token,
                               unk_token=unk_token, pad_token=pad_token, min_word_count=min_word_count,
                               max_vocab_size=max_vocab_size, max_seq_len=max_seq_len,
                               use_pretrained_vectors=use_pretrained_vectors, glove_path=glove_path,
                               glove_name=glove_name, weights_file_name=weights_file_name)

            # Use created vocab.dataset object
            self.dataset = self.vocab.dataset

        else:
            # If vocab was created then dataset should be the vocab.dataset object
            self.dataset = dataset
            self.vocab = vocab

        self.target_col = target_col

        self.word2index = self.vocab.word2index

        # Define the batch_size
        if batch_size:
            self.batch_size = batch_size
        else:
            # Use the length of dataset as batch_size
            self.batch_size = len(self.dataset)

        self.x_lengths = np.array(self.vocab.x_lengths)

        if self.target_col:
            self.y_lengths = np.array(self.vocab.y_lengths)

        self.pad_token = self.vocab.word2index[pad_token]

        self.sort_and_batch()


    def sort_and_batch(self):
        """ Sort examples within entire dataset, then perform batching and shuffle all batches.

        """
        # Extract row indices sorted according to lengths
        if not self.target_col:
            sorted_indices = np.argsort(self.x_lengths)
        else:
            sorted_indices = np.lexsort((self.y_lengths, self.x_lengths))

        # Sort all sets
        self.sorted_dataset = self.dataset[sorted_indices[::-1]]
        self.sorted_x_lengths = np.flip(self.x_lengths[sorted_indices])

        if self.target_col:
            self.sorted_target = self.sorted_dataset[:, self.target_col]
            self.sorted_y_lengths = np.flip(self.x_lengths[sorted_indices])
        else:
            self.sorted_target = self.sorted_dataset[:, -1]

        # Initialize input, target and lengths batches
        self.input_batches = [[] for _ in range(self.sorted_dataset.shape[1]-1)]

        self.target_batches, self.x_len_batches = [], []

        self.y_len_batches = [] if self.target_col else None

        # Create batches
        for i in range(self.sorted_dataset.shape[1]-1):
            # The first column contains always sequences that should be padded.
            if i == 0:
                self.create_batches(self.sorted_dataset[:, i], self.input_batches[i], pad_token=self.pad_token)
            else:
                self.create_batches(self.sorted_dataset[:, i], self.input_batches[i])

        if self.target_col:
            self.create_batches(self.sorted_target, self.target_batches, pad_token=self.pad_token)
            self.create_batches(self.sorted_y_lengths, self.y_len_batches)
        else:
            self.create_batches(self.sorted_target, self.target_batches)

        self.create_batches(self.sorted_x_lengths, self.x_len_batches)

        # Shuffle batches
        self.indices = np.arange(len(self.input_batches[0]))
        np.random.shuffle(self.indices)

        for j in range(self.sorted_dataset.shape[1]-1):
            self.input_batches[j] = [self.input_batches[j][i] for i in self.indices]

        self.target_batches = [self.target_batches[i] for i in self.indices]
        self.x_len_batches = [self.x_len_batches[i] for i in self.indices]

        if self.target_col:
            self.y_len_batches = [self.y_len_batches[i] for i in self.indices]

        print('Batches created')


    def create_batches(self, sorted_dataset, batches, pad_token=-1):
        """ Convert each sequence to pytorch Tensor, create batches and pad them if required.

        """
        # Calculate the number of batches
        n_batches = int(len(sorted_dataset)/self.batch_size)

        # Create list of batches
        list_of_batches = np.array([sorted_dataset[i*self.batch_size:(i+1)*self.batch_size].copy()\
                                    for i in range(n_batches+1)])

        # Convert each sequence to pytorch Tensor
        for batch in list_of_batches:
            tensor_batch = []
            tensor_type = None
            for seq in batch:
                # Check seq data type and convert to Tensor
                if isinstance(seq, np.ndarray):
                    tensor = torch.LongTensor(seq)
                    tensor_type = 'int'
                elif isinstance(seq, np.integer):
                    tensor = torch.LongTensor([seq])
                    tensor_type = 'int'
                elif isinstance(seq, np.float):
                    tensor = torch.FloatTensor([seq])
                    tensor_type = 'float'
                elif isinstance(seq, int):
                    tensor = torch.LongTensor([seq])
                    tensor_type = 'int'
                elif isinstance(seq, float):
                    tensor = torch.FloatTensor([seq])
                    tensor_type = 'float'
                else:
                    raise TypeError('Cannot convert to Tensor. Data type not recognized')

                tensor_batch.append(tensor)
            if pad_token != -1:
                # Pad required sequences
                pad_batch = torch.nn.utils.rnn.pad_sequence(tensor_batch, batch_first=True)
                batches.append(pad_batch)
            else:
                if tensor_type == 'int':
                    batches.append(torch.LongTensor(tensor_batch))
                else:
                    batches.append(torch.FloatTensor(tensor_batch))


    def __iter__(self):
        """ Iterate through batches.

        """
        # Create a dictionary that holds variables batches to yield
        to_yield = {}

        # Iterate through batches
        for i in range(len(self.input_batches[0])):
            feat_list = []
            for j in range(1, len(self.input_batches)):
                feat = self.input_batches[j][i].type(torch.FloatTensor).unsqueeze(1)
                feat_list.append(feat)

            if feat_list:
                input_feat = torch.cat(feat_list, dim=1)
                to_yield['input_feat'] = input_feat

            to_yield['input_seq'] = self.input_batches[0][i]

            to_yield['target'] = self.target_batches[i]
            to_yield['x_lengths'] = self.x_len_batches[i]

            if self.target_col:
                to_yield['y_length'] = self.y_len_batches[i]


            yield to_yield


    def __len__(self):
        """ Return iterator length.

        """
        return len(self.input_batches[0])

