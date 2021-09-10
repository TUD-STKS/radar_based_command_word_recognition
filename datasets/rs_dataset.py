import os
import numpy as np
import torch

class RsDataSet(torch.utils.data.Dataset):
    ''' Data set class for the radarspeech data.
        
    Args:
        X_set list (2-d complex numpy array (float64)): list of all radargrams with varying sizes.
                Each radargram has the shape [num_frames, num_freq_steps]
        y_set list (str): list of all label strings
        y_set_categorical (1-d numpy vector (int)): numeric (i.e., categorical) labels from
                                                    [0... C-1], C = number of classes
    '''
    
    def __init__(self, training_corpus, subject_index, session_indices, transform_fn=None):
        ''' Data set constructor.
            
        Args:
            training_corpus (TrainingCorpus): the training corpus binary file containing all rs sequences from all
                                                    subjects and all sessions.
            subject_index (int): Subject number (S00X)
            session_indices list (int): list of session indices to use for the dataset.
            transform_fn (function): function that applies a transform to the sequence(s) of each sample.

        Returns:
            None.
        '''
        
        # Construct the sequence set from the training_corpus.
        self.S32_set = [] 
        self.S12_set = [] 
        self.y_set = [] 
        self.y_set_categorical = []
        self.unique_labels = []
        self.transform_fn = transform_fn
        
        if(isinstance(session_indices, list) == False):
            # convert non-iterable single integers to a single element list.
            session_indices = [session_indices]
        
        for session_index in session_indices: 
            self.S12_set = self.S12_set + training_corpus.sequences['S12'][subject_index][session_index] # concatenate lists
            self.S32_set = self.S32_set + training_corpus.sequences['S32'][subject_index][session_index] # concatenate lists
            self.y_set = self.y_set + training_corpus.labels[subject_index][session_index]
            print("Added S12 and S32, session {} to set.".format(session_index))
        
        # Convert labels to categorical.
        self.unique_labels = np.unique(self.y_set) # Note: unique_labels are sorted in alphabetical order!
                                                   # E.g.: The first corpus element "Null" does not have the numeric label 0 but 37.
        num_unique_labels = len(self.unique_labels)
        self.y_set_categorical = np.ones(self.__len__(), dtype='int')*(-1) 
                                                                           
        # Loop through all elements in the list,
        for element_index in range(0, self.__len__()):
            # and find the corresponding label from the unique labels list.
            for label_index in range(0, num_unique_labels):
                if(self.y_set[element_index] == self.unique_labels[label_index]):
                    self.y_set_categorical[element_index] = label_index
        
        # Check numeric labels.
        max_numeric_label_value = np.max(self.y_set_categorical)
        min_numeric_label_value = np.min(self.y_set_categorical)
        
        if(max_numeric_label_value > num_unique_labels):
            raise Exception("Error: there are %d unique labels, but the maximal label value is %d " % 
                            (num_unique_labels, max_numeric_label_value))
        if(min_numeric_label_value < 0):
            raise Exception("Error: the minimal label value is %d (>!= 0)" % (min_numeric_label_value))
                           
                
    def __len__(self):
        ''' Overridden function that returns the length of a radargram set.

        Args:
            None.

        Returns:
            length (int): number of sequences in the set.
        '''
        
        return len(self.y_set)
    
            
    def __getitem__(self, index):
        if(torch.is_tensor(index)):
            index = index.tolist()
        ''' Overridden function to pick a sample from the data set.
            
        Args:
            index (int): sample index from the dataset.

        Returns:
            X (2-d tensor (float64/float32)): Complex or transformed 2-d radargram.
            y (1-d tensor (int64/long)): Categorical sequence labels from [0...C-1]
        '''
        
        # convert to tensors.
        S12 = torch.from_numpy(self.S12_set[index])
        S32 = torch.from_numpy(self.S32_set[index])
        y = torch.tensor([self.y_set_categorical[index]]).long()
        
        # transform.
        if self.transform_fn is not None:
            # Returns feature-selected sequence.
            return self.transform_fn([S12, S32]), y
        else:
            # Return concatenated, complex spectra if no transform was specified.
            return torch.cat((S12,S32), 1), y


##############################################################################################

def pad_seqs_and_collate(batch):
    ''' custom collate (="zuordnen") function for the data_loader object. This function can be overridden
        when passed to the collate_fn argument when creating the data_loader object.
        The data loader fetches a list of samples, which can then be used to access the individual samples
        for their, e.g., (sequence, label) pairs and further processed (padded etc.).
        
    Args:
        batch (tuple): in this case a [batch_size, 2] tuple containing the (sequence, label) tuple for each 
                entry in the batch. Passed in by the data loader object. Content is:
                sequence: [2-d tensor (float64/float32)]
                label: [1-d tensor (int64/long)]

    Returns:
        padded_sequence_stack (3-d tensor (float32)): The padded sequences per batch 
                                        with dimensions [batch_size, sequence_lengths, input_size]
        numeric_labels (1-d tensor (int64/long)): Categorical/numeric label for the sequence. 
                                        Has length [batch_size].
        unpadded_sequence_lengths (1-d tensor (int64/long)): The sequence lengths (scalars).
                                        Has length [batch_size].
    ''' 

    sequences, numeric_labels = zip(*batch) # unzip the batch-tuple into sequences and their labels
    
    # Pytorch function for padding.
    sequences = list(sequences) # convert tuple to list

    # Get lengths of each sequence.
    unpadded_sequence_lengths = [sequence.shape[0] for sequence in sequences]
    unpadded_sequence_lengths = torch.tensor(unpadded_sequence_lengths)
    
    padded_sequence_stack = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    # Also convert the scalar tensor tuple to a contiguous pytorch tensor.
    numeric_labels = torch.cat(numeric_labels)
    
    return padded_sequence_stack, numeric_labels, unpadded_sequence_lengths