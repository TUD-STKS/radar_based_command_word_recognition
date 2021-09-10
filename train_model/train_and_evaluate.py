import torch
import collections
import sklearn.model_selection
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt

def get_allowed_spectra_and_transform_keys():
    ''' Returns the allowed spectrum keywords for this evaluation as well as the
        allowed transform options allied to the spectra.

    Args:
        None.
    Returns:
        allowed_spectrum_keywords (list(str)): list of strings for the allowed spectrum keywords.
        allowed_transform_options (list(str)): list of strings for the allowed transform options.
    '''

    allowed_spectrum_keywords = ['S12', 'S32']
    allowed_transform_options = ['mag','mag_db','phase','mag_delta','phase_delta']

    return (allowed_spectrum_keywords, allowed_transform_options)


###########################################################################

def calculate_delta_sequence(sequence):
    ''' Calculates the delta sequence of the input sequence (finite forward differences).

    Args:
        sequence (2-d tensor): input sequence

    Returns:
        delta_sequence (2-d tensor): output sequence
    '''

    num_frames, num_steps = sequence.shape
    # Allocate new sequence tensor to not modify the input sequence.
    delta_sequence = torch.empty((num_frames, num_steps), dtype=sequence.dtype)

    for frame_index in range(1, num_frames):
        delta_sequence[frame_index-1, :] = sequence[frame_index, :] - sequence[frame_index-1, :]
    delta_sequence[-1, :] = delta_sequence[-2, :] # pad the last frame.

    return delta_sequence;


###########################################################################

class ParameterizedTransformFunction():
    ''' Class to parameterize the transform function passed to the dataset object.

    Args:
        spectra (list(str)): list of strings specifying which spectra to use and in which order.
        transforms (list(str)): list of strings specifying the transforms applied to each spectrum 
            in the spectra list.
        is_normalized (bool): Specify whether to normalize every transformed spectrum. 
        freq_start_index (int): Specify the frequency start index for every spectrum.
        freq_stop_index (int): Specify the frequency stop index for every spectrum.

    ToDos:
        Potentially make is_normalized a logic vector of size len(spectra/transforms)

    '''

    def __init__(self, spectra, transforms, is_normalized, freq_start_index, freq_stop_index):
        self.spectra = spectra
        self.transforms = transforms
        self.is_normalized = is_normalized
        self.freq_start_index = freq_start_index
        self.freq_stop_index = freq_stop_index

        self.sequence_indices = []
        self.allowed_spectrum_keywords = ['S12', 'S32']
        self.allowed_transform_options = ['mag','mag_db','phase','mag_delta','phase_delta']
        self.num_pairs = 0

        if(len(self.spectra) != len(self.transforms)):
            sys.exit("Error: length of spectrum keys needs to be equal to length of transform keys.")

        # Check the entered spectrum keys.
        for keyword in self.spectra:
            is_key_valid_list = [keyword == allowed_keyword for allowed_keyword in self.allowed_spectrum_keywords]
            if not(any(is_key_valid_list)):
                sys.exit("Error: allowed keys in spectra_and_transforms are: %s. Passed: %s" % (self.allowed_spectrum_keywords, keyword))
            # Get the numeric index of the spectrum keyword. S12: 0, S32: 1
            self.sequence_indices.append([index for index, key in enumerate(self.allowed_spectrum_keywords) if (key == keyword)][0])
  
        for keyword in self.transforms:
            is_transform_valid_list = [keyword == allowed_transform for allowed_transform in self.allowed_transform_options]
            if not(any(is_transform_valid_list)):
                sys.exit("Error: allowed keys in spectra_and_transforms are: %s. Passed: %s" % (self.allowed_transform_options, keyword))
        
        self.num_pairs = len(self.spectra)


    def transform_sequences(self, sequence_list):
            ''' Transforms one or several spectra into a single feature vector.
            Args:
                spectra (array(str)): Specifies which spectra to use in the feature vector.
                transforms (array(str)): Specifies how each feature in the feature vector is transformed.
                is_normalized (bool): specify whether to normalize the sequences or not.
                freq_start_index (int): frequency index to start from.
                freq_stop_index (int): frequency index to stop at.

            Returns:
                sequence (2-d tensor): transformed, single feature vector of size [num_frames, num_features]
            '''

            num_frames, _ = sequence_list[0].shape

            # Allocate feature vector.
            feature_vector = torch.empty(num_frames, self.freq_stop_index*self.num_pairs)

            for index in range(self.num_pairs):

                # Crop the sequence.
                sequence = sequence_list[self.sequence_indices[index]][:, self.freq_start_index: self.freq_stop_index]

                # Apply basic transform(s).
                if(self.transforms[index] == 'mag_delta' or self.transforms[index] == 'phase_delta'):
                    sequence = calculate_delta_sequence(sequence)

                    if(self.transforms[index] == 'mag_delta'):
                        sequence = torch.abs(sequence)
                    else:
                        sequence = torch.angle(sequence)

                if(self.transforms[index] == 'mag'):
                    sequence = torch.abs(sequence)

                if(self.transforms[index] == 'mag_db'):
                    sequence = 20*torch.log10(torch.abs(sequence))

                if(self.transforms[index] == 'phase'):
                    sequence = torch.angle(sequence)

                # Normalize to [0,1].
                if(self.is_normalized):
                    max_value = torch.max(sequence)
                    min_value = torch.min(sequence)
                    sequence = (sequence - min_value)/(max_value - min_value)

                # if several spectra are used as input features, concatenate them along the feature axis.
                feature_vector[:, index*self.freq_stop_index:(index+1)*self.freq_stop_index] = sequence

            return feature_vector
        

###########################################################################

def evaluate_model(model, data_loader, device='cpu'):
    ''' Parse a given data loader and evaluate the accuracy for a given lstm model on it.
       
    Args:
        model (inherited from nn.Module): lstm model to be evaluated.
        data_loader (DataLoader): data loader containing the data set.
        device (str): chosen computing device.

    Returns:
        sequence_labels (1-d tensor (long/int64)): actual labels of each sequence from the data set.
        predicted_labels (1-d tensor (long/int64)): predicted labels of each sequence from the data set.
    '''
    
    model.eval()
    target_labels = torch.empty([0]).to(device)
    predicted_labels = torch.empty([0]).to(device)

    for sequence_stack, sequence_labels, unpadded_lengths in data_loader:
        
        sequence_labels = sequence_labels.to(model.device)
        
        # Forward the model.
        Y_pred = model(sequence_stack, unpadded_lengths) # [batch_size, output_size]    
        y_pred = torch.argmax(Y_pred, dim=1)
        
        # Append the individual predictions to the predicted labels tensor.
        target_labels = torch.cat((target_labels, sequence_labels), dim=0)
        predicted_labels = torch.cat((predicted_labels, y_pred), dim=0)
    
    return target_labels, predicted_labels


###########################################################################

class StratifiedKFoldWithValidation(sklearn.model_selection.StratifiedKFold):
    '''
        Very simple class to create stratified training, validation and test
        folds in a circular way, i.e., the first k-2 folds are the training
        set, the next fold k-1'th is the validation set and the remaining k'th
        fold is the test set.
    '''

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        '''
            Constructor. Identical to the constructor of the base class 
            sklearn.model_selection.StratifiedKFolds.
        '''

        # Init the base class with the passed arguments for the subclass.
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        # Create an array with enumerated split indices.
        self.split_indices = np.linspace(start=0, 
                                         stop=n_splits, 
                                         num=n_splits, 
                                         endpoint=False,
                                         dtype='int32')
        self.folds = []
        self.num_reps_train = -1
        self.num_reps_val_test = -1
        self.y = []


    ###########################################################################

    def create_train_test_val_splits(self, X, y, num_classes):
        for _, test_indices in self.split(X, y):
            self.folds.append(test_indices.astype('int32'))
    
        # Check, if all sequences are part of the splits. TODO: as unit test.
        sequence_indices = np.empty(0)
        for split_indices in self.folds:
            sequence_indices = np.concatenate((split_indices, sequence_indices), axis=0)

        sequence_indices = np.sort(sequence_indices)
        indices_sum = np.sum(X == sequence_indices)
        if(indices_sum != X.size):
            sys.exit("Error: Some sequences are not part of any fold.")

        # Calculate correct number of repetitions per class.
        self.num_reps_train = y.size*(self.n_splits-2)//(num_classes*self.n_splits)
        self.num_reps_val_test = y.size//(num_classes*self.n_splits)

        # Save the indices for error checking. TODO: remove in later version.
        self.y = y


    ###########################################################################

    def get_train_val_test_indices(self, split_index):
        '''
            Returns the training, validation and test indices for the cycled-through folds.
            Args:
                split_index (int): Specifies the split index, i.e., which folds are part
                    of the train, validation and test set. 
                    Example for n_splits = 5: split_index = 2 -> train = [4,5,1], val = [2], test = [3]
        '''
        indices = np.roll(self.split_indices, split_index)
        train_indices = np.empty(0,dtype='int32')

        # Train indices are the first k-2 folds.
        for index in indices[0:self.n_splits-2]:
            train_indices = np.concatenate((train_indices, self.folds[index]), axis=0)

        # Validation indices is the next fold k-1.
        validation_indices = self.folds[indices[self.n_splits-2]]

        # Test indices is the last fold k.
        test_indices = self.folds[indices[self.n_splits-1]]

        # Check, if each set has the correct number of elements.
        # TODO: as unit test.
        
        values, counts = np.unique(self.y[train_indices], return_counts=True)
        for count_index in counts:
            if(counts[count_index] != self.num_reps_train):
                raise ValueError("Class %d has %d counts but should have %d" % (count_index, counts[count_index]), num_reps_train)
        
        values, counts = np.unique(self.y[validation_indices], return_counts=True)
        for count_index in counts:
            if(counts[count_index] != self.num_reps_val_test):
                raise ValueError("Class %d has %d counts but should have %d" % (count_index, counts[count_index]), num_reps_train)

        values, counts = np.unique(self.y[test_indices], return_counts=True)
        for count_index in counts:
            if(counts[count_index] != self.num_reps_val_test):
                raise ValueError("Class %d has %d counts but should have %d" % (count_index, counts[count_index]), num_reps_train)

        return (train_indices, validation_indices, test_indices)


###########################################################################

class Trainer():
    """ Trainer class to train a given lstm model.
    """

    def __init__(self, num_epochs=200, device='cpu', is_verbose=False, log_fn=None):
        ''' Constructor.

        Args:
            num_epochs (int): number of epochs for training.
            device (str): selected device.
            is_verbose (bool): print progress during training on/off.
            log_fn (function(message : str, file_name : str)): function to log messages.

        Returns:
            None.
        '''

        self.num_epochs = num_epochs
        self.device = device
        self.is_verbose = is_verbose
        self.log_fn = log_fn
        self._log_file_name = "run_log_file"                   # default/dummy private log file name.
        self.full_log_file_name = self._log_file_name + ".txt" # default/dummy log file name with extension.
        training_results = collections.namedtuple('training_results', ['batch_loss_history', 
                                                                       'validation_accuracies_history',
                                                                       'train_accuracies_history',
                                                                       'num_epochs_to_train',
                                                                       'best_model'])
        self.training_results = training_results 


    def fit_model(self, model, optimizer, train_data_loader, validation_data_loader, patience=20):
        ''' Fits/trains the model parameters.
        
        Args:
            model (bool): Toggle print feedbacks on current training status on/off.
            optimizer (torch.optim): Optimizer for the network.
            train_data_loader (torch.utils.data.DataLoader): data loader for the training data.
            validation_data_loader (torch.utils.data.DataLoader): data loader for the validation data.
            patience (int): Maximal number of consecutive fails to reach the current best validation
                accuracy (early stopping).

        Returns:
            None. Results are stored in the member collection "training_results".
            
        '''
      
        criterion = torch.nn.NLLLoss() # Note: criterion is not an argument as it is fixed as the NLLLoss for this problem.
        fail_count = 0
        self.training_results.batch_loss_history = []
        self.training_results.validation_accuracies_history = []
        self.training_results.train_accuracies_history = []
        self.training_results.max_validation_accuracy = -1.0
        self.training_results.num_epochs_to_train = self.num_epochs # Number of epochs that corrspond to the highest val. accuracy
        
        for epoch in range(self.num_epochs):

            model.train() # Tells the model that it is in training mode

            # Select samples (sequences, labels) batch wise.
            batch_number = 0
            for sequence_stack, sequence_labels, unpadded_lengths in train_data_loader:
                # sequence_stack is a padded [3-d tensor] to the longest sequence.
                # sequence_labels are scalar [1-d tensor] with scalar values for each sequence.
            
                # sequence_stack = sequence_stack.to(device) # Sent to GPU in forward() of the model.
                sequence_labels = sequence_labels.to(self.device) 
                # unpadded_lengths = unpadded_lengths.to(device) # Throws error
            
                batch_number += 1

                if(model.debug_mode):
                    msg = "Seq. stack type: %s, label type: %s, lengths type %s" % (type(sequence_stack),
                                                                                    type(sequence_labels),
                                                                                    type(unpadded_lengths)
                                                                                    )
                    log_fn(msg, self.full_log_file_name)
            
                if(model.debug_mode):
                    log_fn("Padded sequence stack: %s" %(sequence_stack.shape), 
                           self.full_log_file_name)

                # Clear the current parameter's gradients.
                optimizer.zero_grad()

                # Forward the model.
                Y_pred = model(sequence_stack, unpadded_lengths) # [batch_size, output_size]
            
                # Calculate the loss for sequence to label. Only the last output frame is
                # considered, after the full sequence has been passed to the LSTM.           
                loss = criterion(Y_pred, sequence_labels) # [batch_size, 1]  
            
                # Generate gradients.
                loss.backward()
            
                # Clip the gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
            
                # Update parameters.
                optimizer.step()

                # Save the summed loss across the batch.
                with torch.no_grad():
                    batch_loss = torch.mean(loss)
                    self.training_results.batch_loss_history.append(batch_loss.item())

            # +++++++++++++ Finished epoch ++++++++++++++

            with torch.no_grad():
                # Check accuracy on the training set (always performed after each epoch).
                targets, predictions = evaluate_model(model=model, 
                                                   data_loader=train_data_loader,
                                                   device=self.device)  

                num_sequences = torch.tensor(targets.shape[0],dtype=torch.float32) # dividing by long() is discouraged
                train_accuracy = torch.sum(targets == predictions)/num_sequences
                self.training_results.train_accuracies_history.append(train_accuracy.item())

                # Check validation accuracy if a data loader is provided.
                validation_accuracy = -1.0 
                if(validation_data_loader is not None):
                    targets, predictions = evaluate_model(model=model, 
                                                        data_loader=validation_data_loader, 
                                                        device=self.device) 

                    num_sequences = torch.tensor(targets.shape[0],dtype=torch.float32) # dividing by long() is discouraged
                    validation_accuracy = torch.sum(targets == predictions)/num_sequences
                    self.training_results.validation_accuracies_history.append(validation_accuracy.item())

                    # Save the best-performing model (w.r.t. the validation accuracy).
                    if(validation_accuracy > self.training_results.max_validation_accuracy):
                        fail_count = 0
                        # update maximum and reset fail count if a new maximum was reached within the count limit.
                        self.training_results.max_validation_accuracy = validation_accuracy.item()
                        self.training_results.num_epochs_to_train = epoch + 1 # +1 b.c. epoch index starts at 0.
                        self.training_results.best_model = copy.deepcopy(model) # Save a copy of the best-performing model. 
                    else:
                        fail_count += 1

                    # Use early stopping if a patience value is provided.
                    if(patience is not None):
                        if(fail_count >= patience):
                            if(self.is_verbose):
                                self.log_fn("Early stopping asserted. Training finished.", 
                                            self.full_log_file_name)
                            break # Stop at the current epoch and finish training.
        

            if(model.debug_mode): # Always stop at the first iteration in debug mode.
                break

            if(self.is_verbose):
                self.log_fn("Epoch %d, Validation accuracy %1.3f, Train accuracy: %1.3f" % (epoch, validation_accuracy, train_accuracy), 
                       self.full_log_file_name)
                self.log_fn("current batch loss: %1.3f , fail count: %d" % (batch_loss, fail_count), 
                       self.full_log_file_name)                                                                                                
        



