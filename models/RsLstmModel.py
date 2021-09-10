import torch

class RsLstmModel(torch.nn.Module):
    '''
        Full LSTM-based model for sequence-to-sequence (labels) prediction of rs frames.
        Layer stack:
        ----- Name -----    |   ------ Dimensions ------
        Input layer (sequences: [batch_size, seq_length, input_size], batch_first=True)
        --> (n) LSTM layer     [input_size, hidden_size]
        --> Dropout layer      integrated in the lstm layer
        --> FC layer           [input_size, output_size]
        --> Softmax layer      [input_size, output_size]
        --> Output probs.      [input_size, output_size]
    '''

    def __init__(self, hyperparameters, device="cpu", debug_mode=False):
        '''
            Args:
                hyperparameters (dict()): dictionary containing all relevant hyperparameters.
                device (str): device chosen for computation.
                debug_mode (bool): Toggle debug outputs on/off.
            Returns:
                None.
        '''

        super(RsLstmModel, self).__init__()
        
        # Store model inputs for easy access.
        self.input_size = hyperparameters['input_size']
        self.output_size = hyperparameters['output_size']
        self.hidden_size = hyperparameters['num_hidden_units']
        self.num_layers = hyperparameters['num_lstm_layers']
        self.batch_size = hyperparameters['batch_size']
        self.dropout_prob = hyperparameters['dropout_prob']
        self.is_bidirectional = hyperparameters['is_bidirectional']
        self.num_directions = 2 if self.is_bidirectional else 1
        self.num_epochs = hyperparameters['num_epochs']
        self.debug_mode = debug_mode
        self.device = device
       
        # Dropout is only applied *between* lstm layers by default. Suppresses the warning.
        if(self.num_layers > 1):
            self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=self.dropout_prob,
                                bidirectional=self.is_bidirectional,
                                batch_first=True)
        else:
            self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                dropout=0,
                                bidirectional=self.is_bidirectional,
                                batch_first=True)
        
        # Dropout layer between the last lstm layer and the fully connected layer 
        # (always present).
        self.dropout = torch.nn.Dropout(p=self.dropout_prob)
        
        # Fully connected output layer.
        self.linear = torch.nn.Linear(in_features=self.hidden_size*self.num_directions, 
                                out_features=self.output_size)
      
        
    def forward(self, X_batch, x_lengths):
        '''
            Necessary function to forward the input through the model.
            @param X_batch: [3d tensor] containing the padded sequences [batch_size, max_seq_length, input_size]
            @ param x_lengths: [1-d tensor] containing the unpadded(!) sequence lengths
            @return y_pred: discrete probability distribution of classes for the last output
                            of each batch item [batch_size, input_size]
        '''
        # Note: 2-d sequences (any lengths) are fed into the lstm layer as one concatenated
        # 2-d array of shape [sum of all sequence lengths, input_size]. Each row = h_T
        
        # Reset hidden state for each new batch of sequences.
        # Not passing in a hidden state in the lstm forward() pass will automatically
        # set the hidden state (c_0, h_0) to 0.
        # https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
        # https://www.kdnuggets.com/2018/06/taming-lstms-variable-sized-mini-batches-pytorch.html
        # https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
        batch_size, max_seq_length, _ = X_batch.shape
                     
        h_0 = torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)
        c_0 = torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_size, requires_grad=True).to(self.device)
              
        if(self.debug_mode):
            print("Hidden/cell state shape: {}".format(h_0.shape))
            
        # Pack the sequences in X_batch.
        # X_batch: [batch_size, max_seq_length, input_size] 
        #          -> [sum of seq_lengths (for all elements in batch), input_size]
        if(self.debug_mode):
            print("X_batch shape before packing: {}".format(X_batch.shape))
        
        # NOTE: x_lengths in pack_padded_sequence() apparently cannot reside on the GPU as for now.
        # https://github.com/pytorch/pytorch/issues/43227
        X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch, 
                                                          x_lengths,
                                                          batch_first=True, 
                                                          enforce_sorted=False)
        
        X_batch = X_batch.to(self.device)

        if(self.debug_mode):
            print("X_batch.data shape after packing: {}".format(X_batch.data.shape))
            
        # Pass the sequences in X_batch to the lstm layer.
        # X_batch: [sum of seq_lengths (for all elements in batch), input_size]
        # X_batch, self.hidden = self.lstm(X_batch, self.hidden) 
        # https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
        # suggested detach() the h_0 and c_0 layers to prevent any backprop. across batches,
        # but this should not be possible anyways as h_0, c_0 are newly created upon each lstm() function call (?)
        output, (h_T, c_T) = self.lstm(X_batch, (h_0.detach(), c_0.detach())) 
        
        if(self.debug_mode):
            print("X_batch.data shape after lstm layer: {}".format(output.data.shape))
            print("h_T shape: {}".format(h_T.shape))
        
        h_T = h_T.contiguous() 

        if(self.is_bidirectional):
            h_T = torch.cat((h_T[-1,:,:], h_T[-2,:,:]), dim=-1) # Note: see bilstm_output_shapes.pdf for an explanation. TODO: ADD TO GIT
        else:
            h_T = h_T[-1,:,:]
        
        if(self.debug_mode):
            print("h_T before linear layer: {}".format(h_T.shape))
        
        # Apply dropout.
        self.dropout(h_T)
        
        # Apply the fully connected layer.
        h_T = self.linear(h_T)
        
        if(self.debug_mode):
            print("h_T shape after linear layer: {}".format(h_T.shape))
        
        # Apply the softmax layer.
        h_T = torch.nn.functional.log_softmax(h_T, dim=1)
        
        if(self.debug_mode):
            print("h_T shape after softmax layer, before returning: {}".format(h_T.shape))
        
        if(self.debug_mode):
            print("h_T gradient: {}".format(h_T.requires_grad))

        return h_T 
