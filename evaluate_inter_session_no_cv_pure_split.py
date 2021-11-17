'''
    Main script for evaluating the inter-session accuracy without cross-validation and
    with the validation set selected from the training set. The remaining left-out test
    session is untouched.
'''

import matplotlib.pyplot as plt
import torch
import numpy as np
import sys, os

from datasets import rs_corpus
from datasets import rs_dataset
from train_model import train_and_evaluate
from models import RsLstmModel
from utils.file_io_functions import save_results_to_file, log_message
from utils.plotting_functions import plot_sequence_stack
import datetime
import sklearn.model_selection


if __name__ == "__main__":

    dt = datetime.datetime.now()
    timestamp = "%d_%d_%d" % (dt.hour, dt.minute, dt.second)
    log_file_name = "run_log_file_" + timestamp + ".txt"

    # +++++++++++++++++ Evaluation input selection ++++++++++++++++++++++++++++++++++++

    num_model_evaluations = 20 # Number of hyperparameter search runs.
    subject_index = 0 # from [0,1]
    split_index = 1 # selected split index from [0,1,2]
    
    spectra = ['S32']
    transforms = ['mag']
    freq_start_index = 0
    freq_stop_index = 67

    max_num_hidden_units = 50

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    num_classes = 50
    num_features = len(spectra)*(freq_stop_index - freq_start_index)
    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=True,
                                                            freq_start_index=0,
                                                            freq_stop_index=freq_stop_index)

    transform_function = ptf.transform_sequences

    # Try to load the corpus from either the local corpus directory of the HPC corpus directory.
    paths = rs_corpus.get_corpus_full_file_paths()

    # Try to load the corpus with one of the provided paths.
    for path_name in paths:
        path = paths[path_name]
        training_corpus = rs_corpus.load_corpus_from_file(os.path.join(path, "processed_training_corpus.pkl"))
        if training_corpus is not None:
            log_message("Loaded Corpus.", log_file_name)
            break

    # If training_corpus is still unloaded, exit with error.
    if training_corpus is None:    
        sys.exit("Error: training_corpus is None. Check the path.")

    ### Define the splits for inter-session evaluation [train (2) | test (1)]. ###
    all_session_split_indices = [[0, 1, 2], [1, 0, 2], [2, 0, 1]] # all possible splits

    train_session_indices = all_session_split_indices[split_index][0:-1]
    test_session_index = all_session_split_indices[split_index][-1]

    train_dataset = rs_dataset.RsDataSet(training_corpus, 
                            subject_index=subject_index, 
                            session_indices=train_session_indices, 
                            transform_fn=transform_function)

    test_dataset = rs_dataset.RsDataSet(training_corpus, 
                           subject_index=subject_index, 
                           session_indices=test_session_index, 
                           transform_fn=transform_function)

    log_message("Created train/test datasets for subject S00%d, split %d." % (subject_index+1, split_index), log_file_name)

    ### Create the train/validation split indices. ###
    X = np.linspace(0,train_dataset.__len__()-1, train_dataset.__len__(),dtype='int32') # radargram sequence indices.
    y = train_dataset.y_set_categorical

    # create a stratified set of indices for the training and validation set.
    num_classes = 50
    train_fraction = 0.8
    validation_fraction = 0.2
    train_indices, validation_indices = sklearn.model_selection.train_test_split(X, 
                                                                                 train_size=train_fraction,
                                                                                 test_size=validation_fraction, 
                                                                                 stratify=y)
     
    # Double Check the splits.
    num_reps_per_session = 10
    expected_counts_per_class_train = train_dataset.__len__()/num_classes*train_fraction
    expected_counts_per_class_valid = train_dataset.__len__()/num_classes*validation_fraction
    values, counts = np.unique(y[train_indices], return_counts=True)
    if(np.sum(counts == expected_counts_per_class_train) != num_classes):
        sys.exit("Error: unbalanced test split.")    
    values, counts = np.unique(y[validation_indices], return_counts=True)
    if(np.sum(counts == expected_counts_per_class_valid) != num_classes):
        sys.exit("Error: unbalanced validation split.")

    # Create the testing & validation data sampler.
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)

    # +++++++++++++++++++++++ Hyperparameter selection ++++++++++++++++++++++++++++++++++
    batch_size = 8
    dropout_prob =  0 # Currently fixed at 0 %.
    num_lstm_layers = 1 # Currently fixed at 1.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    log_message("Current Subject: S00%d" % (subject_index+1), log_file_name)
    log_message("Spectra: %s" % (spectra), log_file_name)
    log_message("Transforms: %s" % (transforms), log_file_name)
    log_message("number of features: %d" % (num_features), log_file_name)
    log_message("Used device: %s" %(device), log_file_name)

    # Initialize the hyperparameter dict with the fixed values and default values for variable ones.
    hyperparameters = {'input_size' : num_features,
                        'output_size' : num_classes,
                        'num_hidden_units' : 20,
                        'num_lstm_layers' : num_lstm_layers,
                        'dropout_prob' : dropout_prob,
                        'is_bidirectional' : True,
                        'batch_size': batch_size,
                        'learning_rate' : 0.001,
                        'num_epochs' : 200}

    log_message("number of features: %d" % (num_features), log_file_name)
    log_message("Used device: %s" %(device), log_file_name)

    # Create a vector with random learning rates to pick from.
    num_learning_rates = 100 # sample the learning rate from a logarithmically spaced distribution.
    lower_th = torch.log10(torch.tensor(0.0005)).item()
    upper_th = torch.log10(torch.tensor(0.005)).item()
    c = torch.logspace(lower_th, upper_th, num_learning_rates)

     # Instantiate the 3 data_loader.
    test_data_loader = torch.utils.data.DataLoader(test_dataset, 
                                                   batch_size=hyperparameters['batch_size'], 
                                                   shuffle=False, 
                                                   num_workers=0,
                                                   collate_fn=rs_dataset.pad_seqs_and_collate)

    validation_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=hyperparameters['batch_size'], 
                                                    shuffle=False, 
                                                    num_workers=0,
                                                    collate_fn=rs_dataset.pad_seqs_and_collate,
                                                    sampler=validation_sampler)

    # Note: validation_data_loader sample is not 100 % necessary, for the training set it
    # accelerates convergence, though.
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=hyperparameters['batch_size'], 
                                                shuffle=False, 
                                                num_workers=0,
                                                collate_fn=rs_dataset.pad_seqs_and_collate,
                                                sampler=train_sampler)

    log_message("Created DataLoader.", log_file_name)

    # +++++++++++++++++++++++ Hyperparameter Optimization ++++++++++++++++++++++++++++++++++

    log_message("+++ Starting hyperparameter random search for " + str(num_model_evaluations) + " runs +++", log_file_name)
    
    for run_index in range(num_model_evaluations):
    
        ### Select the variable hyperparameters subject to optimization. ###
        log_message("Selecting variable hyperparameters.", log_file_name)

        # Pick the learning rate. 
        learning_rate = c[np.random.randint(low=0, high=num_learning_rates-1)].item()

        # Randomly select the number of hidden units.
        num_hidden_units = (torch.randint(low=20, high=max_num_hidden_units, size=(1,))).item()

        hyperparameters['num_epochs'] = 200

        # Assign variable hyperparameters.
        hyperparameters['num_hidden_units'] = num_hidden_units;
        hyperparameters['learning_rate'] = learning_rate;

        log_message("Instantiating model with parameters:", log_file_name)
        for parameter in hyperparameters:
            log_message("%s %s" % (parameter, hyperparameters[parameter]), log_file_name)

        log_message("Starting hyperparameter set evaluation nr. %d" % (run_index), log_file_name)

        # Instantiate new model & optimizer.
        rs_lstm_model = RsLstmModel(hyperparameters, device=device, debug_mode=False).to(device)
        optimizer = torch.optim.Adam(rs_lstm_model.parameters(), 
                                                lr=hyperparameters['learning_rate'])
        trainer = train_and_evaluate.Trainer(num_epochs=hyperparameters['num_epochs'],
                                                device=device,
                                                is_verbose=True,
                                                log_fn=log_message)
        trainer.full_log_file_name = log_file_name # use same logging function for training outputs.

        # Train the model.
        log_message("Started training.", log_file_name)

        trainer.fit_model(model=rs_lstm_model, 
                            optimizer=optimizer, 
                            train_data_loader=train_data_loader, 
                            validation_data_loader=validation_data_loader, 
                            patience=20)

        hyperparameters['num_epochs'] = trainer.training_results.num_epochs_to_train # override initial value for saving.

        print("highest validation accuracy: %1.4f" % (trainer.training_results.max_validation_accuracy)) # For HPC interactve session feedback.

        # Note: copy the tensors back to the cpu for file export.
        training_results = {"batch_losses" : trainer.training_results.batch_loss_history,
                            "validation_metric" : trainer.training_results.validation_accuracies_history,
                            "train_metric" : trainer.training_results.train_accuracies_history,
                            "validation_accuracy" : trainer.training_results.max_validation_accuracy}

        train_results_file_name = "lstm_optimization_" + timestamp + "_" + str(run_index) + "_train.txt"

        save_results_to_file(hyperparameters=hyperparameters, 
                                history=training_results,
                                file_name=train_results_file_name,
                                append=True)

        # +++++ Calculate the test accuracy for the current hyperparameter set. +++++

        # This is necessary to evaluate, whether the out-of-session subset used as the validation set
        # really is a good predictor of the highest test accuracy on the out-of-session test set
        # (or at least better than the in-session validation set, which it was shown not to be).
        # Note: since the full training set was already used for training, there is no need to retrain
        # the model but use the saved best model from the trainer object.

        log_message("Testing on test set.", log_file_name)
        test_targets, test_predictions = train_and_evaluate.evaluate_model(model=trainer.training_results.best_model,
                                                                        data_loader=test_data_loader,
                                                                        device=device)

        num_sequences = torch.tensor(test_targets.shape[0],dtype=torch.float32) # division by long() is discouraged
        test_accuracy = torch.sum(test_targets == test_predictions)/num_sequences 

        test_results = {"batch_losses" : trainer.training_results.batch_loss_history,
                        "train_metric" : trainer.training_results.train_accuracies_history,
                        "test_accuracy" : test_accuracy.cpu(),
                        "test_targets" : test_targets.cpu(),
                        "test_predictions" : test_predictions.cpu()}

        test_results_file_name = "lstm_optimization_" + timestamp + "_" + str(run_index) + "_test.txt"

        save_results_to_file(hyperparameters=hyperparameters, 
                                history=test_results,
                                file_name=test_results_file_name,
                                append=False)



