# radar_based_command_word_recognition
This repository contains supplementary code for the radar-based command word recognition study from the paper 
"Silent speech command word recognition using stepped frequency continuous wave radar".

## Building the corpus

Since the recorded corpus is reasonably small, the corpus is built as a binary file
and fully loaded during training. Please run the script

  `build_training_corpus.py`  

to build the corpus from the individual files, after adjusting the file path to 
where the corpus is stored. 
The corpus itself can be downloaded from
https://www.vocaltractlab.de/index.php?page=birkholz-supplements

the binary corpus file **processed_training_corpus.pkl** will be created in the
same folder that the script was run in. The binary file can reside here or
stored anywhere else.
Afterwards, please manually add the path to the dict that is returned by the function

  `rs_corpus.get_corpus_full_file_paths()`

https://www.vocaltractlab.de/index.php?page=birkholz-supplements also provided
a pre-built binary corpus, both for python (identical to what would be returned
from manually building it) and for MATLAB.

## Experiments and training

### Inter-session experiment 1

To repeat the training of the BiLSTM as explained in the paper, please run

  `evaluate_inter_session_no_cv_pure_split.py`

Training is done without cross validation and for the case where two sessions
are used for training, while the remaining one is used as the test set.
Inside this script, a number of parameters can be changed by the user:

The number of model evaluations for the hyperparameter search:

`num_model_evaluations = 20` 

The subject index (0 or 1 for either subject)

`subject_index = 0` 

The split index, which specifies how the three sessions of each subject
are split into training and test set, following the notation of
{training session 1, training session 2 | training session 3}.

`split_index = 1` # selected split index from [0,1,2]

Example: split_index 0 uses session 1 & 2 for training and session 3 for testing.

The spectra used as input features and the associated transform that is applied
to that spectrum.

`spectra = ['S32']`

`transforms = ['mag']`

Example: spectra = ['S32','S21'], transforms = ['mag','mag_delta']
The allowed spectra and transforms can be returned with the function

`train_and_evaluate.get_allowed_spectra_and_transform_keys()`

The start and stop index of the frequencies used of each spectrum.
The maximal value is 128 (corresponding to 128 frequency points from 1 to 6 GHz).

`freq_start_index = 0`

`freq_stop_index = 67`

The maximal number of hidden units used for the BiLSTM layer.

`max_num_hidden_units = 50`

### Inter-session experiment 2

The second evaluation, where the hyperparameters are optimized on a small subset
of the remaining session, can be repeated using the script

  `evaluate_inter_session_no_cv.py`

The user input is identical.

### Intra-session experiment

The intra-session experiment can be run with the script

  `evaluate_intra_session_no_cv.py`

