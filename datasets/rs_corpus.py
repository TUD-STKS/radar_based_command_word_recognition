import os
import sys
import numpy as np
if(sys.version_info.major >= 3 and sys.version_info.minor >= 8):
    import pickle
else:
    import pickle5 as pickle # Mainly for the HPC to load the corpus correctly.

from datasets.import_rs_data import import_rs_binary_file


def get_corpus_full_file_paths():
    ''' Returns the potential paths where the corpus content is stored, both for the 
        individual files and the binary corpus file.
        MODIFY THIS FUNCTION TO CHANGE THE PATHs TO THE CORPUS.

    Args:
        None.

    Returns:
        paths (dict): Dictionary containing the different full file paths.
    '''

    local_pc_path = "LOCAL_PATH_TO_BINARY_PKL_CORPUS" # PLEASE ADJUST
    hpc_path_warm = 'HPC_WARM_PATH_TO_BINARY_PKL_CORPUS'
    hpc_path_home = 'HPC_HOME_PATH_TO_BINARY_PKL_CORPUS'
    path_dict = {'local_pc_path' : local_pc_path,
             'hpc_path_warm' : hpc_path_warm,
             'hpc_path_home' : hpc_path_home}

    return path_dict


class RsCorpus():
    ''' Class that holds the complete training corpus with all sequences for all subjects
        and all sessions.
        Syntax: access to a list of the complex 2-d numpy arrays (radar spectrograms) via 
            training_corpus.sequences['SXX'][subject_index][session_index]
            SXX is the selected spectrum indentifier and
            training_corpus.labels[subject_index][session_index] for the string labels.
            Notation with respect to notation used in the accompanying paper:
            S12 = S1(f)
            S32 = S2(f)

        Note:
            The corpus must has to follow the following folder structure:
            SXXX for the subject id. (E.g.: S001)
            SESXX for the session id for each subject. (E.g.: SES01)
    '''

    def __init__(self):
        '''Constructor. 
            
            Args:
                None.

            TODO:
                Use self.audio and self.radar_data_info
        '''

        #### Member variables. ###
            
        self.subject_ids = [] # list of all subject ids as strings.
        self.session_ids = {} # dict() of session_ids, indexed by the subject_ids. Ex.: session_ids['S001'] = ['SES01','SES02','SES03']
        self.num_subjects = 0
        self.num_sequences_per_session = {} # tuples of the number of sequences for each session for all subjects.
                                            # Ex.: num_sequences_per_session['S001'] = (500,500,500)

        self.sequences = {'S12' : [],
                          'S32' : []}
        self.sparams = list(self.sequences.keys())
        self.audio = [] # currently unused
        self.labels = [] 
        self.radar_data_info = [] # currently unused
                  
    ### Member Functions. ###

    def load_files(self, path_to_corpus, is_verbose=False):
        ''' Load all radargrams, labels from file and store them
            in the training corpus.

            Args:
                path_to_corpus (str): full file path to where the corpus is located.
                verbose (bool): Toggle misc. print statements on/off.
                    
            Returns:
                None.

            TODO:
                add audio files if necessary.
        '''

        self.subject_ids = os.listdir(path_to_corpus)
        self.num_subjects = len(self.subject_ids)

        # Scan all subject folders for the session folders and -subsequently- rs/label/(audio) files.
        for subject_index in range(self.num_subjects):
            path_to_sessions = os.path.join(path_to_corpus, self.subject_ids[subject_index])
            session_ids = os.listdir(path_to_sessions)
            self.session_ids[self.subject_ids[subject_index]] = session_ids
            
            num_sequences_per_session = []
            # Determine the number of files per session.
            for session_index in range(len(session_ids)):
                path_to_files = os.path.join(path_to_sessions, session_ids[session_index], 'radarData')
                num_sequences = len(os.listdir(path_to_files))
                num_sequences_per_session.append(num_sequences)

            self.num_sequences_per_session[self.subject_ids[subject_index]] = tuple(num_sequences_per_session) # make it immutable.
        
        # List the found content.
        if(is_verbose):
            for subject_id in self.subject_ids:

                num_sessions = len(self.session_ids[subject_id])

                for session_index in range(num_sessions):
                    print("Found subject {}, session {}, {} samples".format(subject_id, 
                                                                            self.session_ids[subject_id][session_index], 
                                                                            self.num_sequences_per_session[subject_id][session_index]))

        
        # Parse the files.
        for subject_id in self.subject_ids:
                    
            # Append the loaded values for each subject in lists.
            all_sessions_labels = []
            all_sessions_S12 = []
            all_sessions_S32 = []

            for session_id in self.session_ids[subject_id]:
        
                if(is_verbose):
                    print("Processing subject {:s}, session {:s}:".format(subject_id, session_id))

                rd_full_file_path = os.path.join(path_to_corpus, subject_id, session_id, 'radarData')
                labels_full_file_Path = os.path.join(path_to_corpus, subject_id, session_id, 'labels')
        
                # Locate the radargrams and label files for each session.
                radar_data_directory = os.listdir(rd_full_file_path)
                num_rs_data_files = len(radar_data_directory)
        
                labels_directory = os.listdir(labels_full_file_Path)
                num_label_files = len(labels_directory)
        
                if(num_rs_data_files != num_label_files):
                    print("Error: {} rs data files, but {} label files.".format(num_rs_data_files, num_label_files))
                    raise ValueError
                else:
                    if(is_verbose):
                        print("{} files found.".format(num_label_files))
           
                # Load the labels for each session.
                labels = []    

                 # file by file.
                for file_index in range(num_label_files):
                    curr_label_full_file_path = os.path.join(labels_full_file_Path, labels_directory[file_index]) 

                    with open(curr_label_full_file_path, 'r') as label_file:
                        try:
                            label = label_file.read().replace('\n','')
                            labels.append(label)
                        except OSError:
                            print("Could not open file at {}".format(curr_label_full_file_path))
                            return
                    
                all_sessions_labels.append(labels)
        
                # Load the radar data for each session.
                S12 = []
                S32 = []
    
                # file by file.
                for file_index in range(num_rs_data_files):
                    try:
                        curr_rd_full_file_path = os.path.join(rd_full_file_path, radar_data_directory[file_index])
                        rs_data = import_rs_binary_file(curr_rd_full_file_path)
                        s12 = rs_data.radargrams[1] # hardcoded index 1 for S12
                        s32 = rs_data.radargrams[7] # hardcoded index 7 for S32
                        S12.append(s12)
                        S32.append(s32)
                    except NameError:
                        print("import_rs_binary_file() function not found.")
                        return
                   
                all_sessions_S12.append(S12)
                all_sessions_S32.append(S32)
                        
            # After parsing all session for the subject, append to the global lists.
            self.labels.append(all_sessions_labels)
            self.sequences['S12'].append(all_sessions_S12)
            self.sequences['S32'].append(all_sessions_S32)
    

    ####################################################################################
   
    def remove_zero_frames(self, is_verbose=False):
        ''' Finds and removes all zero frames from the radargrams stored in the corpus
            due to rare frame skips.
        

        Args:
            is_verbose (bool): print the current state of the search.

        Returns:
            None.
        '''

        for sparam_index in range(len(self.sparams)):
            # 0: S12, 1: S32
            if(is_verbose):
                print("Evaluating for sparam {}".format(self.sparams[sparam_index]))

            for subject_index in range(self.num_subjects):
                
                subject_id = self.subject_ids[subject_index]
                num_sessions = len(self.session_ids[subject_id])

                for session_index in range(num_sessions):

                    num_files = self.num_sequences_per_session[subject_id][session_index]

                    for file_index in range(num_files):

                            if(sparam_index == 0):
                                radargram = self.sequences['S12'][subject_index][session_index][file_index]
                            else:
                                radargram = self.sequences['S32'][subject_index][session_index][file_index]

                            num_frames, num_freqs = np.shape(radargram)
                    
                            # Check for missing frames (their sum is zero across all frequencies).
                            for frame_index in range(num_frames):
                                num_zero_values = np.sum((radargram[frame_index, :] == 0))

                                if(num_zero_values == num_freqs):
                                    if(is_verbose):
                                        print("Found zero frame for {}, subject {}, session {}, file {}, frame {}".format(
                                            self.sparams[sparam_index],
                                            subject_index, 
                                            session_index, 
                                            file_index,
                                            frame_index))

                                    # Check, if the very first frame is affected.
                                    if(frame_index == 0):
                                        if(is_verbose):
                                            print("Found zero frame at frame index 0.")

                                        # Find frame index of first non-zero frame.
                                        local_frame_index = 0
                                        # Check, if any following frames are also zero.
                                        while(True):
                                            num_zero_values = np.sum((radargram[local_frame_index+1, :] == 0))
                                            if(num_zero_values == num_freqs):
                                                local_frame_index += 1
                                                if(is_verbose):
                                                    print("Also found zero frame at frame index {}".format(local_frame_index))
                                            else:
                                                break;

                                        # Pad from first non-zero frame back to to frame index 0.
                                        for index in range(local_frame_index, -1, -1):
                                            radargram[index, :] = radargram[index + 1, :]
                                            if(is_verbose):
                                                print("Padded frame {} with {}".format(index, index + 1))
                                    # If the zero frame is anywhere else in the radargram, pad with previous frame.
                                    else: 
                                        radargram[frame_index, :] = radargram[frame_index - 1, :]


###########################################################################################

def save_corpus_to_file(corpus, file_name):
    ''' Save the corpus with pickle.

    Args:
        corpus (RsCorpus): RsCorpus object.
        file_name (str): file name of the corpus.

    Returns:
        None.

    ToDo:
        Add path selection.
    '''
        
    _file_name = file_name + ".pkl"

    print("Saving corpus to file...")
    with open(_file_name, 'wb') as output:
        pickle.dump(corpus, output, pickle.HIGHEST_PROTOCOL)    

    print("Done.")


###########################################################################################

def load_corpus_from_file(full_corpus_file_path):
    ''' Loads the corpus from the pickle binary file.

    Args:
        full_corpus_file_path (str): file name including its path and extension.
    
    Returns:
        training_corpus (TrainingCorpus): loaded training corpus from file.
        Syntax: access to a list of the complex 2-d numpy arrays (radargrams) via 
            training_corpus.sequences['SXX'][subject_index][session_index]
            SXX is the selected sparameter/spectrum indentifier and
            subject_index, session_index as integers.
            training_corpus.labels['SXX'][subject_index][session_index] for the string labels.


    '''

    training_corpus = RsCorpus()

    try:
        with open(full_corpus_file_path, 'rb') as training_corpus_file:
            training_corpus = pickle.load(training_corpus_file)
            print("Loaded corpus.")
            return training_corpus
    except:
        print("Could not load corpus {}".format(full_corpus_file_path))




