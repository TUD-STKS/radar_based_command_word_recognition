'''
    Script for building the training corpus from individual files.
    Please modify the toplevel path to the specific location where the corpus is stored,
    i.e., 
    .../corpus/ <----
        |_ S001
        |_ S002 
          |_ etc.
'''

from datasets import rs_corpus
import numpy as np

if __name__ == '__main__':

    path_to_corpus = "PATH_TO_CORPUS\\corpus"

    training_corpus = rs_corpus.RsCorpus()
    
    try:
        training_corpus.load_files(path_to_corpus=path_to_corpus,
                                   is_verbose=True)

        training_corpus.remove_zero_frames(is_verbose=True)

        corpus_file_name = "processed_training_corpus"
        rs_corpus.save_corpus_to_file(training_corpus, corpus_file_name)
    except FileNotFoundError:
        print("Could not load the corpus content from %s, please check the path." % (path_to_corpus))
    except NotADirectoryError: 
        print("%s does not contain a valid directory (still zipped?)." % (path_to_corpus))

