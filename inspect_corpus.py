'''
    Script to inspect the radargrams loaded into the rs corpus.
    Please build the training corpus .pkl file first and
    modify the path to locate it to where the .pkl file is located.
'''

from datasets.rs_corpus import load_corpus_from_file
# from datasets.rs_corpus import RsCorpus
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    training_corpus = load_corpus_from_file("PATH-TO-CORPUS\\processed_training_corpus.pkl")

    if training_corpus is not None:
        for r_index in range(10):
            seq_index = np.random.randint(0, 501)
            subject_index = np.random.randint(0, 2)
            session_index = np.random.randint(0, 3)
            s = training_corpus.sequences['S32'][subject_index][session_index][seq_index]
            l = training_corpus.labels[subject_index][session_index][seq_index]

            print("Label: {}".format(l))
            plt.imshow(abs(s))
            plt.show()
    else:
        print("training_corpus is of type None and not loaded correctly. Please check the path.")
