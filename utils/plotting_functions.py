import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch

def plot_packed_sequence_data(packed_sequence, unpadded_lengths):
    '''Function for plotting a packed sequence as a continuous image
    with borders for the variable length sequences overlayed.

    Args:
        packed_sequence (torch.nn.utils.rnn.PackedSequence): packed sequence to be plotted.
        unpadded_lengths(1-d numpy array, 1-d tensor): Vector containing the unpadded lengths' 
            of the sequences.

    Returns:
        None.
    '''

    figure(figsize=(20, 14), dpi=80)
    plt.imshow(packed_sequence.data)
    start_index = 0
    for line_index in range(batch_size):
        stop_index = start_index + unpadded_lengths[line_index]
        plt.plot([0, 127],[stop_index, stop_index], '-r')
        start_index = stop_index


# #####################################################################

def plot_sequence_stack(sequence_stack):
    '''Function for plotting a 3-d stack of sequences as 2-d images.

    Args:
        sequence_stack (3-d tensor): packed sequence to be plotted. Has size [batch_size, max_seq_length, input_size].

    Returns:
        None.
    '''

    figure(figsize=(20, 14), dpi=80)
    
    batch_size, max_length, input_size = sequence_stack.shape
    
    image = sequence_stack[0] # first image.
    
    # concatenate the rest onto it.
    for seq_index in range(1, batch_size):
        image = torch.cat((image, sequence_stack[seq_index]), dim=0)
        
    plt.imshow(image)
    plt.show()
