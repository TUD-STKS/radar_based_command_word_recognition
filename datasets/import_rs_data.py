### This module contains the function to read in the binary radar speech data files. ###
# Version date: 27-05-2021

import numpy as np

def import_rs_binary_file(full_file_path: 'Full folder path and file name.bin'):
    try:
        rs_data_file = open(full_file_path, 'rb')
    except OSError as exception:
        print("Could not open file {}".format(full_file_path))
        print(exception)
        return;

    binary_string = rs_data_file.read()
    rs_data_file.close()
		
    # decode the binary stream.
    # Define the byte sizes for all fixed values.
    time_string_size_bytes = 4      # int32
    num_spectra_size_bytes = 1      # char
    num_steps_size_bytes = 4        # int32
    num_total_frames_size_bytes = 4 # int32
    start_frame_index_bytes = 4     # int32
    stop_frame_index_bytes = 4      # int32
    pll_freq_point_bytes = 4        # int32
    pll_power_level_point_bytes = 4 # int32
    time_stamp_size_bytes = 2       # uint16
    re_im_part_size_bytes = 4       # float32
    byte_order = 'little'

    class Rs_data():
        def __init__(self):
            self.MAX_NUM_FREQ_POINTS = 256 # fixed value for RS HW Revision 1.0
            self.formatted_time = ""
            self.num_spectra = -1
            self.displayed_spectrum_indices = []
            self.num_steps = -1
            self.num_total_frames = -1
            self.start_frame_index = -1
            self.stop_frame_index = -1
            self.pll1_frequencies_kHz = []
            self.pll2_frequencies_kHz = []
            self.pll1_power_levels = []
            self.pll2_power_levels = []
            self.time_stamps = []
            self.radargrams = []

    rs_data = Rs_data()

    # Time stamp bytes.
    start_index = 0
    stop_index = time_string_size_bytes
    num_time_stamp_bytes = int.from_bytes(binary_string[start_index:stop_index], byte_order)

    # Time stamp.
    start_index = stop_index
    stop_index = start_index + num_time_stamp_bytes
    rs_data.formatted_time = binary_string[start_index:stop_index]

    # Number of spectra.
    start_index = stop_index
    stop_index = start_index + num_spectra_size_bytes
    rs_data.num_spectra = int.from_bytes(binary_string[start_index:stop_index], byte_order)

    # Displayed spectrum indices list.
    start_index = stop_index
    stop_index = start_index + rs_data.num_spectra
    rs_data.displayed_spectrum_indices = [num for num in binary_string[start_index:stop_index]]

    # Number of spectra.
    start_index = stop_index
    stop_index = start_index + num_steps_size_bytes
    rs_data.num_steps = int.from_bytes(binary_string[start_index:stop_index], byte_order)

    # Number of total frames.
    start_index = stop_index
    stop_index = start_index + num_total_frames_size_bytes
    rs_data.num_total_frames = int.from_bytes(binary_string[start_index:stop_index], byte_order)

    # Start frame index.
    start_index = stop_index
    stop_index = start_index + start_frame_index_bytes
    rs_data.start_frame_index = int.from_bytes(binary_string[start_index:stop_index], byte_order)

    # Stop frame index.
    start_index = stop_index
    stop_index = start_index + stop_frame_index_bytes
    rs_data.stop_frame_index = int.from_bytes(binary_string[start_index:stop_index], byte_order)

    # Pll1 (TX) frequencies [kHz]
    start_index = stop_index
    stop_index = start_index + rs_data.num_steps*pll_freq_point_bytes
    rs_data.pll1_frequencies_kHz = np.fromstring(binary_string[start_index:stop_index], '<i')

    # Pll2 (LO) frequencies [kHz]
    start_index = stop_index
    stop_index = start_index + rs_data.num_steps*pll_freq_point_bytes
    rs_data.pll2_frequencies_kHz = np.fromstring(binary_string[start_index:stop_index], '<i')

    # Pll1 (TX) Power Level
    start_index = stop_index
    stop_index = start_index + rs_data.num_steps*pll_power_level_point_bytes
    rs_data.pll1_power_levels = np.fromstring(binary_string[start_index:stop_index], '<i')

    # Pll2 (LO) Power Level
    start_index = stop_index
    stop_index = start_index + rs_data.num_steps*pll_power_level_point_bytes
    rs_data.pll2_power_levels = np.fromstring(binary_string[start_index:stop_index], '<i')

    # Timestamps.
    start_index = stop_index
    stop_index = start_index + rs_data.num_total_frames*time_stamp_size_bytes
    rs_data.time_stamps = np.fromstring(binary_string[start_index:stop_index], '<H') # H = unsigned short

    # Radargrams. The radargrams are always rs_data.MAX_NUM_FREQ_POINTS (256) data points long. 
    # Unused entires (if the sweep contains < 256 frequency points) are 0.
    start_index = stop_index
    radargram_re_im_parts = np.fromstring(binary_string[start_index:], dtype='<f')

    radargrams = radargram_re_im_parts[0::2] + radargram_re_im_parts[1::2]*1j # complex values are stored [re,im,re,im...]
    # Note: radargrams is a column array. This affects how to resize it compared to the Matlab implementation of this function.

    for s_index in range(0, rs_data.num_spectra):
        r = np.reshape(radargrams[s_index::rs_data.num_spectra], (rs_data.num_total_frames, rs_data.MAX_NUM_FREQ_POINTS)) # radargram for the sparam index ([1,...9] max)
        rs_data.radargrams.append(r[:, :rs_data.num_steps])

    return rs_data
