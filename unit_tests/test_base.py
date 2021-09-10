import pytest
import torch
import numpy as np
from train_model import train_and_evaluate
from unit_tests import fixtures
from datasets import rs_corpus, rs_dataset

def test_calculate_delta_sequence():

    calculated_output = train_and_evaluate.calculate_delta_sequence(fixtures.return_complex_tensor())
    test_output = fixtures.return_precalculated_delta_features();

    epsi = 1e-04
    num_cols = 15
    num_rows = 10

    for col in range(test_output.shape[0]):
        for row in range(test_output.shape[1]):
            assert(torch.abs(test_output[col][row] - calculated_output[col][row]) < epsi)


####################################################################################################

def test_verify_feature_selection():


    allowed_spectrum_keywords, allowed_transform_options = train_and_evaluate.get_allowed_spectra_and_transform_keys()

    num_pairs = 20
    spectrum_keys_list, sequence_indices_list, transforms_list = fixtures.generate_random_spectra_transform_pairs(allowed_spectrum_keys=allowed_spectrum_keywords, 
                                                                                  allowed_transforms=allowed_transform_options, 
                                                                                  num_pairs=num_pairs)


    for index in range(num_pairs):
        # Test the correct assertion of object instantiation if the spectrum or transform keys are incorrect.
        ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectrum_keys_list[index],
                                                                transforms=transforms_list[index],
                                                                is_normalized=True,
                                                                freq_start_index=0,
                                                                freq_stop_index=127)

        # Test the correct mapping from string keys to spectrum indices inside the ptf object constructor.
        assert(ptf.sequence_indices == sequence_indices_list[index])



####################################################################################################

def test_validate_corpus_content():

    paths = rs_corpus.get_corpus_full_file_paths()

    for path_name in paths:
        path = paths[path_name]
        training_corpus = rs_corpus.load_corpus_from_file("%sprocessed_training_corpus.pkl" % (path))
        if training_corpus is not None:
            print("Loaded corpus.")
            break

    if training_corpus is None:   
        print("Could not load corpus from path %s" % (path_to_corpus))
        assert(False)

    spectra_keys, _ = train_and_evaluate.get_allowed_spectra_and_transform_keys()
    subject_indices = [0,1]
    session_indices = [0,1,2]
    num_reps_per_session = 500

    for spectrum in spectra_keys: 
        for subject_index in subject_indices:
            for session_index in session_indices:
                sequences = training_corpus.sequences[spectrum][subject_index][session_index]
                assert(num_reps_per_session == len(sequences))

                for sequence in sequences:
                    # Check if all zero-frames were removed correctly.
                    [num_frames, num_freqs] = sequence.shape
                    for frame_index in range(num_frames):
                        frame_sum = np.sum(sequence[frame_index,:])
                        assert(frame_sum != 0)

                    # Check for any real/imag values > 2 (not possible with fixed-point calculation on hardware side).
                    mask_real = (np.real(sequence) > 2)
                    mask_imag = (np.imag(sequence) > 2)
                    mask_real_sum = np.sum(mask_real)
                    mask_imag_sum = np.sum(mask_imag)
                    assert(mask_real_sum == 0)
                    assert(mask_imag_sum == 0)


####################################################################################################

# Helper function to compare each element of two tensors.
def compare_tensor_entries(input_tensor, comparison_tensor, epsi):
        
        assert(input_tensor.shape == comparison_tensor.shape)

        for row in range(input_tensor.shape[0]):
            for col in range(input_tensor.shape[1]):
                print("at row %d, col %d" % (row, col))
                assert(torch.abs(input_tensor[row][col] - comparison_tensor[row][col]) < epsi)


####################################################################################################

def test_transform_sequences_mag():

    input_sequence = fixtures.return_complex_tensor()
    [num_frames, num_freqs] = input_sequence.shape
    spectra = ['S12']
    transforms = ['mag']

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=False,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)

    calculated_features = ptf.transform_sequences([input_sequence, torch.randn((num_frames, num_freqs))])
    compare_tensor_entries(calculated_features, fixtures.return_precalculated_mag_features(), 1e-3)


####################################################################################################

def test_transform_sequences_phase():

    input_sequence = fixtures.return_complex_tensor()
    [num_frames, num_freqs] = input_sequence.shape
    spectra = ['S12']
    transforms = ['phase']

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=False,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)

    calculated_features = ptf.transform_sequences([input_sequence, torch.randn((num_frames, num_freqs))])
    compare_tensor_entries(calculated_features, fixtures.return_precalculated_phase_features(), 1e-3)


####################################################################################################

def test_transform_sequences_mag_db():

    input_sequence = fixtures.return_complex_tensor()
    [num_frames, num_freqs] = input_sequence.shape
    spectra = ['S12']
    transforms = ['mag_db']

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=False,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)

    calculated_features = ptf.transform_sequences([input_sequence, torch.randn((num_frames, num_freqs))])
    compare_tensor_entries(calculated_features, fixtures.return_precalculated_mag_db_features(), 1e-3)


####################################################################################################

def test_transform_sequences_mag_delta():

    input_sequence = fixtures.return_complex_tensor()
    [num_frames, num_freqs] = input_sequence.shape
    spectra = ['S12']
    transforms = ['mag_delta']

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=False,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)

    calculated_features = ptf.transform_sequences([input_sequence, torch.randn((num_frames, num_freqs))])
    compare_tensor_entries(calculated_features, fixtures.return_precalculated_mag_delta_features(), 1e-3)


####################################################################################################

def test_transform_sequences_phase_delta():

    input_sequence = fixtures.return_complex_tensor()
    [num_frames, num_freqs] = input_sequence.shape
    spectra = ['S12']
    transforms = ['phase_delta']

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=False,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)

    calculated_features = ptf.transform_sequences([input_sequence, torch.randn((num_frames, num_freqs))])
    compare_tensor_entries(calculated_features, fixtures.return_precalculated_phase_delta_features(), 1e-3)


####################################################################################################

def test_transform_sequences_concat_x4():
    
    input_sequence = fixtures.return_complex_tensor()
    [num_frames, num_freqs] = input_sequence.shape
    spectra = ['S12','S12','S12','S12']
    transforms = ['mag','phase','mag_db','mag_delta']

    target_features = torch.cat((fixtures.return_precalculated_mag_features(), 
                               fixtures.return_precalculated_phase_features(),
                               fixtures.return_precalculated_mag_db_features(),
                               fixtures.return_precalculated_mag_delta_features()),
                               dim=1)

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=False,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)

    calculated_features = ptf.transform_sequences([input_sequence, torch.randn((num_frames, num_freqs))])
    compare_tensor_entries(calculated_features, target_features, 1e-3)


####################################################################################################

def test_transform_sequences_concat():
    
    input_sequence = fixtures.return_complex_tensor()
    [num_frames, num_freqs] = input_sequence.shape
    spectra = ['S12','S12','S32','S12','S32','S12','S12','S32']
    transforms = ['mag','phase','mag_db','mag_delta','phase_delta', 'mag_delta','mag','phase_delta']

    target_features = torch.cat((fixtures.return_precalculated_mag_features(), 
                                 fixtures.return_precalculated_phase_features(),
                                 fixtures.return_precalculated_mag_db_features(),
                                 fixtures.return_precalculated_mag_delta_features(),
                                 fixtures.return_precalculated_phase_delta_features(), 
                                 fixtures.return_precalculated_mag_delta_features(), 
                                 fixtures.return_precalculated_mag_features(), 
                                 fixtures.return_precalculated_phase_delta_features()), 
                                 dim=1)

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=False,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)

    calculated_features = ptf.transform_sequences([input_sequence, input_sequence])
    compare_tensor_entries(calculated_features, target_features, 1e-3)


####################################################################################################


def test_output_shapes():

    spectra = ['S12','S12','S32']
    transforms = ['mag','phase','mag_db']

    num_frames = 100
    freq_start_index = 0
    all_freq_steps = torch.linspace(start=0, end=128, steps=128).long()
    
    for step_index in range(all_freq_steps.shape[0]):
        num_freqs = all_freq_steps[step_index].item()
    
        sequence_list = [torch.randn(num_frames, num_freqs), torch.randn(num_frames, num_freqs)]

        ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                                transforms=transforms,
                                                                is_normalized=False,
                                                                freq_start_index=0,
                                                                freq_stop_index=num_freqs)

        target_dims = (num_frames, num_freqs*len(spectra))
        calculated_features = ptf.transform_sequences(sequence_list)

        assert(target_dims == calculated_features.shape)


####################################################################################################


def test_normalization():

    spectra = ['S12','S12','S32','S12','S12','S32']
    transforms = ['mag','phase','mag_db','mag_delta','phase_delta','mag_delta']

    num_frames = 100
    num_freqs = 128
    sequence_list = [torch.randn(num_frames, num_freqs)*10, torch.randn(num_frames, num_freqs)*10]

    # Make sure negative values are present in the test vector.
    assert(torch.min(sequence_list[0]) < 0)
    assert(torch.min(sequence_list[1]) < 0)
    # Make sure, positive values > 1 are present in vector.
    assert(torch.max(sequence_list[0]) > 1)
    assert(torch.max(sequence_list[1]) > 1)

    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra,
                                                            transforms=transforms,
                                                            is_normalized=True,
                                                            freq_start_index=0,
                                                            freq_stop_index=num_freqs)
    calculated_features = ptf.transform_sequences(sequence_list)
    min_value = torch.min(calculated_features) # Should be near 0.
    max_value = torch.max(calculated_features) # Should be near 1.

    # Check for correct normalization between [1, 0]
    assert(min_value == 0.0) 
    assert(max_value == 1.0)
    


####################################################################################################

if __name__ == "__main__":
    test_calculate_delta_sequence()
    test_verify_feature_selection()
    test_validate_corpus_content()
    test_transform_sequences_mag()
    test_transform_sequences_phase()    
    test_transform_sequences_mag_db()    
    test_transform_sequences_mag_delta()    
    test_transform_sequences_phase_delta()
    test_transform_sequences_concat()
    test_output_shapes()
    test_normalization()