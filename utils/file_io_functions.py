import numpy as np

def save_results_to_file(hyperparameters, history, file_name=None, append=False):
    ''' Save the training results to file.

    Args:
        hyperparameters (dict()): dictionary containing the hyperparameters.
        history (dict()): dictionary containing the training history.
                          (key : str, value : ndarray or tensor)
        file_name (str): file name. If None, a default name is provided.
        append (bool): Define whether to append new output to an existing file or create a new one.
        
    Returns:
        None. 

    '''

    # If the file name is not specified, a default name is given.
    if file_name is None:
        file_name = "lstm_optimization_results.txt"
    
    try:
        if(append):
            file = open(file_name, 'a')
        else:
            file = open(file_name, 'w')

    # Save the keys as one row.
        results_string = ''
        for key in hyperparameters:
            results_string = results_string + ' ' + key
    
        results_string = results_string + '\n'
        file.write(results_string)

        # Save the corresponding values as the second row.
        results_string = ''
        for key in hyperparameters:
            results_string = results_string + ' ' + str(hyperparameters[key])
    
        results_string = results_string + '\n'
        file.write(results_string)

        # save any additional vectors of losses etc.    
        for key in history:
            file.write(key + "\n")
            try:
                np.savetxt(file, np.asarray(history[key]), fmt='%1.4f')
            except ValueError:
                # Catch 0-d-Error, convert to 1-d numpy array and try again.
                value = np.array([history[key]])
                np.savetxt(file, np.asarray(value), fmt='%1.4f')
            except:
                print("Error while writing key %s" % (key))

        file.close()

    except IOError:
        print("Error: Failed to open file.")    

    

#############################################################

def log_message(message, log_file_name=None):

    if log_file_name is not None:
        try:
            with open(log_file_name, 'a') as f:
                print(message, file=f)
        except IOError:
            print("Error: Failed to open file.")            
    else:
        print(message)