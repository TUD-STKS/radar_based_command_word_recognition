function [all_results, hyperparameters_list] = load_result_files(path_to_files, ...
    results_metrics, train_or_test)
    % Loads the results from the results text files.
    % @param path_to_file (str): full file path to the result files.
    % @param hyperparameters (cell(str)): cell array containing the names of
    % the hyperparameters.
    % @param results_metrics (cell(str)): names of the metrics written
    % out during training or testing.
    % @param train_or_test (char): specifiy whether the results are for
    % training ('train') or testing ('test')
    % @return results (struct): struct containing the collected file content.

    if(~ischar(train_or_test))
        fprintf("Error: variable train_or_test needs to be a char array (' ')");
        return
    end

    addpath("C:\Programming\MATLAB\common_functions") % needed for sort_nat()
    % Locate all file infos.
    directory = dir(sprintf("%s\\lstm*", path_to_files));
    try
        file_names = sort_nat({directory.name});
    catch ME
        if(strcmp(ME.identifier,"MATLAB:UndefinedFunction"))
            fprintf("Error: function sort_nat() (Matlab exchange) not found.\n");
            return
        else
            rethrow(ME)
        end
    end
    
    % Select only files with the correct ending keyword.
    valid_file_indices = endsWith(file_names, strcat(train_or_test, '.txt'));
    file_names = file_names(valid_file_indices);
    folder_names = sort_nat({directory.folder});
    folder_names = folder_names(valid_file_indices);
    
    num_files = numel(file_names);
    all_results = cell(1, num_files);
    
    if(num_files <= 0)
        fprintf("There were no files found on path:\n");
        fprintf("%s\n.", path_to_files);
        hyperparameters_list = [];
        return ;
    end
    

    for file_index = 1:num_files
        file_name = sprintf("%s\\%s", folder_names{file_index}, file_names{file_index});
        fprintf("+++ Loading file %s +++\n", file_name);
        
        fid = fopen(file_name,'r');
        results = struct(); % stores all results.

        num_result_metrics = numel(results_metrics);
        for field_index = 1:num_result_metrics
           results.(results_metrics{field_index}) = [];
        end

        % Decode the hyperparameters.
        header1 = fgetl(fid); % names of the hyperparameters.
        hyperparameter_names = split(header1, ' ');
        hyperparameters_list = [];
        header2 = fgetl(fid); % values for each hyperparameter.
        header2 = replace(header2, 'False', '1');
        header2 = replace(header2, 'True', '1');
        str = textscan(header2, '%s', 'Delimiter', ' ');

        % Store the hyperparameters with their respective value.
        return_hp_list_index = 1;
        for hp_index = 1:numel(hyperparameter_names)
            if(~isempty(hyperparameter_names{hp_index}))
                results.(hyperparameter_names{hp_index}) = str2num(str{1}{hp_index});
                hyperparameters_list{return_hp_list_index} = hyperparameter_names{hp_index};
                return_hp_list_index = return_hp_list_index + 1;
            end
        end

        % Decode the numeric values from the training results.
        numeric_values = [];

        while(~feof(fid))   
            line = fgetl(fid);

            % Find the hyperparameter headings.
            headline_found = false;
            % Detect first result metric name.
            if(strcmp(line, results_metrics{1}))
                fprintf("Found result metric %s\n", results_metrics{1});
                headline_found = true;
                index = 1;
                numeric_values = [];
            else
                % Check for remaining result metric names.
                for result_index = 2:num_result_metrics
                    if(strcmp(line, results_metrics{result_index}))
                        fprintf("Found result metric %s\n", results_metrics{result_index});
                        headline_found = true;
                        results.(results_metrics{result_index-1}) = numeric_values;
                        index = 1;
                        numeric_values = [];
                    end
                end
            end
            
            % If no headline was found, read out the numeric value.
            if(~headline_found)
                if(feof(fid))
                    numeric_values(index) = str2num(line);
                    results.(results_metrics{end}) = numeric_values;
                else
                    numeric_values(index) = str2num(line);
                    index = index + 1;
                end
            else
                headline_found = false; % reset flag.
            end
        end
        all_results{file_index} = results;
        fclose(fid);
    end
end


