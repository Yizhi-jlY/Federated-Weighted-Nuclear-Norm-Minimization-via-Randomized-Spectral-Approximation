function results_list = load_results_from_files(result_files, directory)
results_list = {};
if isempty(result_files)
    return;
end
for k = 1:length(result_files)
    filename = result_files(k).name;
    algo_name = extractBetween(filename, 'result_', '.mat');
    if ~isempty(algo_name)
        algo_name = algo_name{1};
        fprintf('loading: %s\n', filename);
        loaded_data = load(fullfile(directory, filename));
        if isfield(loaded_data, 'result')
            loaded_data.result.name = algo_name;
            results_list{end+1} = loaded_data.result;
        end
    end
end
end
