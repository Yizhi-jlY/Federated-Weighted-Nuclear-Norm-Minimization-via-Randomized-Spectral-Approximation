function [algo_path, folder_name] = get_algorithm_path(algo_name, codeFolderPath)
% Get the code path for an algorithm based on its name.
switch algo_name
    case {'AltGD', 'altGDMinCntrl_T', 'altGDMin_T', 'altMinCntrl_T', 'altMinParfor_T', 'altMinPrvt_T', 'factGDNew'}
        folder_name = 'AltGD';
    case 'FedSVT_MC'
        folder_name = 'Fed_SVT';
    case 'FedWNNM_MC'
        folder_name = 'Fed_WNNM';
    case 'SVP_MC'
        folder_name = 'SVP';
    case 'SVT'
        folder_name = 'SVT';
    case 'SVT_Rand'
        folder_name = 'SVT_rand';
    case 'WNNM_MC'
        folder_name = 'WNNM';
    otherwise
        warning('No folder mapping found for %s.', algo_name);
        folder_name = '';
end

if ~isempty(folder_name)
    algo_path = fullfile(codeFolderPath, folder_name);
else
    algo_path = '';
end
end