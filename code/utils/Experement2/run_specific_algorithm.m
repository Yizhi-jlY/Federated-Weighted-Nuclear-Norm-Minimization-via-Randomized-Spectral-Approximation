function result = run_specific_algorithm(algo_name, data, mask, parameters)
switch algo_name
    case 'factGDNew',       result = factGDNew(data, mask, parameters);
    case 'altGDMin_T',      result = altGDMin_T(data, mask, parameters);
    case 'altMinPrvt_T',    result = altMinPrvt_T(data, mask, parameters);
    case 'SVT_Rand',        result = SVT_Rand(data, mask, parameters);
    case 'FedSVT_MC',       result = FedSVT_MC(data, mask, parameters);
    case 'FedWNNM_MC',      result = FedWNNM_MC(data, mask, parameters);
    otherwise
        error('none: "%s"。', algo_name);
end
end

