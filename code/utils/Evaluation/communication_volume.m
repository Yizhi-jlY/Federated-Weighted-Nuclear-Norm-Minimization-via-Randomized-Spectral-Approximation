
function volume = communication_volume(data_sizes)
    total_bytes = 0;
    for i = 1:length(data_sizes)
        sz = data_sizes{i};
        total_bytes = total_bytes + prod(sz) * 8;
    end
    volume = total_bytes / (1024^2);
end


