
function err = singular_value_error(S_true, S_approx)
    err = norm(diag(S_true) - diag(S_approx), 2) / norm(diag(S_true), 2);
end