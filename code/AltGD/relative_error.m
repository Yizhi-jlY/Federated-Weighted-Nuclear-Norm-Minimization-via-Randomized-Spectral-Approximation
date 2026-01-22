function err = relative_error(A_hat, L)
    err = norm(A_hat - L, 'fro') / norm(L, 'fro');
end
