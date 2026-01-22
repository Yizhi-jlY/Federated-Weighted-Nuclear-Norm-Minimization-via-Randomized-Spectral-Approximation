
function dist = subspace_distance(U_true, U_approx)
    P = U_approx * U_approx';
    dist = norm((eye(size(P)) - P) * U_true, 'fro');
end
