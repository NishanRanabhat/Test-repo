using LinearAlgebra


function power_law_to_exp(a,N,n)

    F = Array{Float64,1}(undef,N)

   @inbounds for k in 1:N
        F[k] = 1/k^a
    end

    M = zeros(N-n+1,n)

    @inbounds for j in 1:n
        @inbounds for i in 1:N-n+1
            M[i,j] = F[i+j-1]
        end
    end

    F1 = qr(M)

    Q1 = F1.Q[1:N-n,1:n]
    Q1_inv = pinv(Q1)
    Q2 = F1.Q[2:N-n+1,1:n]

    V = Q1_inv*Q2

    lambda = real(eigvals(V))

    lam_mat = zeros(N,n)

    @inbounds for i in 1:length(lambda)
        @inbounds for k in 1:N
            lam_mat[k,i] = lambda[i]^k
        end
    end

    x = lam_mat\F

    return x, lambda
end
