using LinearAlgebra

"""
This code consists the long and short range Ising Hamiltonians MPO (with and without Kac normalization) 
implemented as in  arXiv:0804.2504
"""

function Hamiltonian_LR_Ising_kac(a,h,N,n,Ham)

    #basic matrices
    sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
    sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];

    Kac = 0.0

    @inbounds for i in 1:N
        Kac += 1.0/(i)^a
    end

    x, lambda = power_law_to_exp(a,N,n)

    #building the local bulk MPO

    H = zeros(n+2,n+2,2,2)

    H[1,1,:,:] = sI; H[n+2,n+2,:,:] = sI; H[n+2,1,:,:] = -h*sZ


    @inbounds for i in 2:n+1
        H[i,1,:,:] = (x[i-1]/Kac)*sX
        H[i,i,:,:] = lambda[i-1]*sI
    end

    @inbounds for j in 2:n+1
        H[n+2,j,:,:] = -lambda[j-1]*sX
    end

    #building the boundary  MPOs
    HL = zeros(1,n+2,2,2)
    HL[1,:,:,:] = H[n+2,:,:,:]
    HR = zeros(n+2,1,2,2)
    HR[:,1,:,:] = H[:,1,:,:]

    #put the hamiltonian in a list so that it can be iteteratively recuperated

    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end


function Hamiltonian_LR_Ising(a,h,N,n,Ham)

    #basic matrices
    sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
    sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];

    x, lambda = power_law_to_exp(a,N,n)

    #building the local bulk MPO

    H = zeros(n+2,n+2,2,2)

    H[1,1,:,:] = sI; H[n+2,n+2,:,:] = sI; H[n+2,1,:,:] = -h*sZ


    @inbounds for i in 2:n+1
        H[i,1,:,:] = x[i-1]*sX
        H[i,i,:,:] = lambda[i-1]*sI
    end

    @inbounds for j in 2:n+1
        H[n+2,j,:,:] = -lambda[j-1]*sX
    end

    #building the boundary  MPOs
    HL = zeros(1,n+2,2,2)
    HL[1,:,:,:] = H[n+2,:,:,:]
    HR = zeros(n+2,1,2,2)
    HR[:,1,:,:] = H[:,1,:,:]

    #put the hamiltonian in a list so that it can be iteteratively recuperated

    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end


function Hamiltonian_Ising(h,N,Ham)

    #basic matrices
    sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
    sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];

    #building the local bulk MPO

    H = zeros(3,3,2,2)

    H[1,1,:,:] = sI; H[3,3,:,:] = sI; H[3,1,:,:] = -h*sZ
    H[2,1,:,:] = sX; H[3,2,:,:] = -sX


    #building the boundary  MPOs
    HL = zeros(1,3,2,2)
    HL[1,:,:,:] = H[3,:,:,:]
    HR = zeros(3,1,2,2)
    HR[:,1,:,:] = H[:,1,:,:]

    #put the hamiltonian in a list so that it can be iteteratively recuperated

    Ham[1] = HL
    Ham[N] = HR

    @inbounds for i in 2:N-1
        Ham[i] = H
    end

    return Ham
end
