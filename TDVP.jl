"""
This code is designed for time evolution of the Ising model (long or short) after a quantum quench. The code consists functions for
DMRG which initializes the system in the GS at a certaion point in parameter space. The system is then suddenly quenched to a final
point in parameter space and is evolved using the TDVP functions.
"""

"""
This code needs fixing as we want to implement one site TDVP where implementing the QR decomposition seems an issue in julia.
PROPOSAL 1 : find the bug and fix it
PROPOSAL 2 : call QR decomposition from python using pycall
"""

using LinearAlgebra
using BenchmarkTools
using StaticArrays
using TensorOperations
using KrylovKit
using Arpack
using PyCall
np = pyimport("numpy")
using DelimitedFiles
using Printf
using JLD
using ExponentialUtilities
using Plots


"""
This function takes the parameters N (lattice length), chi (bond dimension), and d (physical dimension) and
unnormalized random MPS.
"""
function initial_psi!(N,chi,d,M_set)

    M_set[1] = rand(1,d,chi)
    M_set[N] = rand(chi,d,1)

    @inbounds for i in 2:N-1

    M_set[i] = rand(chi,d,chi)
    end

    B_set = copy(M_set)


    return M_set, B_set
end

#compact_SVD, in case we need SVD from linalg this is a thin form of SVD. It is executed by calling numpy by Pycall
function compact_svd(mat)

    U,S,V = np.linalg.svd(mat)

    if size(mat)[1] > size(mat)[2]
        U = U[:,1:size(mat)[2]]

    elseif size(mat)[1] < size(mat)[2]
        V = V[1:size(mat)[1],:]
    end
    return U,S,V
end

#This function takes the unnormalized MPS set and right normalizes it.

"""
While performing the right(or left) normalization of the set of MPS it is cruicial to keep every step normalized
which here is executed by dividing the tensor B[i-1] by norm(F.S). Failing to do so will result in stacking of
the B tensors in every step resulting in a very large number(larger than the device can handle) after a certain
number of steps and breaking down of the algorithm. This was the reason why this function (before proper
normalization) worked for smaller values of N but failed after N exceeded a certain number.
"""

function right_normalize_SVD(B_set,N)

    @inbounds for i in reverse(1:N)

        left_index = size(B_set[i])[1]
        center_index = size(B_set[i])[2]
        right_index = size(B_set[i])[3]

        Mat = reshape(B_set[i],(left_index,center_index*right_index))
        F = svd(Mat)

        B_set[i] = reshape(F.Vt,(:,center_index,right_index))

        if i != 1

            US = F.U*diagm(F.S)
            @tensor B_set[i-1][chi_l,sig,chi_r1] := B_set[i-1][chi_l,sig,chi_r]*US[chi_r,chi_r1]/norm(F.S)
        end
    end

    return B_set
end

"""
function converting power law to sum of exponentials : 1/r^k = Sum_n (x_n * lambda_n^k)
implemented as in arXiv:0804.3976
"""

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

"""
creating the MPO representation of Hamiltonians
implemented as in  arXiv:0804.2504
"""

function Hamiltonian_LR_Ising_Kac(a,h,N,n,Ham)

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


"""
several tensor contraction routines
refer to https://www.tensors.net/tutorial-1 for more on efficient tensor contractions
"""

function contract_right(R,W,B)

    @tensor R1[a1,a2,b2,b3] := conj(B)[a1,a2,c3]*R[c3,b2,b3]
    @tensor R1[a1,b1,b4,a4] := R1[a1,c2,c3,a4]*W[b1,c3,c2,b4]
    @tensor R1[a1,a2,b1] := R1[a1,a2,c3,c4]*B[b1,c3,c4]

    return R1
end

function contract_left(L,W,A)
    #print(size(L),size(conj(A)))
    @tensor L1[a3,a2,b2,b3] := conj(A)[c1,a2,a3]*L[c1,b2,b3]
    #print(size(L1),size(W))
    @tensor L1[a1,b2,b4,a4]:= L1[a1,c2,c3,a4]*W[c3,b2,c2,b4]
    #print(size(L1),size(A))
    @tensor L1[a1,a2,b4]:= L1[a1,a2,c3,c4]*A[c4,c3,b4]
    #print(size(L1))

    return L1
end

function contract_Heff_two(L,W1,W2,R)

    @tensor He2[a1,a3,b3,b2,a4,b4] := W1[a1,c2,a3,a4]*W2[c2,b2,b3,b4]
    @tensor He2[a1,b2,b3,b4,a3,b5,b6] := L[a1,c2,a3]*He2[c2,b2,b3,b4,b5,b6]
    @tensor He2[a1,a2,a3,b1,a5,a6,a7,b3] := He2[a1,a2,a3,c4,a5,a6,a7]*R[b1,c4,b3]
    shp = size(He2)
    He2 = reshape(He2,(shp[1]*shp[2]*shp[3]*shp[4],shp[5]*shp[6]*shp[7]*shp[8]))

    return He2
end

function contract_Heff_one(L,W,R)

   @tensor He1[a1,b3,b2,a3,b4]:= L[a1,c2,a3]*W[c2,b2,b3,b4]
    @tensor He1[a1,a2,b1,a4,a5,b3] := He1[a1,a2,c3,a4,a5]*R[b1,c3,b3]
    shp = size(He1)
    He1 = reshape(He1,(shp[1]*shp[2]*shp[3],shp[4]*shp[5]*shp[6]))

    return He1
end

function contract_Keff(L,R)

    @tensor Ke[a1,b1,a3,b3]:= L[a1,c2,a3]*R[b1,c2,b3]
    shp = size(Ke)
    Ke = reshape(Ke,(shp[1]*shp[2],shp[3]*shp[4]))

    return Ke
end

function contract_left_noop(L,A)

    @tensor L1[a3,a2,b2] := conj(A)[c1,a2,a3]*L[c1,b2]
    @tensor L1[a1,b3] := L1[a1,c2,c3]*A[c3,c2,b3]

    return L1
end


function contract_left_nonmpo(L,W,A)

    @tensor L1[a3,a2,b2] := conj(A)[c1,a2,a3]*L[c1,b2]
    @tensor L1[a1,b2,a3]:= L1[a1,c2,a3]*W[c2,b2]
    @tensor L1[a1,b3]:= L1[a1,c2,c3]*A[c3,c2,b3]

    return L1
end

#function takes the matrix T, does its SBD decomposition and then truncates it by a factor of chi

function svd_truncate(T,chi)

    F = svd(T)

    if length(F.S) > chi

        S = F.S[1:chi]/norm(F.S)
        U = F.U[:,1:chi]
        V = F.Vt[1:chi,:]

    else

        S = F.S/norm(F.S)
        U = F.U
        V = F.Vt
    end

    return U,S,V
end

#This function initializes the L and R tensors before the DMRG sweep

function Initialize(N,M_set,B_set,Ham)

    # get the list for R and initialize boundary R
    R = Array{Any,1}(undef,N+1)
    R[N+1] = ones(1,1,1)

    B = right_normalize_SVD(B_set,N)

    @inbounds for i in reverse(2:N)
        R[i] = contract_right(R[i+1],Ham[i],B[i])
    end

    # get the list for L and initialize boundary L
    L = Array{Any,1}(undef,N+1)
    L[N+1] = ones(1,1,1)

    #get the list for A
    A = Array{Any,1}(undef,N)

    #get initial M
    M = M_set[1]

    return A,B,L,R,M
end

"""
this is a short but a cruicial function which allows us to access the L[0] element of list L of size N+1.
unlike python the element count in julia starts from 1 and it seems(for now) that the periodic indexing isn't
as flexible as in python, so for a vector L of size N+1, if we call for the element L[0] it gives us error
rather than giving us the N+1_th element. This functions remidies that issue by mapping 0 to N+1 while all other
numbers from 1 to N to themselves.
"""

function indx_L(i,N)

    if i == 0
        return N+1
    else
        return i
    end
end

#These functions performs the left and right DMRG sweeps

function right_sweep_DMRG_two_site!(L,R,Ham,M,A,B,N)

    @inbounds for i in 1:N-1

        #create T matrix
        @tensor T[a1,a2,b2,b3] := M[a1,a2,c3]*B[i+1][c3,b2,b3]

        #reshape T into a vector
        shp_T = size(T)
        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]))

        #create He2 matrix
        He2 = contract_Heff_two(L[indx_L(i-1,N)],Ham[i],Ham[i+1],R[i+2])
        He2[diagind(He2)] .= real(diag(He2))

        #find the local GS energy and GS vector
        eig_val,eig_vec, info =  eigsolve(Hermitian(He2), 1, :SR)

        #reshape back to matrix
        vec = reshape(eig_vec[1],(shp_T[1]*shp_T[2],shp_T[3]*shp_T[4]))

        #SVD and truncation
        U,S,V = svd_truncate(vec,chi)

        #reshape U into A (which is a left normalized MPS)
        A[i] = reshape(U,(shp_T[1],shp_T[2],:))

        #get M from S and V
        V = reshape(V,(:,shp_T[3],shp_T[4]))
        S = diagm(S)
        @tensor M[a1,b2,b3] := S[a1,c2]*V[c2,b2,b3]

        #create L[i]
        L[indx_L(i,N)] = contract_left(L[indx_L(i-1,N)],Ham[i],A[i])

        if i != N-1

            R[i+2] = 0.0
        end
    end
    return A,B,L,R,M
end

function left_sweep_DMRG_two_site!(L,R,Ham,M,A,B,N)

    @inbounds for j in reverse(2:N)

        #create T matrix
        @tensor T[a1,a2,b2,b3] := A[j-1][a1,a2,c3]*M[c3,b2,b3]

        #reshape T into vector
        shp_T = size(T)
        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]))

        #create He2 matrix
        He2 = contract_Heff_two(L[indx_L(j-2,N)],Ham[j-1],Ham[j],R[j+1])
        He2[diagind(He2)] .= real(diag(He2))

        #find the local GS energy and GS vector
        eig_val, eig_vec, info = eigsolve(Hermitian(He2),1, :SR)

        #reshape back to matrix
        vec = reshape(eig_vec[1],(shp_T[1]*shp_T[2],shp_T[3]*shp_T[4]))

        #SVD and truncation
        U,S,V = svd_truncate(vec,chi)

        #reshape V into B[j]
        B[j] = reshape(V,(:,shp_T[3],shp_T[4]))

        #get M from U and S
        U = reshape(U,(shp_T[1],shp_T[2],:))
        S = diagm(S)
        @tensor M[a1,a2,b2] := U[a1,a2,c3]*S[c3,b2]

        #create R[j]
        R[j] = contract_right(R[j+1],Ham[j],B[j])

        if j != 2

            #delete L[j-2]
            L[indx_L(j-2,N)] = 0
        end
    end

    return A,B,L,R,M
end

function right_sweep_DMRG_one_site!(L,R,Ham,M,A,B,N)


    @inbounds for i in 1:N

        shp_M = size(M)

        #create He2 matrix
        He1 = contract_Heff_one(L[indx_L(i-1,N)],Ham[i],R[i+1])
        He1[diagind(He1)] .= real(diag(He1))

        #find the local GS energy and GS vector
        eig_val,eig_vec, info =  eigsolve(Hermitian(He1), 1, :SR)

        #reshape back to matrix
        vec = reshape(eig_vec[1],(shp_M[1]*shp_M[2],shp_M[3]))

        #SVD and truncation
        U,S,V = svd_truncate(vec,chi)

        #reshape U into A (which is a left normalized MPS)
        A[i] = reshape(U,(shp_M[1],shp_M[2],:))

        #create L[i]
        L[indx_L(i,N)] = contract_left(L[indx_L(i-1,N)],Ham[i],A[i])

        if i != N

            #create M tenosr using SV and B[i+1]
            SV = diagm(S)*V
            @tensor M[a1,b2,b3] := SV[a1,c2]*B[i+1][c2,b2,b3]

            #delete R[i+1]
            R[i+1] = 0.0
        end
    end

    return A,B,L,R,M
end

function left_sweep_DMRG_one_site!(L,R,Ham,M,A,B,N)

    #E = Array{Any,1}(undef,N)
    @inbounds for i in reverse(1:N)

        shp_M = size(M)

        #create He2 matrix
        He1 = contract_Heff_one(L[indx_L(i-1,N)],Ham[i],R[i+1])
        He1[diagind(He1)] .= real(diag(He1))

        #find the local GS energy and GS vector
        eig_val,eig_vec, info =  eigsolve(Hermitian(He1), 1, :SR)

        #reshape back to matrix
        vec = reshape(eig_vec[1],(shp_M[1],shp_M[2]*shp_M[3]))

        #SVD and truncation
        U,S,V = svd_truncate(vec,chi)

        #reshape V into B[i] (which is a right normalized MPS)
        B[i] = reshape(V,(:,shp_M[2],shp_M[3]))

        #create R[i]
        R[i] = contract_right(R[i+1],Ham[i],B[i])

        if i != 1

            #create M tenosr using US and A[i-1]
            US = U*diagm(S)
            @tensor M[a1,a2,b2] := A[i-1][a1,a2,c3]*US[c3,b2]

            #delete L[i-1]
            L[indx_L(i-1,N)] = 0.0
        end
    end

    return A,B,L,R,M
end


#these functions performs left and right TDVP sweeps

function right_sweep_TDVP_two(dt,L,R,Ham,M,A,B)

    for i in 1:N-1

        #create T matrix
        @tensor T[a1,a2,b2,b3] := M[a1,a2,c3]*B[i+1][c3,b2,b3]

        #reshape T into a vector
        shp_T = size(T)
        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]))

        #create He2 matrix
        He2 = contract_Heff_two(L[indx_L(i-1,N)],Ham[i],Ham[i+1],R[i+2])
        He2[diagind(He2)] .= real(diag(He2))

        # exp(-i*dt*He2)*T,evolve T forward
        T = expv(-im*dt/2,Hermitian(He2),T)

        #reshape T into a matrix
        T = reshape(T,(shp_T[1]*shp_T[2],shp_T[3]*shp_T[4]))

        #SVD and truncate
        U,S,V = svd_truncate(T,chi)

        #reshape U into A
        A[i] = reshape(U,(shp_T[1],shp_T[2],:))

        #get M from S and V
        V = reshape(V,(:,shp_T[3],shp_T[4]))
        S = diagm(S)
        @tensor M[a1,b2,b3] := S[a1,c2]*V[c2,b2,b3]

        if i != N-1

            #create L[i]
            L[indx_L(i,N)] = contract_left(L[indx_L(i-1,N)],Ham[i],A[i])

            #reshape M into a vector
            shp_M = size(M)
            M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

            #create He1 matrix
            He1 = contract_Heff_one(L[indx_L(i,N)],Ham[i+1],R[i+2])

            #evolve M backward
            M =  expv(im*dt/2,Hermitian(He1),M)

            #reshape M into a three legged MPS
            M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

            #delete R[i+2]
            R[i+2] = 0.0
        end
    end
    return A,B,L,R,M
end


function left_sweep_TDVP_two(dt,L,R,Ham,M,A,B,N)

    for j in reverse(2:N)

        #create T matrix
        @tensor T[a1,a2,b2,b3] := A[j-1][a1,a2,c3]*M[c3,b2,b3]

        #reshape T into vector
        shp_T = size(T)
        T = reshape(T,(shp_T[1]*shp_T[2]*shp_T[3]*shp_T[4]))

        #create He2 matrix
        He2 = contract_Heff_two(L[indx_L(j-2,N)],Ham[j-1],Ham[j],R[j+1])
        He2[diagind(He2)] .= real(diag(He2))

        # exp(-i*dt*He2)*T,evolve T forward
        T = expv(-im*dt/2,Hermitian(He2),T)

        #reshape T into a matrix
        T = reshape(T,(shp_T[1]*shp_T[2],shp_T[3]*shp_T[4]))

        #SVD and truncate
        U,S,V = svd_truncate(T,chi)

        #reshape V into B[j]
        B[j] = reshape(V,(:,shp_T[3],shp_T[4]))

        #get M from U and S
        U = reshape(U,(shp_T[1],shp_T[2],:))
        S = diagm(S)
        @tensor M[a1,a2,b2] := U[a1,a2,c3]*S[c3,b2]

        if j != 2

            #create R[j]
            R[j] = contract_right(R[j+1],Ham[j],B[j])

            #reshape M into a vector
            shp_M = size(M)
            M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

            #create He1 matrix
            He1 = contract_Heff_one(L[indx_L(j-2,N)],Ham[j-1],R[j])

            #backward evolve M
            M = expv(im*dt/2,Hermitian(He1),M)

            #reshape M into three legged MPS
            M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

            #delete L[j-2]
            L[indx_L(j-2,N)] = 0.0

        end
    end

    return A,B,L,R,M
end

function right_sweep_TDVP_one(dt,L,R,Ham,M,A,B)

    @inbounds for i in 1:N

        shp_M = size(M)
        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        #create He1 matrix
        He1 = contract_Heff_one(L[indx_L(i-1,N)],Ham[i],R[i+1])
        He1[diagind(He1)] .= real(diag(He1))

        # exp(-i*dt*He1)*M,evolve M forward
        M = expv(-im*dt/2.0,Hermitian(He1),M)

        #reshape M into a three legged MPS
        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        #SVD and truncation
        mat_tem = reshape(M,(shp_M[1]*shp_M[2],shp_M[3]))
        U,S,V = svd_truncate(mat_tem,chi)

        #reshape U into A(left normalized MPS)
        A[i] = reshape(U,(shp_M[1],shp_M[2],:))

        #build L[i]
        L[indx_L(i,N)] = contract_left(L[indx_L(i-1,N)],Ham[i],A[i])

        if i != N

            SV = diagm(S)*V
            shp_SV = size(SV)

            C = reshape(SV,(shp_SV[1]*shp_SV[2]))

            #create Keff matrix
            Keff = contract_Keff(L[indx_L(i,N)],R[i+1])
            Keff[diagind(Keff)] .= real(diag(Keff))

            #evolve C backward
            C = expv(im*dt/2.0,Hermitian(Keff),C)

            #reshape C back into two legged vector
            C = reshape(C,(shp_SV[1],shp_SV[2]))

            #create M
            @tensor M[a1,b2,b3] := C[a1,c2]*B[i+1][c2,b2,b3]

            #delete R[i+1]
            R[i+1] = 0.0
        end
    end

    return A,B,L,R,M
end


function right_sweep_TDVP_one_QR(dt,L,R,Ham,M,A,B)

    @inbounds for i in 1:N

        shp_M = size(M)
        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        #create He1 matrix
        He1 = contract_Heff_one(L[indx_L(i-1,N)],Ham[i],R[i+1])
        He1[diagind(He1)] .= real(diag(He1))

        # exp(-i*dt*He1)*M,evolve M forward
        M = expv(-im*dt/2,Hermitian(He1),M)

        #reshape M into a three legged MPS
        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        #SVD and truncation
        mat_tem = reshape(M,(shp_M[1]*shp_M[2],shp_M[3]))
        QR_dec = qr(mat_tem)
        print(size(QR_dec.Q))
        #reshape U into A(left normalized MPS)
        A[i] = reshape(QR_dec.Q,(shp_M[1],shp_M[2],:))
        print(size(A[i]),size(Ham[i]),size(L[indx_L(i-1,N)]))
        #build L[i]
        L[indx_L(i,N)] = contract_left(L[indx_L(i-1,N)],Ham[i],A[i])

        if i != N


            C = QR_dec.R
            shp_C = size(C)

            C = reshape(C,(shp_C[1]*shp_C[2]))

            #create Keff matrix
            Keff = contract_Keff(L[indx_L(i,N)],R[i+1])
            Keff[diagind(Keff)] .= real(diag(Keff))

            #evolve C backward
            C = expv(im*dt/2,Hermitian(Keff),C)

            #reshape C back into two legged vector
            C = reshape(C,(shp_C[1],shp_C[2]))

            #create M
            @tensor M[a1,b2,b3] := C[a1,c2]*B[i+1][c2,b2,b3]

            #delete R[i+1]
            R[i+1] = 0.0
        end
    end

    return A,B,L,R,M
end



function left_sweep_TDVP_one(dt,L,R,Ham,M,A,B)

    @inbounds for i in reverse(1:N)

        shp_M = size(M)
        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        #create He1 matrix
        He1 = contract_Heff_one(L[indx_L(i-1,N)],Ham[i],R[i+1])
        He1[diagind(He1)] .= real(diag(He1))

        # exp(-i*dt*He1)*M,evolve M forward
        M = expv(-im*dt/2.0,Hermitian(He1),M)

        #reshape M into a three legged MPS
        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        #SVD and truncation
        mat_tem = reshape(M,(shp_M[1],shp_M[2]*shp_M[3]))
        U,S,V = svd_truncate(mat_tem,chi)

        #reshape V into B[i] (which is a right normalized MPS)
        B[i] = reshape(V,(:,shp_M[2],shp_M[3]))

        #create R[i]
        R[i] = contract_right(R[i+1],Ham[i],B[i])

        if i != 1

            US = U*diagm(S)
            shp_US = size(US)

            C = reshape(US,(shp_US[1]*shp_US[2]))

            #create Keff matrix
            Keff = contract_Keff(L[indx_L(i-1,N)],R[i])
            Keff[diagind(Keff)] .= real(diag(Keff))

            #evolve C backward
            C = expv(im*dt/2.0,Hermitian(Keff),C)

            #reshape C back into two legged vector
            C = reshape(C,(shp_US[1],shp_US[2]))

            #create M
            @tensor M[a1,a2,b2] := A[i-1][a1,a2,c3]*C[c3,b2]

            #delete L[i-1]
            L[indx_L(i-1,N)] = 0.0
        end
    end

    return A,B,L,R,M
end


function left_sweep_TDVP_one_QR(dt,L,R,Ham,M,A,B)

    @inbounds for i in reverse(1:N)

        shp_M = size(M)
        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        #create He1 matrix
        He1 = contract_Heff_one(L[indx_L(i-1,N)],Ham[i],R[i+1])
        He1[diagind(He1)] .= real(diag(He1))

        # exp(-i*dt*He1)*M,evolve M forward
        M = expv(-im*dt/2,Hermitian(He1),M)

        #reshape M into a three legged MPS
        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        #SVD and truncation
        mat_tem = reshape(M,(shp_M[1],shp_M[2]*shp_M[3]))
        LQ_dec = lq(mat_tem)

        #reshape V into B[i] (which is a right normalized MPS)
        B[i] = reshape(LQ_dec.Q,(:,shp_M[2],shp_M[3]))

        #create R[i]
        R[i] = contract_right(R[i+1],Ham[i],B[i])

        if i != 1

            C = LQ_dec.L
            shp_C = size(C)

            C = reshape(C,(shp_C[1]*shp_C[2]))

            #create Keff matrix
            Keff = contract_Keff(L[indx_L(i-1,N)],R[i])
            Keff[diagind(Keff)] .= real(diag(Keff))

            #evolve C backward
            C = expv(im*dt/2,Hermitian(Keff),C)

            #reshape C back into two legged vector
            C = reshape(C,(shp_C[1],shp_C[2]))

            #create M
            @tensor M[a1,a2,b2] := A[i-1][a1,a2,c3]*C[c3,b2]

            #delete L[i-1]
            L[indx_L(i-1,N)] = 0.0
        end
    end

    return A,B,L,R,M
end


#calculates single site magnetization
function mag_single_site(N,psi)

    sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
    sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];

    X = ones(1,1)
    Y = copy(X)

    @inbounds for i in 1:49
        X = contract_left_noop(X,psi[i])
    end

    @inbounds for i in 50
         X = contract_left_nonmpo(X,2.0*sX,psi[i])
    end

    @inbounds for i in 51:N
        X = contract_left_noop(X,psi[i])
    end

    @tensor X[a1,b2] := X[a1,c2]*Y[c2,b2]

    return X[1]
end
"""
This function performs DMRG to get GS, Quenches it, evolves using TDVP, calculates magnetization in each time slices and saves
it in a txt file
"""

function TDVP(N,chi,d,param,h1,n,dt,num_DMRG,num_TDVP,mag_bag)

    a,h = param

    M_set = Array{Any,1}(undef,N)
    M_set, B_set = initial_psi!(N,chi,d,M_set)

    Ham = Array{Any,1}(undef,N)
    Ham = Hamiltonian_LR_Ising(a,h,N,n,Ham)

    A,B,L,R,M = Initialize(N,M_set,B_set,Ham)

    for i in 1:num_DMRG
        A,B,L,R,M = right_sweep_DMRG_one_site!(L,R,Ham,M,A,B,N)
        A,B,L,R,M = left_sweep_DMRG_one_site!(L,R,Ham,M,A,B,N)
    end

    Ham1 = Array{Any,1}(undef,N)
    Ham1 = Hamiltonian_LR_Ising(a,h1,N,n,Ham1)

    for i in 1:num_TDVP

        psi = Array{Any,1}(undef,N)

        A,B,L,R,M = right_sweep_TDVP_one(dt,L,R,Ham1,M,A,B)

        A,B,L,R,M = left_sweep_TDVP_one(dt,L,R,Ham1,M,A,B)

        psi[1] = M

        for j in 2:N
            psi[j] = B[j]
        end

        magnetization = mag_single_site(N,psi)
        mag_bag[i] = [i*dt,real(magnetization)]

        print(i,real(magnetization),"\n")
    end

    open("NISHAN.txt","w") do io
        writedlm(io,mag_bag,',')
    end
end
