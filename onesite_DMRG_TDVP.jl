"""
This code consists of both DMRG-TDVP routine. Here we entirely use the Lanczos
routines for eigensolver and exponential solver included in the code rather
than any external packages. It is observed that the efficiency of the code is equal to
or even slower than the one that uses external exponentialsolver and eigensolver
for lower bond dimensions, however this code is orders of magnitude faster than
the one using external packages for higher bond dimension, which is our target.
The code has some issues however:
1) The convergence of DMRG part isn't as good as the one that uses external packages
2) The code currently uses krylov dimension as an external parameter, the code will be
smoother if the Lanczos algorithm has a convergence check mechanism. Different articles
proposes different conditions for convergence.
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



function Hamiltonian_LR_Ising(a,h,N,n,Ham)

    #basic matrices
    sX = 0.5*[0 1; 1 0]; sY = 0.5*[0 -im; im 0];
    sZ = 0.5*[1 0; 0 -1]; sI = [1 0; 0 1];

    Kac = 0.0
    for i in 1:N
        Kac += (N-i)/i^a
    end

    Kac = Kac/(N-1)

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


#our ISING
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

    @tensor L1[a3,a2,b2,b3] := conj(A)[c1,a2,a3]*L[c1,b2,b3]
    @tensor L1[a1,b2,b4,a4]:= L1[a1,c2,c3,a4]*W[c3,b2,c2,b4]
    @tensor L1[a1,a2,b4]:= L1[a1,a2,c3,c4]*A[c4,c3,b4]

    return L1
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

function MpoToMpsOneSite(L,W,R,M)

    M = reshape(M,(size(L,3),size(W,4),size(R,3)))

    @tensor fin[l1,w3,r1] := L[l1,c1,c2]*M[c2,c3,c5]*W[c1,c4,w3,c3]*R[r1,c4,c5]

    fin = reshape(fin,(size(L,1)*size(W,3)*size(R,1)))

    return fin
end

function MpoToMpsOneSiteKeff(L,R,M)

    M = reshape(M,(size(L,3),size(R,3)))

    @tensor fin[l1,r1] := L[l1,c1,c2]*M[c2,c3]*R[r1,c1,c3]

    fin = reshape(fin,(size(L,1)*size(R,1)))

    return fin
end


"""
The following three functions does the SVD(followed by truncation), QR, and LQ
decomposition of a matrix respectively.
"""
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

function QC(mat)

    F = qr(mat)
    Q = F.Q
    C = F.R

    return Q,C

end

function CQ(mat)

    F=lq(mat)
    C =  F.L
    Q = F.Q

    return C,Q
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


"""
lanczos algorithm for finding lowest eigenvalue (eigensolver) and matrix exponential (exponentialsolver)
the algorithm is taken from https://www.tensors.net/j-dmrg which provides a very compact way for Lanczos
algorithm
"""

"""
eigensolver: the lanczos algorithm is repeated twice to ensure accuracy
"""
function EigenLanczOneSite(psivec,L,W,R,krydim)

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(length(psivec),krydim+1);
    A = zeros(krydim,krydim);
    dval = 0;

    #the lanczos algorithm is repeated twice
    for k = 1:2

        psi[:,1] = psivec/max(norm(psivec),1e-16);

        for p = 2:krydim+1
            psi[:,p] = MpoToMpsOneSite(L,W,R,psi[:,p-1])

            #iterative building of the tridiagonal matrix
            for g = p-2:1:p-1
                if g >= 1
                    A[p-1,g] = dot(psi[:,p],psi[:,g]);
                    A[g,p-1] = conj(A[p-1,g]);
                end
            end

            #full reorthogonalization
            for g = 1:1:p-1
                psi[:,p] = psi[:,p] - dot(psi[:,g],psi[:,p])*psi[:,g];
                psi[:,p] = psi[:,p]/max(norm(psi[:,p]),1e-16);
            end

        end

        G = eigen(0.5*(A+A'));
        dval, xloc = findmin(G.values);
        psivec = psi[:,1:krydim]*G.vectors[:,xloc[1]];
    end

    psivec = psivec/norm(psivec);

    return psivec, dval
end

"""
exponentialsolver: the algorithm is taken from chapter 5 (titled Global Krtlov Method) of
an excellent review (Time-evolution methods for matrix-product states), arXiv:1901.
"""

function ExpLanczOneSite(psivec,L,W,R,krydim,dt)

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(ComplexF64, length(psivec),krydim+1)
    A = zeros(ComplexF64,krydim,krydim);
    vec = zeros(ComplexF64,length(psivec))

        nom = norm(psivec)

        psi[:,1] = psivec/max(norm(psivec),1e-16);

        for p = 2:krydim+1

            psi[:,p] = MpoToMpsOneSite(L,W,R,psi[:,p-1])

            for g = p-2:1:p-1
                if g >= 1
                    A[p-1,g] = dot(psi[:,p],psi[:,g]);
                    A[g,p-1] = conj(A[p-1,g]);
                end
            end

            for g = 1:1:p-1
                psi[:,p] = psi[:,p] - dot(psi[:,g],psi[:,p])*psi[:,g];
                psi[:,p] = psi[:,p]/max(norm(psi[:,p]),1e-16);
            end

        end

    c = exp(-im*dt*A)*I(length(A[:,1]))[:,1]

    for i in 1:length(c)
        vec += c[i]*psi[:,i]
    end

    return nom*vec
end

"""
this one is similar to the one above except it performs the exponential of
Keff matrix
"""
function  ExpLanczOneSiteKeff(psivec,L,R,krydim,dt)

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(ComplexF64, length(psivec),krydim+1);
    A = zeros(ComplexF64,krydim,krydim);
    vec = zeros(ComplexF64,length(psivec))

    nom = norm(psivec)

    psi[:,1] = psivec/max(norm(psivec),1e-16);

        for p = 2:krydim+1

            psi[:,p] = MpoToMpsOneSiteKeff(L,R,psi[:,p-1])

           for g = p-2:1:p-1

                if g >= 1
                    A[p-1,g] = dot(psi[:,p],psi[:,g]);
                    A[g,p-1] = conj(A[p-1,g]);
                end

            end

            for g = 1:1:p-1
                psi[:,p] = psi[:,p] - dot(psi[:,g],psi[:,p])*psi[:,g];
                psi[:,p] = psi[:,p]/max(norm(psi[:,p]),1e-16);
            end
        end

    c = exp(im*dt*A)*I(length(A[:,1]))[:,1]

    for i in 1:length(c)
        vec += c[i]*psi[:,i]
    end

    return nom*vec
end


"""
The following two functions is the right and left one-site DMRG sweeps using
the lanczos routine.
"""

function right_sweep_DMRG_one_site_Lancz!(L,R,Ham,M,A,B,N,krydim)

    @inbounds for i in 1:N

        shp_M = size(M)

        #reshape M into a vector
        psivec = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        """
        get the lowest eigenvalues and eigenvector of H_eff only this time we dont
        build H_eff explicitly. We input L,H,R,psivec, and krydim and get the lowest
        eigenvalue and eigenvector.
        """
        @time eig_vec, eig_val = EigenLanczOneSite(psivec,L[indx_L(i-1,N)],Ham[i],R[i+1],krydim)

        print(eig_val/N,"\n")

        #reshape eigen_vec as vector into a matric for SVD
        vec = reshape(eig_vec,(shp_M[1]*shp_M[2],shp_M[3]))

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

function left_sweep_DMRG_one_site_Lancz!(L,R,Ham,M,A,B,N,krydim)

    @inbounds for i in reverse(1:N)

        shp_M = size(M)

        #reshape M into a vector
        psivec = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        """
        get the lowest eigenvalues and eigenvector of H_eff only this time we dont
        build H_eff explicitly. We input
        """

        @time eig_vec, eig_val = EigenLanczOneSite(psivec,L[indx_L(i-1,N)],Ham[i],R[i+1],krydim)

        print(eig_val/N,"\n")

        #reshape back to matrix
        vec = reshape(eig_vec,(shp_M[1],shp_M[2]*shp_M[3]))

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


N = 100
chi = 20
d = 2
num_DMRG = 4

a = 2.0
h = 3.0
n = 14
krydim = 40

M_set = Array{Any,1}(undef,N)
M_set, B_set = initial_psi!(N,chi,d,M_set)

Ham = Array{Any,1}(undef,N)
Ham = Hamiltonian_LR_Ising(a,h,N,n,Ham)

A,B,L,R,M = Initialize(N,M_set,B_set,Ham)

@time for i in 1:num_DMRG
    A,B,L,R,M = right_sweep_DMRG_one_site_Lancz!(L,R,Ham,M,A,B,N,krydim)
    A,B,L,R,M = left_sweep_DMRG_one_site_Lancz!(L,R,Ham,M,A,B,N,krydim)
end


function right_sweep_TDVP_one_QR_Lancz(dt,L,R,Ham,M,A,B,N,krydim)

    @inbounds for i in 1:N

        shp_M = size(M)

        #reshape M into a vector
        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        #exponenriate Heff and apply on psivec
        M = ExpLanczOneSite(M,L[indx_L(i-1,N)],Ham[i],R[i+1],krydim,dt/2)

        #print(M,"\n")
        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        #SVD and truncation
        mat_tem = reshape(M,(shp_M[1]*shp_M[2],shp_M[3]))

        #QR decompose mat_tem
        QR_dec = qr(mat_tem)
        Q = Matrix(QR_dec.Q)

        #reshape Q into A(left normalized MPS)
        A[i] = reshape(Q,(shp_M[1],shp_M[2],:))

        #build L[i]
        L[indx_L(i,N)] = contract_left(L[indx_L(i-1,N)],Ham[i],A[i])

        if i != N

            C = QR_dec.R
            shp_C = size(C)

            #reshape C into a vector
            C = reshape(C,(shp_C[1]*shp_C[2]))

            #exponentiate C and apply on C
            C = ExpLanczOneSiteKeff(C,L[indx_L(i,N)],R[i+1],krydim,dt/2)

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

function left_sweep_TDVP_one_QR_Lancz(dt,L,R,Ham,M,A,B,N,krydim)

    @inbounds for i in reverse(1:N)

        shp_M = size(M)

        #reshape M into a vector
        M = reshape(M,(shp_M[1]*shp_M[2]*shp_M[3]))

        #exponenriate Heff and apply on psivec
        M = ExpLanczOneSite(M,L[indx_L(i-1,N)],Ham[i],R[i+1],krydim,dt/2)

        M = reshape(M,(shp_M[1],shp_M[2],shp_M[3]))

        #SVD and truncation
        mat_tem = reshape(M,(shp_M[1],shp_M[2]*shp_M[3]))

        #LQ decomposition of mat_tem
        LQ_dec = lq(mat_tem)
        Q = Matrix(LQ_dec.Q)

        #reshape V into B[i] (which is a right normalized MPS)
        B[i] = reshape(Q,(:,shp_M[2],shp_M[3]))

        #create R[i]
        R[i] = contract_right(R[i+1],Ham[i],B[i])

        if i != 1

            C = LQ_dec.L
            shp_C = size(C)

            #reshape C into a vector
            C = reshape(C,(shp_C[1]*shp_C[2]))

            #exponentiate C and apply on C
            C = ExpLanczOneSiteKeff(C,L[indx_L(i-1,N)],R[i],krydim,dt/2)

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
This functions initializes the system at the GS at a given parameter space using
DMRG, quenches the system suddenly to another point in parameter space, and
evolves in real time using one-site TDVP.

Here all the states are saved in one .jld files. out.jld has num_TDVP number of
objects for each time slices, each of these objects has N number of MPS that
give the state at the given time slice
"""
function TDVP_call(N,chi,d,param,n,dt,num_DMRG,num_TDVP,krydim_DMRG,krydim_TDVP)

    a,h,h1 = param

    M_set = Array{Any,1}(undef,N)
    M_set, B_set = initial_psi!(N,chi,d,M_set)

    Ham = Array{Any,1}(undef,N)
    Ham = Hamiltonian_LR_Ising(a,h,N,n,Ham)

    A,B,L,R,M = Initialize(N,M_set,B_set,Ham)

    for i in 1:num_DMRG
        A,B,L,R,M = right_sweep_DMRG_one_site_Lancz!(L,R,Ham,M,A,B,N,krydim_DMRG)
        A,B,L,R,M = left_sweep_DMRG_one_site_Lancz!(L,R,Ham,M,A,B,N,krydim_DMRG)
    end

    Ham1 = Array{Any,1}(undef,N)
    Ham1 = Hamiltonian_LR_Ising(a,h1,N,n,Ham1)

    out = Array{Any,1}(undef,num_TDVP)
    psi1 = Array{Any,1}(undef,N)

    for i in 1:num_TDVP
        @time A,B,L,R,M = right_sweep_TDVP_one_QR_Lancz(dt,L,R,Ham1,M,A,B,N,krydim_TDVP)
        @time A,B,L,R,M = left_sweep_TDVP_one_QR_Lancz(dt,L,R,Ham1,M,A,B,N,krydim_TDVP)

        psi1[1] = M

        for j in 2:N
            psi1[j] = B[j]
        end

        mag = mag_single_site(N,psi1)
        print(real(mag),"\n")
        out[i] = psi1
    end

    save(@sprintf("state_a=%1.2f_hi=%1.2f_hf=%1.2f.jld",a,h,h1), "data",out)
end

N = 100
chi = 100
d = 2
num_DMRG = 3
num_TDVP = 300
dt = 0.05
param = (6.0,0.0,3.0)
n = 14

"""
Exponentialsolver converges in less number of krylov iteration than eigensolver
so we define two different krylov dimensions
"""
krydim_DMRG = 30
krydim_TDVP = 14

@time TDVP_call(N,chi,d,param,n,dt,num_DMRG,num_TDVP)

"""
The part of the code below these are to check the algorithms above.
"""
function mag(N,param,mag_bag)

    res = load(@sprintf("state_a=%1.2f_hi=%1.2f_hf=%1.2f.jld",param[1],param[2],param[3]))["data"]

    for i in 1:num_TDVP

        psi = res[i]

        magnetization = mag_single_site(N,psi)
        mag_bag[i] = [i*dt,real(magnetization)]

        print(i,real(magnetization),"\n")
    end
    open(@sprintf("mag_a=%1.2f_hi=%1.2f_hf=%1.2f.txt",param[1],param[2],param[3]),"w") do io
        writedlm(io,mag_bag,',')
    end
end

N = 100
param = (6.0,0.0,3.0)
num_TDVP =300

mag_bag = Array{Any,1}(undef,num_TDVP)
mag(N,param,mag_bag)

x = readdlm(@sprintf("mag_a=%1.2f_hi=%1.2f_hf=%1.2f.txt",param[1],param[2],param[3]), ',')



x_list = Array{Any,1}(undef,num_TDVP)
y_list = Array{Any,1}(undef,num_TDVP)


for i in 1:num_TDVP
    x_list[i] = x[:,1][i]
    y_list[i] = x[:,2][i]
end

plot(x_list,y_list)
