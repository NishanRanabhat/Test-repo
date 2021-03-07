using LinearAlgebra
using BenchmarkTools
using TensorOperations

"""
This part consists of the code for the Lanczos algorithm for finding the lowest eigenvalue and
eigenvector and for finding the exponential of a matrix applied to a vector. Both procedures implies
same algorithm with some minor changes. This particular implication of the Lanczos algorithm is taken
from a very resourseful website https://www.tensors.net/j-dmrg.
"""

"""
Lanczos algorithm in general takes a matrix(Hamiltonian) and a vector(initial state)
as an input and builds the Krylov space by successively applying the the matrix to
the vector and orthogonalizing so that the algorithm itself doesn't need the matrix
and vector seperately but the result of application of the matrix onto the vector. In
addition to this we are working on the tensor network formalism which gives us a more
efficient way to build this matrix vector multiplication. So computing H_eff*M = M' as
(L*W*R)*M = M' gives us a considerable gain in efficiency.
"""

"""
This function applies one site effective hamiltonian H_eff1 (in the form (L*W*R) ) onto the state
"""
function MpoToMpsOneSite(L,W,R,M)

    M = reshape(M,(size(L,3),size(W,4),size(R,3)))

    @tensor fin[l1,w3,r1] := L[l1,c1,c2]*M[c2,c3,c5]*W[c1,c4,w3,c3]*R[r1,c4,c5]

    fin = reshape(fin,(size(L,1)*size(W,3)*size(R,1)))

    return fin
end

"""
This function applies one site effective hamiltonian K_eff1 (in the form (L*R) ) onto the state
"""
function MpoToMpsOneSiteKeff(L,R,M)

    M = reshape(M,(size(L,3),size(R,3)))

    @tensor fin[l1,r1] := L[l1,c1,c2]*M[c2,c3]*R[r1,c1,c3]

    fin = reshape(fin,(size(L,1)*size(R,1)))

    return fin
end

"""
This function applies two site effective hamiltonian H_eff2 (in the form (L*W_i*W_i+1*R) ) onto the state
"""
function MpoToMpsTwoSite(L,W1,W2,R,M)

    M = reshape(M,(size(L,3),size(W1,4),size(W2,4),size(R,3)))

    @tensor fin[l1,w13,w23,r1] := L[l1,c1,c2]*M[c2,c3,c5,c7]*W1[c1,c4,w13,c3]*W2[c4,c6,w23,c5]*R[r1,c6,c7]

    fin = reshape(fin,(size(L,1)*size(W1,3)*size(W2,3)*size(R,1)))

    return fin
end


"""
Eigensolver for one site DMRG: the lanczos algorithm is repeated twice to ensure accuracy
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
Eigensolver for one site hamiltonian without tensor network formalism
"""
function EigenLanczOneSite_Ham(psivec,H,krydim)

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
            psi[:,p] = H*psi[:,p-1] #direct application of H onto ps

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
Exponentialsolver: the algorithm is taken from chapter 5 (titled Global Krtlov Method) of
an excellent review (Time-evolution methods for matrix-product states), arXiv:1901.
For the forward evolution of one-site TDVP
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

    c = exp(-im*dt*A)*I(length(A[:,1]))[:,1] #minus sign ensures forward evolution

    for i in 1:length(c)
        vec += c[i]*psi[:,i]
    end

    return nom*vec
end

"""
Same as above without Tensor network formalism
"""
function ExpLanczOneSite_Ham(psivec,H,krydim,dt)

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(ComplexF64, length(psivec),krydim+1)
    A = zeros(ComplexF64,krydim,krydim);
    vec = zeros(ComplexF64,length(psivec))

        nom = norm(psivec)

        psi[:,1] = psivec/max(norm(psivec),1e-16);

        for p = 2:krydim+1
            psi[:,p] = H*psi[:,p-1] #direct application of H onto psi

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
This one is similar to the one above except it performs the exponential of
Keff matrix onto the C vector. It is used for backward evolution of the C vector
in one-site TDVP.
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

    c = exp(im*dt*A)*I(length(A[:,1]))[:,1] #plus sign ensures backward evolution

    for i in 1:length(c)
        vec += c[i]*psi[:,i]
    end

    return nom*vec
end

"""
Same thing as above without tensor network formalism
"""
function  ExpLanczOneSiteKeff_Ham(psivec,H,krydim,dt)

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(ComplexF64, length(psivec),krydim+1);
    A = zeros(ComplexF64,krydim,krydim);
    vec = zeros(ComplexF64,length(psivec))

    nom = norm(psivec)

    psi[:,1] = psivec/max(norm(psivec),1e-16);

        for p = 2:krydim+1
            psi[:,p] = H*psi[:,p-1]

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
Eigensolver for two site DMRG
"""
function EigenLanczTwoSite(psivec,L,W1,W2,R,krydim)

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
            psi[:,p] = MpoToMpsTwoSite(L,W1,W2,R,psi[:,p-1])

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
Exponentialsolver for the forward evolution in two-site TDVP
"""
function ExpLanczTwoSite(psivec,L,W1,W2,R,krydim,dt)

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(ComplexF64, length(psivec),krydim+1);
    A = zeros(ComplexF64,krydim,krydim);
    vec = zeros(ComplexF64,length(psivec))

        nom = norm(psivec)

        psi[:,1] = psivec/max(norm(psivec),1e-16);

        for p = 2:krydim+1
            psi[:,p] = MpoToMpsTwoSite(L,W1,W2,R,psi[:,p-1])

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

    c = exp(-im*dt*A)*I(length(A[:,1]))[:,1] #minus sign ensures forward evolution

    for i in 1:length(c)
        vec += c[i]*psi[:,i]
    end

    return nom*vec
end

"""
Exponentialsolver for the backward evolution for two-site TDVP
"""
function ExpLanczOneSite_Back(psivec,L,W,R,krydim,dt)

    if norm(psivec) == 0
        psivec = rand(length(psivec),1);
    end

    psi = zeros(ComplexF64, length(psivec),krydim+1);
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

    c = exp(im*dt*A)*I(length(A[:,1]))[:,1] #plus sign ensures backward evolution

    for i in 1:length(c)
        vec += c[i]*psi[:,i]
    end

    return nom*vec
end
