import numpy as np
from numpy import linalg as LA
import scipy.linalg as la
from scipy.sparse.linalg import eigsh, eigs, expm_multiply
from scipy.linalg import expm
import time
import sys
import multiprocessing as mp

def initial_psi(N,chi,d):

    M_set = [0 for x in range(N)]
    M_set[0] = np.random.rand(1,d,chi)
    M_set[N-1] = np.random.rand(chi,d,1)

    for i in range(1,N-1):
        M_set[i] = np.random.rand(chi,d,chi)

    B = M_set.copy()

    return M_set,B


def compact_svd(mat):
    
    U,S,V = LA.svd(mat)
    
    if mat.shape[0] > mat.shape[1]:
        U = U[:,:mat.shape[1]]

    elif mat.shape[0] < mat.shape[1]:
        V = V[:mat.shape[0],:]

    return U,S,V


def right_normalize(B,N):

    for i in range(N-1,0,-1):
    
        U,S,V = compact_svd(np.reshape(B[i],(B[i].shape[0],B[i].shape[1]*B[i].shape[2])))
        B[i] = np.reshape(V,(-1,B[i].shape[1],B[i].shape[2]))
        B[i-1] = np.tensordot(B[i-1],np.dot(U,np.diag(S)),axes = [2,0])/LA.norm(S) 

    U,S,V = compact_svd(np.reshape(B[0],(B[0].shape[0],B[0].shape[1]*B[0].shape[2])))
    B[0] = np.reshape(V,(-1,B[0].shape[1],B[0].shape[2]))

    return B


def check_right_normalization(B,N):
    for i in range(N):
        summa = np.add(np.dot(B[i][:,0,:],B[i][:,0,:].T), np.dot(B[i][:,1,:],B[i][:,1,:].T))
        print("Test right unitarity: %s" % np.allclose(summa,np.eye(B[i].shape[0])))


def power_law_to_exp(a,N,n):
#create the F list

    F = []

    for k in range(1,N+1):
        f = 1/k**(a)
        F.append(f)

    F = np.array(F)

#make M matrix out of this list F

    M = np.zeros((N-n+1,n))

    for i in range(N-n+1):
        for j in range(n):

            M[i,j] = F[i+j]

# M goes through QR decomposition first and the eigenvals of Q1_inv.Q gives exponents

    Q,R = la.qr(M)

    Q1 = Q[0:N-n,0:n]
    Q1_inv = np.linalg.pinv(Q1)
    Q2 = Q[1:N-n+1,0:n]

    V = np.dot(Q1_inv,Q2)

    
    lambda1 = np.real(la.eig(V)[0])

# lets fit the least square fit to find the weights

#create the lambda matrix 

    lam_mat = np.zeros((N,n))

    for k in range(N):
        for i in range(len(lambda1)):

            lam_mat[k,i] = (lambda1[i])**(k+1)

#the weight is extrated by solving (lam_mat) x = F

    x = np.linalg.lstsq(lam_mat,F,rcond=None)[0]

    lambda1 = np.array(lambda1)
    x = np.array(x)

    return x,lambda1


def Hamiltonian_LR_Ising(a,N,n,h):

    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0,-1]])
    sI = np.array([[1, 0], [0, 1]])
    
    Kac = 0
    for i in range(1,N+1):
        Kac += (N-i)/(i**a)
    
    Kac = Kac/(N-1)
    
    x, lambda1 = power_law_to_exp(a,N,n)

    #building the local bulk MPO
    H = np.zeros([n+2,n+2,2,2])

    H[0,0,:,:] = sI; H[n+1,n+1,:,:] = sI; H[n+1,0,:,:] = -h*sZ

    for i in range(1,n+1):
        H[i,0,:,:] = (x[i-1]/Kac)*sX
        H[i,i,:,:] = lambda1[i-1]*sI

    for j in range(1,n+1):
        H[n+1,j,:,:] = -lambda1[j-1]*sX

    #building the boundary MPOs
    HL = np.zeros((1,n+2,2,2))
    HL[0,:,:,:] = H[n+1,:,:,:]
    HR = np.zeros((n+2,1,2,2))
    HR[:,0,:,:] = H[:,0,:,:]
    
    #put the hamiltonian in a list so that it can be iteteratively recuperated

    Ham = [0 for x in range(N)]

    Ham[0] = HL
    Ham[N-1] = HR
    for i in range(1,N-1):
        Ham[i] = H

    return Ham


def contract_right(R,W,B):
    
    R1 = np.tensordot(B.conj(),R,axes = [2,0])
    R1 = np.tensordot(R1,W,axes = [[1,2],[2,1]])
    R1 = np.tensordot(R1,B,axes = [[1,3],[2,1]])
    
    return R1

def contract_left(L,W,A):
    
    L1 = np.tensordot(A.conj(),L,axes = [0,0])
    L1 = np.tensordot(L1,W,axes = [[0,2],[2,0]])
    L1 = np.tensordot(L1,A,axes = [[1,3],[0,1]])
    return L1

def contract_left_noop(L,A):
    
    L1 = np.tensordot(A.conj(),L,axes = [0,0])
    L1 = np.tensordot(L1,A,axes = [[0,2],[1,0]])
    return L1 

def contract_right_noop(R,B):
    
    R1 = np.tensordot(B.conj(),R,axes = [2,0])
    R1 = np.tensordot(R1,B,axes = [[1,2],[1,2]])
    return R1 

def magnetization_val(L,M,W,R):
    
    mag = np.tensordot(M.conj(),W,axes = [1,0])
    mag = np.tensordot(mag,L,axes = [0,0])
    mag = np.tensordot(mag,M,axes = [[2,1],[0,1]])
    mag = np.tensordot(mag,R,axes = [[0,1],[0,1]])
    
    return mag

def contract_left_nonmpo(L,W,A):
    
    L1 = np.tensordot(A.conj(),L,axes = [0,0])
    L1 = np.tensordot(L1,W,axes = [0,0])
    L1 = np.tensordot(L1,A,axes = [[1,2],[0,1]])
    
    return L1

def MpoToMpsOneSite(L,W,R,M):
    
    M = np.reshape(M,(L.shape[2],W.shape[3],R.shape[2]))
    
    fin = np.tensordot(W,R,axes = [1,1])
    fin = np.tensordot(M,fin,axes = [[1,2],[1,4]])
    fin = np.tensordot(L,fin, axes = [[1,2],[1,0]])
    
    fin = np.reshape(fin,(L.shape[0]*W.shape[2]*R.shape[0]))
    
    return fin

def MpoToMpsOneSiteKeff(L,R,M):

    M = np.reshape(M,(L.shape[2],R.shape[2]))
    fin = np.tensordot(M,R, axes = [1,2])
    fin = np.tensordot(L,fin, axes = [[1,2],[2,0]])

    fin = np.reshape(fin,(L.shape[0]*R.shape[0]))
    
    return fin


def svd_truncate(T,chi):

    U,S,V = compact_svd(T)

    if len(S) > chi:
        
        S = S[0:chi]
        U = U[:,0:chi]
        V = V[0:chi,:]

    S = S/LA.norm(S)

    return U,S,V


def Initialize(M_set,B,Ham,N):

    #get the list for R and boundary R
    R = [0 for x in range(N+1)] 
    R[N] = np.ones((1,1,1))

    #putting non-normalized MPS to right canonical form
    B = right_normalize(B,N)

    #generating R tensors
    for j in range(N-1,0,-1):
        R[j] = contract_right(R[j+1],Ham[j],B[j])

    # get the list for L and initialize boundary L
    L = [0 for x in range(N+1)]
    L[-1] = np.ones((1,1,1))

    #get the list for A
    A = [0 for x in range(N)]

    #get initial M 
    M = M_set[0]
    
    return A,B,L,R,M


def Initialize_magnetization(psi,N,l):
    
    R = [0 for x in range(N+1)] 
    R[N] = np.ones((1,1))
    
    L = [0 for x in range(N+1)]
    L[-1] = np.ones((1,1))
    
    a = int(N/2)-int(l/2)
    b = int(N/2)+int(l/2)
    
    for j in range(N-1,a,-1):
        R[j] = contract_right_noop(R[j+1],psi[j])
        
    for i in range(b-1):
        L[i] = contract_left_noop(L[i-1],psi[i])
        
    return L,R    


def EigenLanczOneSite(psivec,L,W,R,krydim,maxit):
    
    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec))
    
    psi = np.zeros([len(psivec),krydim+1],dtype=complex)
    A = np.zeros([krydim,krydim],dtype=complex)
    
    
    for k in range(maxit):
        
        psi[:,0] = psivec/max(LA.norm(psivec),1e-16)
        
        for p in range(1,krydim+1):
                
            psi[:,p] = MpoToMpsOneSite(L,W,R,psi[:,p-1])
            
            for g in range(p-2,p):
                if g >= 0:
                    A[p-1,g] = np.dot(psi[:,p].conj(),psi[:,g])
                    A[g,p-1] = np.conj(A[p-1,g])
            
            for g in range(p):
                psi[:,p] = psi[:,p] - np.dot(psi[:,g].conj(),psi[:,p])*psi[:,g]
                psi[:,p] = psi[:,p] / max(LA.norm(psi[:,p]),1e-16)
                    
        [dtemp,utemp] = LA.eigh(A)
        psivec = psi[:,range(0,krydim)] @ utemp[:,0]
        
    psivec = psivec/LA.norm(psivec)
    dval = dtemp[0]
    
    return psivec, dval


def ExpLanczOneSite(psivec,L,W,R,krydim,dt):
    
    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec))
        
    psi = np.zeros([len(psivec),krydim+1],dtype=complex)
    A = np.zeros([krydim,krydim],dtype=complex)
    vec = np.zeros(len(psivec),dtype=complex)
    
    nom = LA.norm(psivec)
        
    psi[:,0] = psivec/max(LA.norm(psivec),1e-16)
        
    for p in range(1,krydim+1):
                
        psi[:,p] = MpoToMpsOneSite(L,W,R,psi[:,p-1])
            
        for g in range(p-2,p):
            if g >= 0:
                A[p-1,g] = np.dot(psi[:,p].conj(),psi[:,g])
                A[g,p-1] = A[p-1,g].conj()

        for g in range(p):
            psi[:,p] = psi[:,p] - np.dot(psi[:,g].conj(),psi[:,p])*psi[:,g]
            psi[:,p] = psi[:,p] / max(LA.norm(psi[:,p]),1e-16)
                    
    c = expm(-1j*dt*A)[:,0]
    
    for i in range(len(c)):
        vec += c[i]*psi[:,i]
        
    return nom*vec


def ExpLanczOneSiteKeff(psivec,L,R,krydim,dt):
    
    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec))
        
    psi = np.zeros([len(psivec),krydim+1],dtype=complex)
    A = np.zeros([krydim,krydim],dtype=complex)
    vec = np.zeros(len(psivec),dtype=complex)
    
    nom = LA.norm(psivec)
        
    psi[:,0] = psivec/max(LA.norm(psivec),1e-16)
        
    for p in range(1,krydim+1):
                
        psi[:,p] = MpoToMpsOneSiteKeff(L,R,psi[:,p-1])
            
        for g in range(p-2,p):
            if g >= 0:
                A[p-1,g] = np.dot(psi[:,p].conj(),psi[:,g])
                A[g,p-1] = A[p-1,g].conj()

        for g in range(p):
            psi[:,p] = psi[:,p] - np.dot(psi[:,g].conj(),psi[:,p])*psi[:,g]
            psi[:,p] = psi[:,p] / max(LA.norm(psi[:,p]),1e-16)
                    
    c = expm(1j*dt*A)[:,0]
    
    for i in range(len(c)):
        vec += c[i]*psi[:,i]
        
    return nom*vec


def right_DMRG_sweep(L,R,Ham,M,A,B,chi,N,krydim_DMRG,maxit):
    
    for i in range(N):
        
        shp_M = M.shape
        
        #reshape M into a vector
        psivec = np.reshape(M,(shp_M[0]*shp_M[1]*shp_M[2]))
        
        #local minimization at site i with Lanczos algorithm
        eig_vec, eig_val = EigenLanczOneSite(psivec,L[i-1],Ham[i],R[i+1],krydim_DMRG,maxit)
        
        #reshape eig_vec into matrix for SVD
        vec = np.reshape(eig_vec,(shp_M[0]*shp_M[1],shp_M[2]))
        
        #SVD and truncation
        U,S,V = svd_truncate(vec,chi)
        
        #reshape U into A
        A[i] = np.reshape(U,(shp_M[0],shp_M[1],-1))
        
        #create L[i]
        L[i] = contract_left(L[i-1],Ham[i],A[i])
        
        if i != N-1:
            
            #create M tensor SV and B[i+1] 
            SV = np.dot(np.diag(S),V)
            M = np.tensordot(SV,B[i+1],axes = [1,0])
            
            #delete R[i+1]
            R[i+1] = 0.0
            
    return A,B,L,R,M           


def left_DMRG_sweep(L,R,Ham,M,A,B,chi,N,krydim_DMRG,maxit):
    
    for i in range(N-1,-1,-1):
        
        shp_M = M.shape
        
        #reshape M into a vector
        psivec = np.reshape(M,(shp_M[0]*shp_M[1]*shp_M[2]))
        
        #local minimization at site i with Lanczos algorithm
        eig_vec, eig_val = EigenLanczOneSite(psivec,L[i-1],Ham[i],R[i+1],krydim_DMRG,maxit)
        
        #reshape eig_vec into matrix for SVD
        vec = np.reshape(eig_vec,(shp_M[0],shp_M[1]*shp_M[2]))

        #SVD and truncation
        U,S,V = svd_truncate(vec,chi)
        
        #reshape U into A
        B[i] = np.reshape(V,(-1,shp_M[1],shp_M[2]))
        #print(B[i].shape)
        
        #create L[i]
        R[i] = contract_right(R[i+1],Ham[i],B[i])
        
        if i != 0:
            
            #create M tensor SV and B[i+1] 
            US = np.dot(U,np.diag(S))
            M = np.tensordot(A[i-1],US,axes = [2,0])
            
            #delete R[i+1]
            L[i-1] = 0.0
            
    return A,B,L,R,M  


def right_TDVP_sweep(dt,L,R,Ham,M,A,B,chi,N,krydim_TDVP):
    
    for i in range(N):
        shp_M = M.shape
              
        #reshape M into a vector
        M = np.reshape(M,(shp_M[0]*shp_M[1]*shp_M[2]))
        
        #exponentiate Heff at site i with Lanczos algorithm
        M = ExpLanczOneSite(M,L[i-1],Ham[i],R[i+1],krydim_TDVP,dt/2)      
        
        M = np.reshape(M,(shp_M[0],shp_M[1],shp_M[2]))

        #reshape M into matrix mat_tem
        mat_tem = np.reshape(M,(shp_M[0]*shp_M[1],shp_M[2]))
        
        #QR decompose mat_tem
        q,r = np.linalg.qr(mat_tem, mode='reduced')
        
        #reshape U into A
        A[i] = np.reshape(q,(shp_M[0],shp_M[1],-1))
        
        #create L[i]
        L[i] = contract_left(L[i-1],Ham[i],A[i])
        
        if i != N-1:
            
            shp_r = r.shape
            
            C = np.reshape(r,(shp_r[0]*shp_r[1]))
            
            C =  ExpLanczOneSiteKeff(C,L[i],R[i+1],krydim_TDVP,dt/2)
            
            C = np.reshape(C,(shp_r[0],shp_r[1]))
            
            #create M tensor from C and B[i+1] 
            M = np.tensordot(C,B[i+1],axes = [1,0])
            
            #delete R[i+1]
            R[i+1] = 0.0
                        
    return A,B,L,R,M 


def left_TDVP_sweep(dt,L,R,Ham,M,A,B,chi,N,krydim_TDVP):
    
    for i in range(N-1,-1,-1):
        
        shp_M = M.shape
            
        #reshape M into a vector
        M = np.reshape(M,(shp_M[0]*shp_M[1]*shp_M[2]))
            
        #lexponentiate Heff at site i with Lanczos algorithm
        M = ExpLanczOneSite(M,L[i-1],Ham[i],R[i+1],krydim_TDVP,dt/2)
       
        M = np.reshape(M,(shp_M[0],shp_M[1],shp_M[2]))
        
        #reshape M into matrix mat_tem
        mat_tem = np.reshape(M,(shp_M[0],shp_M[1]*shp_M[2]))
        
        #QR decompose mat_tem
        r,q =la.rq(mat_tem, mode='economic')
        
        #reshape U into A
        B[i] = np.reshape(q,(-1,shp_M[1],shp_M[2]))
        
        #create R[i]
        R[i] = contract_right(R[i+1],Ham[i],B[i])
        
        if i != 0:
            
            shp_r = r.shape
            
            C = np.reshape(r,(shp_r[0]*shp_r[1]))
            
            C =  ExpLanczOneSiteKeff(C,L[i-1],R[i],krydim_TDVP,dt/2)
            
            C = np.reshape(C,(shp_r[0],shp_r[1]))
            
            #create M tensor from C and B[i+1] 
            M = np.tensordot(A[i-1],C,axes = [2,0])
            
            #delete R[i+1]
            L[i-1] = 0.0
            
    return A,B,L,R,M 


def magnetization_single_site(psi):
    
    sX = 0.5*np.array([[0, 1], [1, 0]])
    sY = 0.5*np.array([[0, -1j], [1j, 0]])
    sZ = 0.5*np.array([[1, 0], [0,-1]])
    sI = np.array([[1, 0], [0, 1]])    

    X = np.ones((1,1))
    Y = X.copy()

    for i in range(N):
        
        if i == 50:
            
            mag = contract_left_nonmpo(X,sX,psi[i])
            
        else:
            mag = contract_left_noop(X,psi[i])
            
        X = mag

    mag = np.tensordot(mag,Y,axes = [[0,1],[0,1]])

    return mag


def magnetization_subsystem(psi,N,l):
    
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0,-1]])
    sI = np.array([[1, 0], [0, 1]]) 
    
    a = int(N/2)-int(l/2)
    b = int(N/2)+int(l/2)
    
    L,R =Initialize_magnetization(psi,N,l)
    
    mag = 0.0
    
    for i in range(a,b,1):
        
        mag += magnetization_val(L[i-1],psi[i],sX,R[i+1])
        
    mag = mag/l
    
    return mag


def product_opt(N,l,psi,theta):
    
    sX = 0.5*np.array([[0, 1], [1, 0]])
    sY = 0.5*np.array([[0, -1j], [1j, 0]])
    sZ = 0.5*np.array([[1, 0], [0,-1]])
    sI = np.array([[1, 0], [0, 1]])
    
    X = np.ones((1,1))
    Y = np.ones((1,1))
    
    k = int(N/2)-int(l/2)
    
    oprt = expm(1j*theta*sX)
    
    for i in range(k):
        X = contract_left_noop(X,psi[i])
        
    for i in range(k,k+l):
        X = contract_left_nonmpo(X,oprt,psi[i])
        
    for i in range(k+l,N):
        X = contract_left_noop(X,psi[i])
        
    X = np.tensordot(X,Y,axes=[[0,1],[0,1]])
    
    return X    


def eq_fcs(N,l,state_list,num_TDVP):
    
    out = [0 for x in range(num_TDVP)]
    
    for i in range(num_TDVP):
        
        psi = state_list[i]
        
        list_theta = np.arange(-np.pi,np.pi+0.05,0.05)
        
        overlap_list = [0 for x in range(len(list_theta))]
        
        print(i)
        for j in range(len(list_theta)):
            
            overlap = product_opt(N,l,psi,list_theta[j])
            
            overlap_list[j] = [list_theta[j],np.real(overlap),np.imag(overlap)]
            
        overlap_list = np.asarray(overlap_list)
        
        out[i] = overlap_list
        
    out = np.asarray(out)
    
    return out       
