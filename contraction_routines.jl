
using LinearAlgebra
using TensorOperations

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
