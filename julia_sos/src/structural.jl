"""Exact box SOS for affine p. Use t=(t^2+(x-l)(u-x))/(u-l),
where t=x-l or u-x. Independent replay checks the resulting identity.
"""
function affine_box_record(o,x)
    p=qpoly(o.p,x)
    MP.maxdegree(p)<=1 || throw(ArgumentError("affine target required"))
    coeff=[rat(MP.coefficient(p,xi^1)) for xi in x]
    c=rat(MP.coefficient(p,prod(xi^0 for xi in x)))
    grams=Any[]
    for i in eachindex(x)
        l,u=o.box[i]; a=coeff[i]
        if a>=0
            c+=a*l; t=x[i]-l
        else
            c+=a*u; t=u-x[i]
        end
        w=abs(a)/(u-l)
        if !iszero(w)
            push!(grams,gramdata(reshape([w],1,1),[t],0,x))
            push!(grams,gramdata(reshape([w],1,1),[1+0*x[1]],i,x))
        end
    end
    c>=0 || throw(ArgumentError("negative affine box minimum"))
    push!(grams,gramdata(reshape([c],1,1),[1+0*x[1]],0,x))
    return Dict("name"=>o.name,"reserve"=>"0//1","grams"=>grams)
end
"""Exact finite-order witness, not free synthesis. Degree one and cyclic
coupling. Periodic propagation has zero residual and zero numerical reserve.
"""
function exact_rotation(name="Rotation2")
    P=benchmark(name); x=P.x; n=length(x); c=rat(6//5); sep=rat(1//100)
    B=vcat([xi-c for xi in x],[-xi-c for xi in x]); m=2*n
    bf=[compose(p,x,P.f) for p in B]
    A=zeros(QQ,m,m)
    for i in 1:m
        j=findfirst(p->iszero(p-bf[i]),B)
        j===nothing && error("rotation does not permute the facets")
        A[i,j]=1
    end
    bundle=bundle_base(P,:forward_vbc,B,A,sep)
    for o in obligations(P,:forward_vbc,B,A;separation=sep)
        push!(bundle["proofs"],affine_box_record(o,x))
    end
    bundle["origin"]="analytical_cyclic_affine_certificate"
    bundle["verification"]=verify_bundle(bundle)
    @assert bundle["verification"]["verified"]
    return bundle
end
"""Normalize path edges and preserve the loop. Feasibility is invariant before
fixed margins/norm bounds; arbitrary fixed normalizations are NOT invariant.
"""
function canonical_path(A)
    n=size(A,1); checked_matrix(A,n)
    for i in 1:n,j in 1:n
        j==min(i+1,n) || iszero(A[i,j]) || throw(ArgumentError("not a path matrix"))
    end
    all(A[i,min(i+1,n)]>0 for i in 1:n) || throw(ArgumentError("positive edges required"))
    d=fill(rat(1),n)
    for i in 1:n-1; d[i+1]=d[i]*rat(A[i,i+1]); end
    D=Diagonal(d)
    return D*rat.(A)*inv(D),D
end
