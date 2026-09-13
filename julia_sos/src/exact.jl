"""Exact rational LDL-style PSD test, including singular pivots. No tolerance."""
function psd_exact(Q::AbstractMatrix)
    size(Q,1)==size(Q,2) || return false
    Q==transpose(Q) || return false
    M=rat.(Matrix(Q)); n=size(M,1)
    for k in 1:n
        d=M[k,k]
        d<0 && return false
        if iszero(d)
            any(!iszero(M[i,k]) for i in (k+1):n) && return false
        else
            for i in (k+1):n, j in (k+1):n
                M[i,j] -= M[i,k]*M[k,j]/d
            end
        end
    end
    return true
end
"""Round and shift a numerical Gram matrix; exact PSD is checked after each shift.
The introduced shift is NOT ignored: the verifier bounds its full polynomial error.
"""
function certify_gram(Q)
    R=rat.(Matrix(Q)); R=(R+transpose(R))/2
    shifts=vcat([rat(0)],[BigInt(1)//BigInt(10)^k for k in 12:-1:2])
    for d in shifts
        S=copy(R)
        for i in axes(S,1); S[i,i]+=d; end
        psd_exact(S) && return S,d
    end
    throw(ArgumentError("unable to establish exact PSD after bounded diagonal shifts"))
end
function nonnegative_bound(p,x,K)
    pp=p+0*x[1]
    radii=[max(abs(a),abs(b)) for (a,b) in K]
    return sum((abs(rat(MP.coefficient(t)))*prod(radii[i]^MP.degree(MP.monomial(t),x[i]) for i in eachindex(x)) for t in MP.terms(pp));init=rat(0))
end
function gramdata(Q,z,index,x)
    return Dict("Q"=>[string.(Q[i,:]) for i in axes(Q,1)],
                "basis"=>[polydata(qpoly(p,x),x) for p in z],"generator"=>index)
end
function readgram(record,x,n)
    rows=record["Q"]; k=length(rows)
    k>0 || throw(ArgumentError("empty Gram matrix"))
    all(length(row)==k for row in rows) || throw(DimensionMismatch("Gram matrix"))
    Q=[rat(String(rows[i][j])) for i in 1:k,j in 1:k]
    z=[frompolydata(p,x) for p in record["basis"]]
    length(z)==k || throw(DimensionMismatch("Gram basis"))
    idx=Int(record["generator"])
    0<=idx<=n || throw(ArgumentError("invalid generator"))
    return Q,z,idx
end
function bundle_base(P,family,B,parameter,separation)
    pm = family in (:forward_vbc,:backward_vbc) ?
        [string.(rat.(parameter[i,:])) for i in axes(parameter,1)] : string.(rat.(parameter))
    return Dict{String,Any}("schema"=>"dt-vbc-sos-exact-v1","problem"=>P.name,
        "family"=>String(family),"B"=>[polydata(qpoly(p,P.x),P.x) for p in B],
        "parameter"=>pm,"separation"=>string(rat(separation)),
        "source_commit"=>get(ENV,"GITHUB_SHA","local-unrecorded"),
        "propagation_domain"=>"full_X","anchor"=>1,"proofs"=>Any[])
end
"""Independent replay. Rebuild targets from fixed benchmark data and certificate
coefficients, not from exported target polynomials or a solver's claimed status.
Every Gram matrix, coefficient residual and bound is checked in Rational{BigInt}.
"""
function verify_bundle(bundle)
    String(bundle["schema"])=="dt-vbc-sos-exact-v1" || error("unknown schema")
    P=benchmark(String(bundle["problem"])); x=P.x
    String(bundle["propagation_domain"])=="full_X" || error("unsupported domain")
    Int(bundle["anchor"])==1 || error("unsupported anchor")
    family=Symbol(bundle["family"])
    B=[frompolydata(p,x) for p in bundle["B"]]; m=length(B)
    p0=bundle["parameter"]
    par=family in (:forward_vbc,:backward_vbc) ?
        [rat(String(p0[i][j])) for i in 1:m,j in 1:m] : [rat(String(v)) for v in p0]
    sep=rat(String(bundle["separation"])); sep>0 || error("strict separation required")
    obs=obligations(P,family,B,par;separation=sep)
    records=bundle["proofs"]
    length(records)==length(obs) || error("missing or extra obligations")
    checks=Any[]
    for (o,record) in zip(obs,records)
        String(record["name"])==o.name || error("obligation order/name mismatch")
        reserve=rat(String(record["reserve"])); reserve>=0 || error("negative reserve")
        gs=generators(x,o.box); representation=rat(0)*x[1]
        for block in record["grams"]
            Q,z,idx=readgram(block,x,length(gs))
            psd_exact(Q) || error("Gram matrix is not exactly PSD")
            s=sum(Q[i,j]*z[i]*z[j] for i in eachindex(z),j in eachindex(z))
            representation+=(idx==0 ? rat(1) : gs[idx])*s
        end
        error_poly=o.p-reserve-representation
        beta=nonnegative_bound(error_poly,x,o.box)
        lower=reserve-beta
        push!(checks,Dict("name"=>o.name,"verified"=>lower>=0,
            "absolute_error_bound"=>string(beta),"nonnegative_lower_bound"=>string(lower)))
    end
    poly_ok=all(c["verified"] for c in checks)
    domain=domain_audit(P)
    safe=poly_ok && domain["status"]=="EXACT_INVARIANT"
    return Dict("verified"=>safe,"polynomial_obligations_verified"=>poly_ok,
        "status"=>safe ? "EXACT_RATIONAL_VERIFIED" : (poly_ok ? "DOMAIN_NOT_ESTABLISHED" : "UNVERIFIED_CANDIDATE"),
        "domain"=>domain,"checks"=>checks)
end

"""Construct exact rational weighted-square SOS witnesses for the original S1.
These are analytical witnesses, NOT newly optimized margins or solver results.
"""
function exact_s1()
    P=benchmark("S1");x,y=P.x; X=P.X
    a0,d0=rat(303//625),rat(327//625)
    t,u,b,c,r,alpha,delta=rat.([3//25,2//25,1//10,2//25,3//50,41//50,1//50])
    q=rat(82//125); V=x^2+y^2; g=V-rat(1//10)
    # Terms (weight, square-root polynomial, domain-generator index).
    Hterms=[(b*a0,x-y,0),(t*a0,x^2,0),(t*b,x*y,0),
        (c*d0,x+y,0),(r*c,x,0),(r*d0,y,0),(u*c,x*y,0),
        (u*d0,y^2,0),(alpha*t,x^2,0),(alpha*delta,y,0),(alpha*u,y^2,0),
        (b*t,x-y,1),(t*t,x^2,1),(c*u,x+y,2),(u*r,y,2),(u*u,y^2,2)]
    gs=generators(P.x,X)
    H=sum(w*p^2*(idx==0 ? 1 : gs[idx]) for (w,p,idx) in Hterms)
    @assert iszero(H-(q*V-sum(fi^2 for fi in P.f)))
    @assert all(w>=0 for (w,p,idx) in Hterms)
    configs=[(:forward_vbc,fill(g,3),path_matrix(fill(5//4,3);reciprocal=true),rat(1),rat(4//5)-q,rat(1//50)),
        (:backward_vbc,fill(-g,3),path_matrix(fill(6//5,3)),rat(6//5),1-rat(6//5)*q,rat(1//50)),
        (:forward_ibc,fill(g,3),rat.(fill(6//5,3)),rat(6//5),1-rat(6//5)*q,rat(1//50)),
        (:backward_ibc,fill(-g,3),rat.(fill(5//4,3)),rat(5//4),1-rat(5//4)*q,rat(1//40))]
    bundles=Any[]; sep=rat(1//100)
    for (family,B,parameter,hscale,vscale,constant) in configs
        bundle=bundle_base(P,family,B,parameter,sep)
        for o in obligations(P,family,B,parameter;separation=sep)
            if iszero(o.p-(-g-sep))
                terms=[(rat(1//50)-sep,1+0*x,0),(rat(1),1+0*x,1),(rat(1),1+0*x,2)]
            elseif iszero(o.p-(g-sep))
                lo,hi=rat(19//20),rat(5//4)
                terms=[(2*lo^2-rat(1//10)-sep,1+0*x,0),
                    ((hi+lo)/(hi-lo),x-lo,0),((hi+lo)/(hi-lo),y-lo,0),
                    (2*lo/(hi-lo),1+0*x,1),(2*lo/(hi-lo),1+0*x,2)]
            else
                terms=vcat([(hscale*w,p,idx) for (w,p,idx) in Hterms],
                           [(vscale,x,0),(vscale,y,0),(constant,1+0*x,0)])
            end
            @assert all(w>=0 for (w,p,idx) in terms)
            localgs=generators(P.x,o.box)
            rep=sum(w*p^2*(idx==0 ? 1 : localgs[idx]) for (w,p,idx) in terms)
            @assert iszero(o.p-rep)
            grams=[gramdata(reshape([w],1,1),[p],idx,P.x) for (w,p,idx) in terms]
            push!(bundle["proofs"],Dict("name"=>o.name,"reserve"=>"0//1","grams"=>grams))
        end
        bundle["origin"]="analytical_exact_witness_not_solver_output"
        result=verify_bundle(bundle); @assert result["verified"]
        bundle["verification"]=result
        push!(bundles,bundle)
    end
    return bundles
end
"""Transport an IBC proof to the sign-reversed VBC, with no new SDP solve.
Backward IBC propagation identities and their SOS witnesses are divided by lambda.
Separation proofs are reordered by obligation name. No equal-margin claim is
made after unrelated positive component scalings or different normalizations.
"""
function transfer_bundle(bundle)
    P=benchmark(String(bundle["problem"]));x=P.x
    family=Symbol(bundle["family"])
    family in (:forward_ibc,:backward_ibc) || throw(ArgumentError("source must be IBC"))
    b=[frompolydata(p,x) for p in bundle["B"]]
    lambda=[rat(String(v)) for v in bundle["parameter"]]
    target=family==:backward_ibc ? :forward_vbc : :backward_vbc
    A=path_matrix(lambda;reciprocal=family==:backward_ibc)
    sep=rat(String(bundle["separation"]))
    out=bundle_base(P,target,-b,A,sep)
    byname=Dict(String(r["name"])=>r for r in bundle["proofs"])
    for o in obligations(P,target,-b,A;separation=sep)
        source=byname[o.name]
        factor=startswith(o.name,"propagation_") && family==:backward_ibc ?
               inv(lambda[parse(Int,split(o.name,"_")[end])]) : rat(1)
        grams=Any[]
        for block in source["grams"]
            Q,z,idx=readgram(block,x,length(P.X))
            push!(grams,gramdata(factor*Q,z,idx,x))
        end
        push!(out["proofs"],Dict("name"=>o.name,"reserve"=>string(factor*rat(String(source["reserve"]))),"grams"=>grams))
    end
    out["origin"]="transported_IBC_proof_without_resynthesis"
    out["verification"]=verify_bundle(out)
    return out
end
