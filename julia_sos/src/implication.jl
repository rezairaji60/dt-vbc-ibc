"""Exact replay for implication-style forward IBC witnesses.

This verifier is deliberately separate from the globally scaled SOS synthesis
path. It checks a fixed certificate and its SOS implication proof in exact
rational arithmetic; it does not claim convex free synthesis of an unknown
frame and unknown state-dependent multiplier.
"""
function _verify_putinar_record(p,K,record,x;extra_generators=Any[])
    reserve=rat(String(record["reserve"])); reserve>=0 || error("negative reserve")
    gs=vcat(generators(x,K),extra_generators)
    representation=rat(0)*x[1]
    for block in record["grams"]
        Q,z,idx=readgram(block,x,length(gs))
        psd_exact(Q) || error("Gram matrix is not exactly PSD")
        s=sum(Q[i,j]*z[i]*z[j] for i in eachindex(z),j in eachindex(z))
        representation+=(idx==0 ? rat(1) : gs[idx])*s
    end
    error_poly=qpoly(p,x)-reserve-representation
    beta=nonnegative_bound(error_poly,x,K)
    lower=reserve-beta
    return Dict("name"=>String(record["name"]),"verified"=>lower>=0,
        "absolute_error_bound"=>string(beta),"nonnegative_lower_bound"=>string(lower))
end

function verify_implication_bundle(bundle)
    String(bundle["schema"])=="dt-ibc-implication-exact-v1" || error("unknown implication schema")
    String(bundle["direction"])=="forward" || error("only forward implication replay is implemented")
    P=benchmark(String(bundle["problem"])); x=P.x
    frames=[frompolydata(p,x) for p in bundle["frames"]]
    length(frames)==1 || error("this exact replay schema currently supports one terminal frame")
    b=frames[1]
    sep=rat(String(bundle["separation"])); sep>0 || error("strict separation required")
    records=bundle["proofs"]; length(records)==3 || error("expected two separation proofs and one implication proof")
    byname=Dict(String(r["name"])=>r for r in records)
    all(haskey(byname,n) for n in ("initial","unsafe","terminal_implication")) || error("missing implication obligation")
    checks=Any[]
    push!(checks,_verify_putinar_record(-b-sep,P.X0,byname["initial"],x))
    push!(checks,_verify_putinar_record(b-sep,P.Xu,byname["unsafe"],x))
    target=-compose(b,x,P.f)
    # The added generator -b is nonnegative exactly on the implication antecedent b<=0.
    push!(checks,_verify_putinar_record(target,P.X,byname["terminal_implication"],x;extra_generators=[-b]))
    domain=domain_audit(P)
    ok=all(c["verified"] for c in checks) && domain["status"]=="EXACT_INVARIANT"
    return Dict("verified"=>ok,"status"=>ok ? "EXACT_RATIONAL_VERIFIED" : "UNVERIFIED_CANDIDATE",
        "domain"=>domain,"checks"=>checks,
        "scope"=>"fixed-certificate implication replay; not free joint frame/multiplier synthesis")
end

"""Exact affine implication IBC for ImplicationGap1D.

For b=x and f=x(x+1)/2,
-f = f^2 + (x/2)^2 (1-x^2) + ((x+1)^2/2)(-x).
The final factor -x is the implication antecedent generator.
"""
function exact_forward_implication_witness()
    P=benchmark("ImplicationGap1D"); x=P.x[1]; f=P.f[1]; b=x; sep=rat(1//5)
    initial=(;name="initial",p=-b-sep,box=P.X0,kind="separation")
    unsafe=(;name="unsafe",p=b-sep,box=P.Xu,kind="separation")
    initial_record=affine_box_record(initial,P.x)
    unsafe_record=affine_box_record(unsafe,P.x)
    propagation=Dict("name"=>"terminal_implication","reserve"=>"0//1","grams"=>Any[
        gramdata(reshape(QQ[1],1,1),[f],0,P.x),
        gramdata(reshape(QQ[1],1,1),[x/2],1,P.x),
        gramdata(reshape(QQ[1//2],1,1),[x+1],2,P.x),
    ])
    bundle=Dict{String,Any}("schema"=>"dt-ibc-implication-exact-v1","problem"=>P.name,
        "direction"=>"forward","frames"=>[polydata(b,P.x)],"separation"=>string(sep),
        "source_commit"=>get(ENV,"GITHUB_SHA","local-unrecorded"),
        "proofs"=>Any[initial_record,unsafe_record,propagation],
        "origin"=>"analytical_implication_witness_with_exact_replay")
    bundle["verification"]=verify_implication_bundle(bundle)
    bundle["verification"]["verified"] || error("implication witness verification failed")
    return bundle
end
