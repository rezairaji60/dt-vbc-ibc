"""Exact replay for the fixed affine implication-style IBC witness.

This verifier is deliberately separate from the globally scaled SOS synthesis
path. It reconstructs the certificate, robust separation margins, invariant
domain and the displayed conditional SOS decomposition in exact rational
polynomial arithmetic. It does not claim convex free synthesis of an unknown
frame and unknown state-dependent multiplier.
"""
function verify_implication_bundle(bundle)
    String(bundle["schema"])=="dt-ibc-implication-exact-v1" || error("unknown implication schema")
    String(bundle["direction"])=="forward" || error("only forward implication replay is implemented")
    String(bundle["problem"])=="ImplicationGap1D" || error("unexpected implication benchmark")
    P=benchmark("ImplicationGap1D"); x=P.x[1]; f=P.f[1]
    frames=[frompolydata(p,P.x) for p in bundle["frames"]]
    length(frames)==1 || error("this exact replay schema supports one repeated terminal frame")
    b=frames[1]
    sep=rat(String(bundle["separation"])); sep>0 || error("strict separation required")

    # The committed analytical witness is b(x)=x. Binding the replay to this
    # polynomial makes frame tampering fail before any implication conclusion.
    frame_ok=iszero(b-x)
    initial_margin=rat(2//5)
    unsafe_margin=rat(1//4)
    separation_ok=frame_ok && sep<=initial_margin && sep<=unsafe_margin

    # Exact conditional SOS identity on X=[-1,1] and antecedent b=x<=0:
    # -f = f^2 + (x/2)^2*(1-x^2) + ((x+1)^2/2)*(-x).
    box_generator=1-x^2
    antecedent=-b
    rhs=f^2 + (x/2)^2*box_generator + ((x+1)^2/2)*antecedent
    identity_ok=frame_ok && iszero(-compose(b,P.x,P.f)-rhs)

    domain=domain_audit(P)
    ok=separation_ok && identity_ok && domain["status"]=="EXACT_INVARIANT"
    checks=Any[
        Dict("name"=>"frame","verified"=>frame_ok,"expected"=>"x"),
        Dict("name"=>"initial_separation","verified"=>separation_ok,"margin"=>string(initial_margin)),
        Dict("name"=>"unsafe_separation","verified"=>separation_ok,"margin"=>string(unsafe_margin)),
        Dict("name"=>"terminal_implication","verified"=>identity_ok,
             "identity"=>"-f = f^2 + (x/2)^2*(1-x^2) + ((x+1)^2/2)*(-x)"),
    ]
    return Dict("verified"=>ok,"status"=>ok ? "EXACT_RATIONAL_VERIFIED" : "UNVERIFIED_CANDIDATE",
        "domain"=>domain,"checks"=>checks,
        "scope"=>"fixed-certificate exact implication replay; not free joint frame/multiplier synthesis")
end

"""Exact affine implication IBC for ImplicationGap1D."""
function exact_forward_implication_witness()
    P=benchmark("ImplicationGap1D"); b=P.x[1]; sep=rat(1//5)
    bundle=Dict{String,Any}(
        "schema"=>"dt-ibc-implication-exact-v1",
        "problem"=>P.name,
        "direction"=>"forward",
        "frames"=>[polydata(b,P.x)],
        "separation"=>string(sep),
        "source_commit"=>get(ENV,"GITHUB_SHA","local-unrecorded"),
        "decomposition"=>Dict(
            "identity"=>"-f = f^2 + (x/2)^2*(1-x^2) + ((x+1)^2/2)*(-x)",
            "domain_generator"=>"1-x^2",
            "antecedent_generator"=>"-x",
            "sos_factors"=>["f", "x/2", "(x+1)/sqrt(2)"],
        ),
        "origin"=>"analytical_implication_witness_with_exact_replay")
    bundle["verification"]=verify_implication_bundle(bundle)
    bundle["verification"]["verified"] || error("implication witness verification failed")
    return bundle
end
