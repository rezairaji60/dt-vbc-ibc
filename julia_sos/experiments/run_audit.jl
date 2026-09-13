using Dates, SHA
import JSON3, Pkg
include(joinpath(@__DIR__,"..","src","AuditSOS.jl"))
using .AuditSOS
out=abspath(get(ENV,"AUDIT_OUTPUT_DIR",joinpath(@__DIR__,"..","results")))
mkpath(out)
metadata=Dict("timestamp_utc"=>string(now(UTC)),"julia"=>string(VERSION),
    "tested_sha"=>get(ENV,"GITHUB_SHA","local-unrecorded"),
    "head_sha"=>get(ENV,"AUDIT_HEAD_SHA","local-unrecorded"),
    "normalization"=>"per-component coefficient l1 <= 1; sign symmetric",
    "scope"=>"current cases; robust anchored, full-domain constraints",
    "not_claimed"=>"No best margins, class nonexistence, comparative novelty or publication readiness.")
manifest=joinpath(@__DIR__,"..","Manifest.toml")
isfile(manifest) && (metadata["manifest_sha256"]=bytes2hex(sha256(read(manifest))))
write_json(joinpath(out,"run_metadata.json"),metadata)
function notice(title, obj)
    message=replace(JSON3.write(obj),"%"=>"%25","\r"=>"%0D","\n"=>"%0A")
    println("::notice title=$title::$message")
end
exact=exact_s1()
for bundle in exact
    family=String(bundle["family"])
    write_json(joinpath(out,"S1_$(family)_analytical.json"),bundle)
    transferred=Symbol(family) in (:forward_ibc,:backward_ibc) ? AuditSOS.transfer_bundle(bundle) : nothing
    if transferred!==nothing
        write_json(joinpath(out,"S1_$(family)_transported.json"),transferred)
        @assert transferred["verification"]["verified"]
    end
end
notice("S1 exact analytical result",Dict("status"=>"EXACT_RATIONAL_VERIFIED","formulations"=>4,
    "degree"=>2,"V"=>"x1^2+x2^2","B_forward"=>"V-1/10","B_backward"=>"1/10-V",
    "contraction"=>"82/125","initial_separation_bound"=>"1/50","unsafe_separation_bound"=>"341/200"))
for name in ("S1","S2_original","S2_repaired")
    d=domain_audit(benchmark(name));write_json(joinpath(out,name*"_domain.json"),d)
    notice(name*" domain",d)
end
families=(:forward_vbc,:backward_vbc,:forward_ibc,:backward_ibc)
results=Any[]
function parameters(family,rho_f,rho_b,m=3)
    family==:forward_vbc && return path_matrix(fill(inv(rat(rho_f)),m);reciprocal=true)
    family==:backward_vbc && return path_matrix(fill(rat(rho_b),m))
    family==:forward_ibc && return rat.(fill(rho_b,m))
    return rat.(fill(inv(rat(rho_f)),m))
end
for name in ("S1","S2_repaired")
    P=benchmark(name);rho_f=name=="S1" ? 4//5 : 17//20;rho_b=6//5
    for family in families
        par=parameters(family,rho_f,rho_b)
        meta,bundle=synthesize(P,family,par)
        push!(results,meta)
        prefix="$(name)_$(family)_free"
        write_json(joinpath(out,prefix*"_status.json"),meta)
        bundle!==nothing && write_json(joinpath(out,prefix*"_certificate.json"),bundle)
        if bundle!==nothing && get(meta,"exact_verified",false) && family in (:forward_ibc,:backward_ibc)
            transported=AuditSOS.transfer_bundle(bundle)
            @assert transported["verification"]["verified"]
            write_json(joinpath(out,prefix*"_transported.json"),transported)
            meta["transport_without_resynthesis_verified"]=true
        end
        notice(prefix,meta)
    end
end
# Controlled normalization ablation: all other settings match the S1 backward run.
P=benchmark("S1")
meta,bundle=synthesize(P,:backward_vbc,parameters(:backward_vbc,4//5,6//5);
    normalization=:legacy_positive_trace)
meta["experiment"]="normalization_ablation_not_legacy_collocation_reproduction"
push!(results,meta);write_json(joinpath(out,"S1_legacy_normalization_status.json"),meta)
bundle!==nothing && write_json(joinpath(out,"S1_legacy_normalization_candidate.json"),bundle)
notice("S1 normalization ablation",meta)
write_json(joinpath(out,"summary.json"),Dict("metadata"=>metadata,"results"=>results))
# A code/regression failure must not masquerade as mathematical infeasibility.
corrected=[r for r in results if r["normalization"]=="symmetric_l1"]
@assert length(corrected)==8 && all(r["status"]=="EXACT_RATIONAL_VERIFIED" for r in corrected)
println("Current-case audit complete. See exact certificates and fail-closed statuses in ",out)
