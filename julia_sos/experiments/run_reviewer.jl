# Full audit: preserve negative outcomes and distinguish analytical witnesses.
include(joinpath(@__DIR__,"run_audit.jl"))
function run_extensions(results,metadata,out)
    for name in ("BB_rotation","Logistic_adapted")
        P=benchmark(name)
        rho_f=name=="BB_rotation" ? 4//5 : 1//10
        rho_b=name=="BB_rotation" ? 6//5 : 10//1
        for family in families
            par=parameters(family,rho_f,rho_b)
            meta,bundle=synthesize(P,family,par;degree=2,order=2)
            prefix="$(name)_$(family)_free"
            push!(results,meta)
            write_json(joinpath(out,prefix*"_status.json"),meta)
            if bundle!==nothing
                write_json(joinpath(out,prefix*"_certificate.json"),bundle)
                if get(meta,"exact_verified",false) && family in (:forward_ibc,:backward_ibc)
                    transferred=AuditSOS.transfer_bundle(bundle)
                    @assert transferred["verification"]["verified"]
                    write_json(joinpath(out,prefix*"_transported.json"),transferred)
                    meta["transport_without_resynthesis_verified"]=true
                end
            end
            println(name," ",family," ",meta["status"])
        end
    end
    for name in ("S1","S2_repaired","BB_rotation","Logistic_adapted")
        P=benchmark(name)
        a=name=="S1" ? 4//5 : name=="S2_repaired" ? 17//20 : name=="BB_rotation" ? 4//5 : 1//10
        meta,bundle=synthesize(P,:forward_vbc,reshape(rat.([a]),1,1);order=3)
        meta["experiment"]="scalar_baseline"
        push!(results,meta)
        write_json(joinpath(out,name*"_scalar_status.json"),meta)
        bundle!==nothing && write_json(joinpath(out,name*"_scalar_certificate.json"),bundle)
    end
    for name in ("Rotation2","Rotation4")
        bundle=AuditSOS.exact_rotation(name)
        write_json(joinpath(out,name*"_cyclic_affine_certificate.json"),bundle)
        push!(results,Dict("problem"=>name,"family"=>"forward_vbc","degree"=>1,
            "components"=>length(bundle["B"]),"status"=>bundle["verification"]["status"],
            "experiment"=>"analytical_structural_witness_not_free_SDP","propagation_reserve"=>0))
    end

    # Analytical complementarity evidence is intentionally separate from the
    # solver-result table. Nonexistence is established by theorem hypotheses,
    # not inferred from a failed optimization run. The implication-side witness
    # is written as its own replayable exact proof bundle.
    implication_bundle=AuditSOS.exact_forward_implication_witness()
    @assert implication_bundle["verification"]["verified"]
    write_json(joinpath(out,"ImplicationGap1D_forward_implication_ibc_analytical.json"),implication_bundle)
    complementarity=AuditSOS.complementarity_report()
    @assert complementarity["verified"]
    write_json(joinpath(out,"complementarity_report.json"),complementarity)
    println("COMPLEMENTARITY_REPORT ",JSON3.write(complementarity))

    compact=[Dict(k=>r[k] for k in ("problem","family","degree","components","status","experiment","termination_status") if haskey(r,k)) for r in results]
    metadata["scope"]="four synthesis problems; scalar baselines; two finite-order witnesses; exact analytical complementarity evidence"
    metadata["complementarity_evidence"]="complementarity_report.json"
    metadata["implication_proof_bundle"]="ImplicationGap1D_forward_implication_ibc_analytical.json"
    write_json(joinpath(out,"reviewer_summary.json"),Dict("metadata"=>metadata,"results"=>compact))
    println("REVIEWER_SUMMARY ",JSON3.write(compact))
    # The manuscript's fixed numerical table is a regression assertion, not a
    # blanket acceptance of an arbitrary solver result. The analytical
    # complementarity report is checked independently by the release contract.
    accepted=count(r->r["status"]=="EXACT_RATIONAL_VERIFIED",compact)
    accepted==22 || error("Published reviewer table changed: expected 22 exact-positive rows, got $accepted")
end
run_extensions(results,metadata,out)
