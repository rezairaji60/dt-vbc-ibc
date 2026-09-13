"""Release checks are separate from mathematical verification. They bind the
reviewer scope and immutable archived evidence, without changing proofs.
"""
module ReviewerRelease
using SHA, TOML
import JSON3
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PROBLEMS = ("S1", "S2_repaired", "BB_rotation", "Logistic_adapted")
const FAMILIES = ("forward_vbc", "backward_vbc", "forward_ibc", "backward_ibc")
const POSITIVE = "EXACT_RATIONAL_VERIFIED"
const NEGATIVE = "NO_CERTIFIED_CANDIDATE"
const SCALAR = "scalar_baseline"
const ANALYTICAL = "analytical_structural_witness_not_free_SDP"
const ABLATION = "normalization_ablation_not_legacy_collocation_reproduction"
const FIXED_TITLE = "Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity"

canonical_text(path) = replace(read(path, String), "\r\n" => "\n")
function git_blob_hash(path)
    bytes = Vector{UInt8}(codeunits(canonical_text(path)))
    return bytes2hex(sha1(vcat(Vector{UInt8}(codeunits("blob $(length(bytes))\0")), bytes)))
end
function contract(root=ROOT)
    return TOML.parsefile(joinpath(root, "docs", "REVIEWER_RELEASE.toml"))
end
function check_complementarity(report)
    get(report,"schema","")=="dt-vbc-ibc-complementarity-report-v1" || error("Unknown complementarity schema")
    get(report,"verified",false)==true || error("Complementarity report not verified")
    String(report["title"])==FIXED_TITLE || error("Fixed paper title changed")
    rotations=report["rotation_degree_separation"]
    length(rotations)==2 || error("Expected two rotation degree-separation cases")
    for r in rotations
        String(r["problem"]) in ("Rotation2","Rotation4") || error("Unexpected rotation case")
        Int(r["vbc"]["minimum_degree"])==1 || error("Rotation VBC degree regression")
        Int(r["ibc"]["minimum_degree"])==2 || error("Rotation IBC degree regression")
        get(r["ibc"],"affine_excluded_by_orbit_average_theorem",false)==true || error("Missing affine IBC obstruction")
        get(r,"obstruction_hypotheses_verified",false)==true || error("Rotation obstruction hypotheses not verified")
    end
    gap=report["implication_gap"]
    String(gap["problem"])=="ImplicationGap1D" || error("Wrong implication-gap benchmark")
    Int(gap["implication_ibc"]["degree"])==1 || error("Affine implication IBC missing")
    get(gap["implication_ibc"],"identity_verified",false)==true || error("Implication identity not verified")
    String(gap["implication_ibc"]["exact_replay_status"])==POSITIVE || error("Implication exact replay missing")
    get(gap["affine_global_vbc_obstruction"],"verified_hypotheses",false)==true || error("Affine VBC obstruction hypotheses missing")
    String(gap["affine_global_vbc_obstruction"]["weighted_state"][1])=="-1//2" || error("Obstruction initial barycenter changed")
    String(gap["affine_global_vbc_obstruction"]["weighted_image"][1])=="1//4" || error("Obstruction unsafe barycenter changed")
    Int(gap["global_vbc_recovery"]["degree"])==4 || error("Quartic recovery witness missing")
    get(gap["global_vbc_recovery"],"propagation_identity_verified",false)==true || error("Quartic identity not verified")
    profiles=report["complexity_accounting"]
    length(profiles)==3 || error("Complexity accounting changed")
    bydegree=Dict(Int(p["certificate_degree"])=>p for p in profiles)
    Int(bydegree[2]["dense_coefficients_per_component"])==28 || error("Degree-two count changed")
    Int(bydegree[4]["dense_coefficients_per_component"])==210 || error("Degree-four count changed")
    Int(bydegree[2]["symmetric_gram_entries"])==406 || error("Degree-two Gram count changed")
    Int(bydegree[4]["symmetric_gram_entries"])==22155 || error("Degree-four Gram count changed")
    return true
end
function check_inputs(root=ROOT)
    spec = contract(root)
    spec["schema"] == "scl-reviewer-release-v2" || error("Unknown release contract")
    for (path, expected) in spec["science_blobs"]
        git_blob_hash(joinpath(root, path)) == expected || error("Frozen scientific input changed: $path")
    end
    for path in spec["required_documentation"]
        isfile(joinpath(root, path)) || error("Missing reviewer documentation: $path")
    end
    index_path = joinpath(root, "evidence", "reviewer", "artifact_index.json")
    bytes2hex(sha256(read(index_path))) == spec["archive_index_sha256"] ||
        error("Archive index bytes differ. On an old Windows clone, run repair_archive_checkout.jl --apply; do not regenerate hashes.")
    idx = JSON3.read(read(index_path, String))
    names = String[String(item["name"]) for item in idx["files"]]
    length(names) == spec["indexed_files"] || error("Unexpected archive size")
    length(unique(lowercase.(names))) == length(names) || error("Duplicate archive entries")
    for item in idx["files"]
        name = String(item["name"])
        occursin(r"^[A-Za-z0-9_.-]+$", name) && name != "." && name != ".." || error("Unsafe archive path")
        bytes2hex(sha256(read(joinpath(dirname(index_path), name)))) == item["sha256"] ||
            error("Archive digest mismatch: $name; use the documented checkout diagnosis, never replace expected hashes")
    end
    cpath=joinpath(root,"evidence","complementarity","complementarity_report.json")
    bytes2hex(sha256(read(cpath))) == spec["complementarity_evidence_sha256"] || error("Complementarity evidence bytes changed")
    check_complementarity(JSON3.read(read(cpath,String)))
    println("RELEASE_INPUTS_VERIFIED: ", length(spec["science_blobs"]), " scientific files; ", length(names), " archived digests; complementarity evidence verified.")
    return true
end
function expected_rows()
    expected = Dict{Tuple{String,String,String},Tuple{Int,Int,String}}()
    for problem in PROBLEMS, family in FAMILIES
        expected[(problem, family, "multi_function")] = (2, 3, POSITIVE)
    end
    for problem in PROBLEMS
        expected[(problem, "forward_vbc", SCALAR)] = (2, 1, POSITIVE)
    end
    for (problem, n) in (("Rotation2", 4), ("Rotation4", 8))
        expected[(problem, "forward_vbc", ANALYTICAL)] = (1, n, POSITIVE)
    end
    expected[("S1", "backward_vbc", ABLATION)] = (2, 3, NEGATIVE)
    return expected
end
function check_rows(rows)
    expected = expected_rows()
    length(rows) == length(expected) || error("Reviewer table must contain exactly 23 identified rows")
    seen = Set{Tuple{String,String,String}}()
    for row in rows
        key = (String(row["problem"]), String(row["family"]), String(get(row, "experiment", "multi_function")))
        key in seen && error("Duplicate reviewer row: $key")
        haskey(expected, key) || error("Unexpected reviewer row: $key")
        observed = (Int(row["degree"]), Int(row["components"]), String(row["status"]))
        observed == expected[key] || error("Reviewer result changed: $key: $observed")
        push!(seen, key)
    end
    println("REVIEWER_TABLE_VERIFIED: 16 multi-function, 4 scalar, 2 analytical, 1 intentional negative row.")
    return true
end
function check_generated(out, replay, implication_replay)
    summary = JSON3.read(read(joinpath(out, "reviewer_summary.json"), String))
    check_rows(summary["results"])
    comp_path=joinpath(out,"complementarity_report.json")
    isfile(comp_path) || error("Missing generated complementarity report")
    check_complementarity(JSON3.read(read(comp_path,String)))
    expected = Set{String}()
    for problem in PROBLEMS, family in FAMILIES
        push!(expected, "$(problem)_$(family)_free_certificate.json")
        if family in ("forward_ibc", "backward_ibc")
            push!(expected, "$(problem)_$(family)_free_transported.json")
        end
    end
    for problem in PROBLEMS; push!(expected, "$(problem)_scalar_certificate.json"); end
    for family in FAMILIES; push!(expected, "S1_$(family)_analytical.json"); end
    for family in ("forward_ibc", "backward_ibc"); push!(expected, "S1_$(family)_transported.json"); end
    for problem in ("Rotation2", "Rotation4"); push!(expected, "$(problem)_cyclic_affine_certificate.json"); end
    found = Set{String}()
    for name in readdir(out)
        endswith(name, ".json") || continue
        obj = JSON3.read(read(joinpath(out, name), String))
        if get(obj, "schema", "") == "dt-vbc-sos-exact-v1"
            name in expected || error("Unexpected proof bundle in current run: $name")
            replay(obj)["verified"] || error("Generated proof replay failed: $name")
            push!(found, name)
        end
    end
    found == expected || error("Missing generated proof bundles: $(setdiff(expected, found))")
    implication_path=joinpath(out,"ImplicationGap1D_forward_implication_ibc_analytical.json")
    isfile(implication_path) || error("Missing generated implication proof bundle")
    implication=JSON3.read(read(implication_path,String))
    implication_replay(implication)["verified"] || error("Generated implication proof replay failed")
    println("CURRENT_RUN_REPLAY_VERIFIED: ", length(found), " globally scaled/VBC bundles plus one implication-IBC bundle; all transports and complementarity evidence verified.")
    return true
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ReviewerRelease.check_inputs()
    if !isempty(ARGS)
        length(ARGS) == 1 || error("Pass at most one current-results directory")
        include(joinpath(@__DIR__, "..", "src", "AuditSOS.jl"))
        ReviewerRelease.check_generated(abspath(ARGS[1]), AuditSOS.verify_bundle, AuditSOS.verify_implication_bundle)
    end
end
