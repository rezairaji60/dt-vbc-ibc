"""Release checks are separate from mathematical verification. They bind the
fixed reviewer scope and immutable archived evidence, without changing proofs.
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

canonical_text(path) = replace(read(path, String), "\r\n" => "\n")
function git_blob_hash(path)
    bytes = Vector{UInt8}(codeunits(canonical_text(path)))
    return bytes2hex(sha1(vcat(Vector{UInt8}(codeunits("blob $(length(bytes))\0")), bytes)))
end
function contract(root=ROOT)
    return TOML.parsefile(joinpath(root, "docs", "REVIEWER_RELEASE.toml"))
end
function check_inputs(root=ROOT)
    spec = contract(root)
    spec["schema"] == "scl-reviewer-release-v1" || error("Unknown release contract")
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
    println("RELEASE_INPUTS_VERIFIED: ", length(spec["science_blobs"]), " scientific files; ", length(names), " archived digests.")
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
function check_generated(out, replay)
    summary = JSON3.read(read(joinpath(out, "reviewer_summary.json"), String))
    check_rows(summary["results"])
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
    println("CURRENT_RUN_REPLAY_VERIFIED: ", length(found), " freshly generated proof bundles, including all transports.")
    return true
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ReviewerRelease.check_inputs()
    if !isempty(ARGS)
        length(ARGS) == 1 || error("Pass at most one current-results directory")
        include(joinpath(@__DIR__, "..", "src", "AuditSOS.jl"))
        ReviewerRelease.check_generated(abspath(ARGS[1]), AuditSOS.verify_bundle)
    end
end
