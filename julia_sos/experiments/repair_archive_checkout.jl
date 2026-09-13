"""Repair ONLY a CRLF-expanded checkout of the committed reviewer archive.
No hash is regenerated, no certificate is recomputed, and arbitrary edits fail.
Usage: julia --project=julia_sos julia_sos/experiments/repair_archive_checkout.jl --apply
Without --apply, report a read-only diagnosis.
"""
module ArchiveCheckoutRepair
using SHA
import JSON3

function crlf_expanded(bytes::Vector{UInt8})
    out = UInt8[]
    for i in eachindex(bytes)
        if bytes[i] == 0x0a && (i == 1 || bytes[i-1] != 0x0d)
            push!(out, 0x0d)
        end
        push!(out, bytes[i])
    end
    return out
end
sha(bytes) = bytes2hex(sha256(bytes))
gitbytes(repo, revision, name) = read(`git -C $repo show $(revision * ":evidence/reviewer/" * name)`)

function repair(repo::AbstractString; apply::Bool=false)
    repo = abspath(repo)
    revision = strip(read(`git -C $repo rev-parse HEAD`, String))
    root = joinpath(repo, "evidence", "reviewer")
    for p in (joinpath(repo, "evidence"), root)
        isdir(p) && !islink(p) || error("Missing or symlinked archive directory: $p")
    end
    canonical_index = gitbytes(repo, revision, "artifact_index.json")
    index = JSON3.read(String(copy(canonical_index)))
    isempty(index["files"]) && error("Empty archive index")
    names = [String(item["name"]) for item in index["files"]]
    all(n -> occursin(r"^[A-Za-z0-9_.-]+$", n) && n != "." && n != "..", names) || error("Unsafe archive filename")
    length(unique(lowercase.(names))) == length(names) || error("Duplicate archive filenames")
    "artifact_index.json" in names && error("Index cannot index itself")
    expected = Dict(String(i["name"]) => String(i["sha256"]) for i in index["files"])
    canonical = Dict("artifact_index.json" => canonical_index)
    observed = Dict{String,Vector{UInt8}}()
    changes = String[]
    # Complete all validation before writing even one byte.
    for name in vcat(["artifact_index.json"], names)
        bytes = name == "artifact_index.json" ? canonical_index : gitbytes(repo, revision, name)
        if name != "artifact_index.json"
            occursin(r"^[0-9a-f]{64}$", expected[name]) || error("Invalid recorded digest: $name")
            sha(bytes) == expected[name] || error("Committed blob/index mismatch: $name; refusing repair")
        end
        path = joinpath(root, name)
        isfile(path) && !islink(path) || error("Missing or symlinked archived file: $name; refusing repair")
        actual = read(path)
        if actual != bytes
            actual == crlf_expanded(bytes) || error("Not a CRLF-only difference: $name; refusing ALL writes")
            push!(changes, name)
        end
        canonical[name] = bytes
        observed[name] = actual
    end
    if !apply
        println("ARCHIVE_CHECKOUT_DIAGNOSIS: ", length(names), " committed digests valid; ", length(changes), " CRLF-only files; no files modified.")
        return (; indexed=length(names), changed=length(changes), repaired=0, backup=nothing)
    end
    # Reject concurrent edits after preflight. Preserve originals outside Git.
    all(read(joinpath(root,n)) == observed[n] for n in keys(observed)) || error("Archive changed during diagnosis; refusing repair")
    backup = isempty(changes) ? nothing : mktempdir(; prefix="scl-archive-crlf-backup-", cleanup=false)
    if backup !== nothing
        for name in changes
            write(joinpath(backup, name), observed[name])
        end
        println("Original checkout bytes backed up to: ", backup)
        for name in changes
            write(joinpath(root, name), canonical[name])
        end
    end
    all(sha(read(joinpath(root,n))) == expected[n] for n in names) || error("Post-repair digest check failed; retain the backup")
    read(joinpath(root,"artifact_index.json")) == canonical_index || error("Post-repair index mismatch")
    println("ARCHIVE_CHECKOUT_REPAIRED: ", length(changes), " files; all ", length(names), " original indexed SHA256 digests verified. Run replay_all.jl next.")
    return (; indexed=length(names), changed=length(changes), repaired=length(changes), backup)
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(a -> a == "--apply", ARGS) || error("Only --apply is supported")
    ArchiveCheckoutRepair.repair(normpath(joinpath(@__DIR__,"..","..")); apply="--apply" in ARGS)
end
