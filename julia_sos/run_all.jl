# One reviewer command, including both fresh proofs and committed archive replay.
using Dates, SHA
VERSION == v"1.10.10" || error("The frozen reviewer run requires Julia 1.10.10. Invoke your installed 1.10.10 executable directly; do not update the Manifest.")
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
include(joinpath(@__DIR__, "experiments", "release_check.jl"))
ReviewerRelease.check_inputs()
root = normpath(joinpath(@__DIR__, ".."))
# Never mix new results with a previous run or overwrite the committed evidence.
mkpath(joinpath(@__DIR__, "results"))
out = mktempdir(joinpath(@__DIR__, "results"); prefix="run-" * Dates.format(now(UTC), "yyyymmddTHHMMSS") * "-", cleanup=false)
environment = copy(ENV)
revision = "source-hashes-verified-export"
if ispath(joinpath(root, ".git"))
    revision = strip(read(`git -C $root rev-parse HEAD`, String))
elseif isfile(joinpath(root, "SOURCE_REVISION.txt"))
    candidate = strip(read(joinpath(root, "SOURCE_REVISION.txt"), String))
    occursin(r"^[0-9a-f]{40}$", candidate) && (revision = candidate)
end
environment["GITHUB_SHA"] = revision
environment["AUDIT_HEAD_SHA"] = revision
environment["AUDIT_OUTPUT_DIR"] = out
function step(relative; args=String[])
    println("\n=== ", relative, " ===")
    script = joinpath(@__DIR__, relative)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script $args`
    run(setenv(command, environment))
end
for script in ("test/runtests.jl", "test/extended.jl", "test/archive_portability.jl", "test/release_contract.jl", "experiments/replay_all.jl", "experiments/run_reviewer.jl")
    step(script)
end
step("experiments/release_check.jl"; args=[out])
ReviewerRelease.check_inputs()
import JSON3
open(joinpath(@__DIR__, "results", "LATEST_RUN.json"), "w") do io
    JSON3.write(io, Dict("status"=>"REVIEWER_RELEASE_VERIFIED", "source_commit"=>revision, "julia"=>string(VERSION), "directory"=>basename(out)))
    println(io)
end
println("\nREVIEWER_RELEASE_VERIFIED: fixed table, fresh proofs, immutable archive and portability tests passed.")
println("Current results: ", out)
