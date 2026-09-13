# Single entrypoint. Separate processes avoid module redefinition across tests.
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
for relative in ("test/runtests.jl", "test/extended.jl", "experiments/run_reviewer.jl")
    script=joinpath(@__DIR__,relative)
    run(`$(Base.julia_cmd()) --project=$(@__DIR__) $script`)
end
println("Completed reviewer suite. Results are in julia_sos/results/.")
