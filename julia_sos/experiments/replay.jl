import JSON3
include(joinpath(@__DIR__,"..","src","AuditSOS.jl"))
using .AuditSOS
length(ARGS)==1 || error("Usage: julia --project=julia_sos julia_sos/experiments/replay.jl CERTIFICATE.json")
bundle=JSON3.read(read(ARGS[1],String))
result=verify_bundle(bundle)
println(JSON3.write(result))
result["verified"] || exit(1)
