using SHA
import JSON3
include(joinpath(@__DIR__,"..","src","AuditSOS.jl"))
using .AuditSOS
root=isempty(ARGS) ? normpath(joinpath(@__DIR__,"..","..","evidence","reviewer")) : abspath(ARGS[1])
indexfile=joinpath(root,"artifact_index.json")
isfile(indexfile) || error("Missing artifact_index.json; use the committed evidence archive.")
idx=JSON3.read(read(indexfile,String))
count=0
for item in idx["files"]
    name=String(item["name"])
    basename(name)==name || error("Unsafe archive path")
    path=joinpath(root,name)
    isfile(path) || error("Missing archived file: $name")
    bytes2hex(sha256(read(path)))==String(item["sha256"]) || error("Digest mismatch: $name")
    if endswith(name,".json")
        obj=JSON3.read(read(path,String))
        if haskey(obj,"schema") && obj["schema"]=="dt-vbc-sos-exact-v1"
            result=verify_bundle(obj)
            result["verified"] || error("Exact replay failed: $name")
            count+=1
        end
    end
end
count>0 || error("No proof bundles found")
println("ARCHIVE_REPLAY_VERIFIED: ",count," proof bundles; all indexed SHA256 digests checked.")
