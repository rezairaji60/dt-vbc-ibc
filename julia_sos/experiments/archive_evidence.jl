using SHA
import JSON3
repo=normpath(joinpath(@__DIR__,"..",".."))
source=readchomp(Cmd(["git","rev-parse","HEAD"];dir=repo))
out=joinpath(repo,"evidence","reviewer")
mkpath(out)
# This is a one-time, guarded archive. Later runs never silently replace it.
isfile(joinpath(out,"artifact_index.json")) && error("Archive already exists")
results=joinpath(repo,"julia_sos","results")
for name in readdir(results)
    (endswith(name,".json") || endswith(name,".log")) || continue
    cp(joinpath(results,name),joinpath(out,name);force=false)
end
for name in ("Project.toml","Manifest.toml")
    cp(joinpath(repo,"julia_sos",name),joinpath(out,name);force=false)
end
write(joinpath(out,"SOURCE_COMMIT.txt"),source*"\n")
files=[Dict("name"=>n,"sha256"=>bytes2hex(sha256(read(joinpath(out,n))))) for n in sort(readdir(out))]
open(joinpath(out,"artifact_index.json"),"w") do io
    JSON3.write(io,Dict("source_commit"=>source,"run_id"=>get(ENV,"GITHUB_RUN_ID","local"),"files"=>files))
    println(io)
end
println("Archived evidence for source ",source)
