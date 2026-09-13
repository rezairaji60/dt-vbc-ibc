using Test, SHA
import JSON3
include(joinpath(@__DIR__,"..","experiments","repair_archive_checkout.jl"))
using .ArchiveCheckoutRepair

@testset "Byte-exact archive under Windows-style Git checkout" begin
    source = normpath(joinpath(@__DIR__,"..",".."))
    revision = strip(read(`git -C $source rev-parse HEAD`, String))
    mktempdir() do temp
        clone = joinpath(temp,"clone")
        run(`git clone --quiet --no-checkout --no-hardlinks $source $clone`)
        run(`git -C $clone config core.autocrlf true`)
        run(`git -C $clone config core.eol crlf`)
        run(`git -C $clone -c advice.detachedHead=false checkout --quiet --detach $revision`)
        root = joinpath(clone,"evidence","reviewer")
        index = JSON3.read(read(joinpath(root,"artifact_index.json"),String))
        @test length(index["files"]) == 69
        for item in index["files"]
            @test bytes2hex(sha256(read(joinpath(root,String(item["name"]))))) == item["sha256"]
        end
        report = ArchiveCheckoutRepair.repair(clone)
        @test report.changed == 0
        @test report.indexed == 69
        indexpath=joinpath(root,"artifact_index.json")
        firstpath=joinpath(root,String(index["files"][1]["name"]))
        secondpath=joinpath(root,String(index["files"][2]["name"]))
        original_index=read(indexpath); original=read(firstpath); second=read(secondpath)
        converted=ArchiveCheckoutRepair.crlf_expanded(original)
        @test converted != original
        @test bytes2hex(sha256(converted)) != index["files"][1]["sha256"]
        write(firstpath, converted)
        write(indexpath,ArchiveCheckoutRepair.crlf_expanded(original_index))
        diagnosis=ArchiveCheckoutRepair.repair(clone)
        @test diagnosis.changed == 2
        @test read(firstpath) == converted # dry run never rewrites
        # Arbitrary corruption elsewhere must cause zero repairs.
        corrupted=copy(second); corrupted[1] = xor(corrupted[1],UInt8(1))
        write(secondpath,corrupted)
        @test_throws ErrorException ArchiveCheckoutRepair.repair(clone;apply=true)
        @test read(firstpath) == converted
        @test read(secondpath) == corrupted
        write(secondpath,second)
        repaired=ArchiveCheckoutRepair.repair(clone;apply=true)
        @test repaired.repaired == 2
        @test read(firstpath) == original
        @test read(indexpath) == original_index
        @test read(joinpath(repaired.backup,basename(firstpath))) == converted
        @test ArchiveCheckoutRepair.repair(clone;apply=true).repaired == 0
        rm(repaired.backup;recursive=true)
        # Neither the committed archive nor any tracked scientific input changed.
        @test isempty(strip(read(`git -C $clone status --porcelain -- evidence/reviewer`,String)))
    end
end
