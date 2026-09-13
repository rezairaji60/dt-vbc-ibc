using Test, SHA
import JSON3
include(joinpath(@__DIR__,"..","experiments","repair_archive_checkout.jl"))
using .ArchiveCheckoutRepair

@testset "Byte-exact archive under Windows-style Git checkout" begin
    source = normpath(joinpath(@__DIR__,"..",".."))
    mktempdir() do temp
        # Create a local fixture so the same test works from a source ZIP.
        # No user Git history, credentials, remotes or global config is altered.
        seed = joinpath(temp, "seed")
        mkpath(joinpath(seed, "evidence"))
        cp(joinpath(source, "evidence", "reviewer"), joinpath(seed, "evidence", "reviewer"))
        cp(joinpath(source, "evidence", "complementarity"), joinpath(seed, "evidence", "complementarity"))
        cp(joinpath(source, ".gitattributes"), joinpath(seed, ".gitattributes"))
        run(`git init --quiet $seed`)
        run(`git -C $seed add .gitattributes evidence`)
        run(`git -C $seed -c user.name=ReviewerFixture -c user.email=fixture@example.invalid -c commit.gpgsign=false commit --quiet -m fixture`)
        revision = strip(read(`git -C $seed rev-parse HEAD`, String))
        clone = joinpath(temp,"clone")
        run(`git clone --quiet --no-checkout --no-hardlinks $seed $clone`)
        run(`git -C $clone config core.autocrlf true`)
        run(`git -C $clone config core.eol crlf`)
        run(`git -C $clone -c advice.detachedHead=false checkout --quiet --detach $revision`)
        root = joinpath(clone,"evidence","reviewer")
        index = JSON3.read(read(joinpath(root,"artifact_index.json"),String))
        @test length(index["files"]) == 69
        for item in index["files"]
            @test bytes2hex(sha256(read(joinpath(root,String(item["name"]))))) == item["sha256"]
        end
        complementarity = joinpath(clone,"evidence","complementarity","complementarity_report.json")
        @test bytes2hex(sha256(read(complementarity))) == "2b87acfc0fb109667e9cd9a8b1ce45e7778e58a42b91b05424434a2f8a2622d9"
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
        @test read(firstpath) == converted
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
        @test isempty(strip(read(`git -C $clone status --porcelain -- evidence/reviewer evidence/complementarity`,String)))
    end
end
