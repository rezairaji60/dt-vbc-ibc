using Test
import JSON3
include(joinpath(@__DIR__, "..", "experiments", "release_check.jl"))
using .ReviewerRelease
@testset "Frozen reviewer scope rejects misleading success" begin
    root = ReviewerRelease.ROOT
    @test ReviewerRelease.check_inputs(root)
    path = joinpath(root, "evidence", "reviewer", "reviewer_summary.json")
    summary = JSON3.read(read(path, String))
    rows = [Dict{String,Any}(string(k)=>v for (k,v) in pairs(row)) for row in summary["results"]]
    @test ReviewerRelease.check_rows(rows)
    @test length(rows) == 23
    @test count(r -> r["status"] == ReviewerRelease.POSITIVE, rows) == 22
    @test_throws ErrorException ReviewerRelease.check_rows(rows[1:end-1])
    duplicate = deepcopy(rows); duplicate[end] = deepcopy(duplicate[1])
    @test_throws ErrorException ReviewerRelease.check_rows(duplicate)
    failed = deepcopy(rows); failed[1]["status"] = "OPTIMAL"
    @test_throws ErrorException ReviewerRelease.check_rows(failed)
    wrong_degree = deepcopy(rows); wrong_degree[1]["degree"] = 4
    @test_throws ErrorException ReviewerRelease.check_rows(wrong_degree)
    unexpected = deepcopy(rows); unexpected[1]["problem"] = "unreported_problem"
    @test_throws ErrorException ReviewerRelease.check_rows(unexpected)
    hidden_negative = deepcopy(rows)
    only(r for r in hidden_negative if r["status"] == ReviewerRelease.NEGATIVE)["status"] = ReviewerRelease.POSITIVE
    @test_throws ErrorException ReviewerRelease.check_rows(hidden_negative)
    mktempdir() do temp
        # LF/CRLF equivalence is allowed only for source-text fingerprints.
        a=joinpath(temp,"a"); b=joinpath(temp,"b")
        write(a,"line1\nline2\n"); write(b,"line1\r\nline2\r\n")
        @test ReviewerRelease.git_blob_hash(a) == ReviewerRelease.git_blob_hash(b)
        write(b,"line1\nchanged\n")
        @test ReviewerRelease.git_blob_hash(a) != ReviewerRelease.git_blob_hash(b)
    end
end
