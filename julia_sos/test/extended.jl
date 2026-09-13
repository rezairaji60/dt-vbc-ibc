using Test, LinearAlgebra
include(joinpath(@__DIR__,"..","src","AuditSOS.jl"))
using .AuditSOS
@testset "Literature provenance and exact domain guards" begin
    for name in AuditSOS.EXTRA_CASES
        P=benchmark(name)
        @test domain_audit(P)["status"]=="EXACT_INVARIANT"
        @test !domain_audit(P)["initial_box_invariant"]
        bad=merge(P,(;f=[p+1 for p in P.f]))
        @test domain_audit(bad)["status"]=="DOMAIN_NOT_ESTABLISHED"
    end
    P=benchmark("BB_rotation");x,y=P.x
    @test iszero(P.f[1]+x*0+y/100)
    @test iszero(P.f[2]-x/100)
    @test P.X0==AuditSOS.box([(1//10,2//5),(1//10,11//20)])
    P=benchmark("Logistic_adapted")
    @test iszero(compose(P.f[1],P.x,[rat(1//2),rat(1//2)])-rat(4//5))
    @test iszero(compose(P.f[2],P.x,[rat(1//2),rat(1//2)])-rat(7//10))
end
@testset "Finite-order non-path witnesses" begin
    for name in ("Rotation2","Rotation4")
        P=benchmark(name);x=P.x;n=length(x)
        f4=reduce((a,b)->[compose(p,x,b) for p in a],fill(P.f,4))
        @test all(iszero(f4[i]-x[i]) for i in 1:n)
        b=AuditSOS.exact_rotation(name)
        @test verify_bundle(b)["verified"]
        @test length(b["B"])==2*n
        @test maximum(AuditSOS.MP.maxdegree(frompolydata(p,x)) for p in b["B"])==1
        @test all(r["absolute_error_bound"]=="0//1" for r in b["verification"]["checks"])
        R=zeros(QQ,n,n)
        for i in 1:n; R[i,isodd(i) ? i+1 : i-1]=isodd(i) ? -1 : 1; end
        @test det(R-I)!=0
        bad=deepcopy(b);bad["B"][1][1]["c"]="100//1"
        @test !verify_bundle(bad)["verified"]
    end
end
@testset "Canonical path identities and reject non-path matrices" begin
    A=path_matrix([2//1,3//1,5//1])
    C,D=AuditSOS.canonical_path(A)
    @test C[1,2]==1 && C[2,3]==1 && C[3,3]==5
    @test C*D==D*A
    @test_throws ArgumentError AuditSOS.canonical_path(QQ[1 1;1 1])
    P=benchmark("S1");B=[P.x[1]+i*P.x[2]-1 for i in 1:3]
    @test all(iszero(v) for v in D*(A*B-[compose(p,P.x,P.f) for p in B])-
        (C*(D*B)-[compose(p,P.x,P.f) for p in D*B]))
end
