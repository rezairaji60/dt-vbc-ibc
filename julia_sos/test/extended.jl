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
    @test iszero(P.f[1]+rat(1//100)*y)
    @test iszero(P.f[2]-rat(1//100)*x)
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
@testset "Orbit-average obstruction and its spectral assumption" begin
    # These tests support the algebra in the implication-IBC theorem;
    # they are not a mechanization of its universal mathematical proof.
    for n in (2,4)
        R=zeros(QQ,n,n)
        for i in 1:n; R[i,isodd(i) ? i+1 : i-1]=isodd(i) ? -1 : 1; end
        orbit_sum=zeros(QQ,n,n)
        for j in 0:3; orbit_sum+=R^j; end
        @test orbit_sum==zeros(QQ,n,n)
    end
    R=QQ[0 -1 0;1 0 0;0 0 1]
    orbit_sum=sum(R^j for j in 0:3)
    @test det(R-I)==0
    @test orbit_sum!=zeros(QQ,3,3)
    @test orbit_sum[3,3]==4
end

@testset "Complementary expressiveness: implication IBC versus global VBC" begin
    P=benchmark("ImplicationGap1D")
    @test length(P.x)==1
    @test domain_audit(P)["status"]=="EXACT_INVARIANT"
    report=exact_implication_gap()
    @test report["verified"]
    @test report["implication_ibc"]["degree"]==1
    @test report["implication_ibc"]["identity_verified"]
    @test report["implication_ibc"]["exact_replay_status"]=="EXACT_RATIONAL_VERIFIED"
    bundle=exact_forward_implication_witness()
    @test verify_implication_bundle(bundle)["verified"]
    damaged=deepcopy(bundle)
    damaged["frames"][1][1]["c"]="100//1"
    @test !verify_implication_bundle(damaged)["verified"]
    @test report["affine_global_vbc_obstruction"]["verified_hypotheses"]
    @test report["affine_global_vbc_obstruction"]["weighted_state"]==["-1//2"]
    @test report["affine_global_vbc_obstruction"]["weighted_image"]==["1//4"]
    @test report["global_vbc_recovery"]["degree"]==4
    @test report["global_vbc_recovery"]["propagation_identity_verified"]
    bad=merge(P,(;Xu=AuditSOS.box([(2//5,1//2)])))
    @test !affine_vbc_obstruction(bad,[[-1],[1]],rat.([3//4,1//4]))["verified_hypotheses"]
end

@testset "Tight rotation degree separation" begin
    for name in ("Rotation2","Rotation4")
        r=rotation_degree_report(name)
        @test r["verified"]
        @test r["vbc"]["minimum_degree"]==1
        @test r["ibc"]["minimum_degree"]==2
        @test r["ibc"]["quadratic_invariant"]
        @test r["obstruction_hypotheses_verified"]
    end
end

@testset "Transparent polynomial/SOS complexity accounting" begin
    q2=complexity_profile(6,2,1;dynamics_degree=2)
    q4=complexity_profile(6,4,1;dynamics_degree=2)
    @test q2["dense_coefficients_per_component"]==28
    @test q4["dense_coefficients_per_component"]==210
    @test q2["degree_matched_dense_gram_dimension"]==28
    @test q4["degree_matched_dense_gram_dimension"]==210
    @test q2["symmetric_gram_entries"]==406
    @test q4["symmetric_gram_entries"]==22155
    full=complementarity_report()
    @test full["verified"]
    @test length(full["rotation_degree_separation"])==2
    @test length(full["complexity_accounting"])==3
end
