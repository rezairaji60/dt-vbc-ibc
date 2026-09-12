using Test, LinearAlgebra, JuMP, SumOfSquares, DynamicPolynomials
import JSON3, CSDP
include(joinpath(@__DIR__,"..","src","AuditSOS.jl"))
using .AuditSOS
const MOI=JuMP.MOI
@testset "Exact arithmetic verifier rejects invalid witnesses" begin
    @test psd_exact(QQ[0 0;0 1])
    @test !psd_exact(QQ[0 1;1 0])
    @test !psd_exact(QQ[1 0;0 -1//1000000000000])
    @test !psd_exact(QQ[1 1;0 1])
    @test_throws ArgumentError path_matrix([1,0])
end
@testset "Domains and trajectories" begin
    @test domain_audit(benchmark("S1"))["status"]=="EXACT_INVARIANT"
    @test domain_audit(benchmark("S2_original"))["status"]=="NOT_INVARIANT"
    @test domain_audit(benchmark("S2_repaired"))["status"]=="EXACT_INVARIANT"
    P=benchmark("S2_original")
    @test compose(P.f[1],P.x,fill(rat(3//25),2)) > rat(3//25)
end
@testset "Exact S1 proofs, replay and tamper rejection" begin
    bundles=exact_s1()
    @test length(bundles)==4
    for b in bundles
        @test verify_bundle(b)["verified"]
        replay=JSON3.read(JSON3.write(b))
        @test verify_bundle(replay)["verified"]
        if Symbol(b["family"]) in (:forward_ibc,:backward_ibc)
            @test AuditSOS.transfer_bundle(b)["verification"]["verified"]
        end
    end
    bad=deepcopy(bundles[1])
    bad["proofs"][1]["grams"][1]["Q"][1][1]="-1//1"
    @test_throws ErrorException verify_bundle(bad)
end
@testset "IBC mapping, including reciprocal and terminal self-loop" begin
    P=benchmark("S1");x,y=P.x
    for m in (1,2,3)
        b=[rat(i)+x^2-rat(i+1)*y+x*y for i in 1:m]
        l=[rat(i+1) for i in 1:m]
        source=obligations(P,:backward_ibc,b,l)
        target=obligations(P,:forward_vbc,-b,path_matrix(l;reciprocal=true))
        p=Dict(o.name=>o.p for o in source);q=Dict(o.name=>o.p for o in target)
        for i in 1:m
            @test iszero(p["propagation_$i"]-l[i]*q["propagation_$i"])
        end
        s2=obligations(P,:forward_ibc,b,l)
        t2=obligations(P,:backward_vbc,-b,path_matrix(l))
        a=Dict(o.name=>o.p for o in s2);d=Dict(o.name=>o.p for o in t2)
        @test all(iszero(a[k]-d[k]) for k in keys(a))
        # The manuscript's non-reciprocal bIBC matrix fails this identity.
        wrong=obligations(P,:forward_vbc,-b,path_matrix(l))
        w=Dict(o.name=>o.p for o in wrong)
        @test !iszero(p["propagation_$m"]-l[m]*w["propagation_$m"])
    end
end
@testset "Real SDP/SOS runtime" begin
    @polyvar t
    model=SOSModel(CSDP.Optimizer);set_silent(model)
    @variable(model,alpha)
    @constraint(model,t^2+1-alpha in SOSCone())
    @objective(model,Max,alpha)
    optimize!(model)
    @test termination_status(model)==MOI.OPTIMAL
    @test isapprox(value(alpha),1;atol=1e-6)
    P=benchmark("S1")
    meta,bundle=synthesize(P,:forward_vbc,reshape(rat.([4//5]),1,1))
    @test bundle!==nothing
    @test meta["status"]=="EXACT_RATIONAL_VERIFIED"
    @test verify_bundle(JSON3.read(JSON3.write(bundle)))["verified"]
end
println("::notice title=Julia tests::Exact proofs, domain checks, mappings, tamper rejection and genuine SOS tests passed.")
