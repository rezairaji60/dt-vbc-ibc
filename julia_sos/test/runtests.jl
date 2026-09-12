using Test, SumOfSquares, DynamicPolynomials, JuMP
import CSDP
@testset "Genuine SOS runtime smoke test" begin
    @polyvar x
    model = SOSModel(CSDP.Optimizer)
    set_silent(model)
    @variable(model, t)
    @constraint(model, x^2 + 1 - t in SOSCone())
    @objective(model, Max, t)
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test isapprox(value(t), 1.0; atol=1e-6)
end
println("BOOTSTRAP_ONLY: benchmark implementation is not yet complete.")
