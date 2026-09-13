module AuditSOS
using LinearAlgebra, Dates, SHA
using JuMP, SumOfSquares, DynamicPolynomials
import MultivariatePolynomials as MP
import JSON3, CSDP
const MOI = JuMP.MOI
const QQ = Rational{BigInt}
include("problems.jl")
include("benchmarks.jl")
include("obligations.jl")
include("exact.jl")
include("synthesis.jl")
include("structural.jl")
include("implication.jl")
include("complementarity.jl")
export benchmark, obligations, path_matrix, compose, exact_s1, domain_audit,
       synthesize, verify_bundle, write_json, QQ, rat, polydata, frompolydata,
       psd_exact, certify_gram, nonnegative_bound, putinar!,
       verify_implication_bundle, exact_forward_implication_witness,
       exact_implication_gap, affine_vbc_obstruction, rotation_degree_report,
       complexity_profile, complementarity_report
end
