# Complementarity evidence used by the revised paper story.
# These functions verify concrete algebraic hypotheses/witnesses. They are not
# a proof-assistant formalization of the universal theorems stated in the paper.

function _inside_box(point,K)
    length(point)==length(K) || return false
    return all(K[i][1] <= point[i] <= K[i][2] for i in eachindex(K))
end

function _eval_poly(p,x,point)
    v=MP.subs(p,x=>point)
    v isa Number || error("expected a scalar polynomial evaluation")
    return rat(v)
end

"""Verify the finite convex-combination data used by the affine global-VBC
obstruction theorem. If xbar is initial and ybar is unsafe, affine global
comparison with a constant nonnegative A would also have to respect the
spurious transition xbar -> ybar, which contradicts separation.
"""
function affine_vbc_obstruction(P,points,weights)
    length(points)==length(weights) && !isempty(points) || throw(ArgumentError("points/weights"))
    all(w->w>=0,weights) || throw(ArgumentError("weights must be nonnegative"))
    sum(weights)==1 || throw(ArgumentError("weights must sum to one"))
    n=length(P.x)
    all(length(p)==n for p in points) || throw(DimensionMismatch("point dimension"))
    xbar=[sum(weights[j]*rat(points[j][i]) for j in eachindex(points)) for i in 1:n]
    images=[[_eval_poly(P.f[i],P.x,rat.(points[j])) for i in 1:n] for j in eachindex(points)]
    ybar=[sum(weights[j]*images[j][i] for j in eachindex(points)) for i in 1:n]
    initial=_inside_box(xbar,P.X0); unsafe=_inside_box(ybar,P.Xu)
    return Dict("verified_hypotheses"=>initial && unsafe,
        "weights"=>string.(weights),"points"=>[string.(rat.(p)) for p in points],
        "weighted_state"=>string.(xbar),"weighted_image"=>string.(ybar),
        "weighted_state_in_initial"=>initial,"weighted_image_in_unsafe"=>unsafe,
        "conclusion"=>"The theorem excludes every affine forward or backward VBC with any finite component count and a constant nonnegative comparison matrix on this fixed domain.")
end

"""Author-constructed reverse separation example.

An affine implication-style forward IBC exists. A checked convex-combination
witness satisfies the hypotheses of the universal affine global-VBC obstruction.
A quartic scalar global-comparison certificate is also verified algebraically,
showing that degree escalation can recover that formulation (without claiming
that degree four is minimal).
"""
function exact_implication_gap()
    P=benchmark("ImplicationGap1D"); x=P.x[1]; f=P.f[1]
    domain=domain_audit(P); domain["status"]=="EXACT_INVARIANT" || error("domain")
    b=x; sep=rat(1//5)
    implication_bundle=exact_forward_implication_witness()
    implication_bundle["verification"]["verified"] || error("exact implication replay")
    # Robust separation for b=x.
    initial_margin=rat(2//5)
    unsafe_margin=rat(1//4)
    initial_margin>=sep || error("initial separation")
    unsafe_margin>=sep || error("unsafe separation")
    # Exact implication certificate on {1-x^2 >= 0, -x >= 0}:
    # -f = f^2 + (x/2)^2(1-x^2) + ((x+1)^2/2)(-x).
    g=1-x^2; antecedent=-x; target=-f
    s0=f^2; s1=(x/2)^2; tau=(x+1)^2/2
    identity_ok=iszero(target-(s0+s1*g+tau*antecedent))
    identity_ok || error("implication identity")
    obstruction=affine_vbc_obstruction(P,[[-1],[1]],rat.([3//4,1//4]))
    obstruction["verified_hypotheses"] || error("obstruction hypotheses")
    # A degree-four scalar global VBC with A=1. This is an upper bound on the
    # degree needed by that formulation, not a minimal-degree theorem.
    p=x^2*(x+1)^2-rat(3//40)
    t=f
    propagation_identity=iszero(p-compose(p,P.x,P.f)-t^2*(1-t)*(3+t))
    propagation_identity || error("quartic propagation identity")
    return Dict("schema"=>"dt-vbc-ibc-complementarity-v1","problem"=>P.name,"verified"=>true,
        "domain"=>domain,
        "implication_ibc"=>Dict("direction"=>"forward","frame"=>"x","frames_can_be_repeated"=>true,
            "degree"=>1,"separation_used"=>string(sep),"initial_margin"=>string(initial_margin),
            "unsafe_margin"=>string(unsafe_margin),"identity_verified"=>identity_ok,
            "exact_replay_status"=>implication_bundle["verification"]["status"],
            "proof_scope"=>implication_bundle["verification"]["scope"],
            "identity"=>"-f = f^2 + (x/2)^2*(1-x^2) + ((x+1)^2/2)*(-x)"),
        "affine_global_vbc_obstruction"=>obstruction,
        "global_vbc_recovery"=>Dict("degree"=>4,"comparison_gain"=>"1//1",
            "certificate"=>"x^2*(x+1)^2-3/40","initial_margin"=>"1//80",
            "unsafe_margin"=>"29//1280","propagation_identity_verified"=>propagation_identity,
            "propagation_identity"=>"p(x)-p(f(x))=f(x)^2*(1-f(x))*(3+f(x))",
            "degree_minimum_claim"=>"2 <= d_min <= 4; affine (degree 1) is excluded, but quadratic/cubic vector certificates are not excluded."))
end

"""Tight degree data for the finite-order rotations. Existing exact_rotation
supplies degree-one cyclic VBCs. The orbit-average theorem excludes every affine
implication IBC of finite length. The invariant quadratic below supplies an IBC,
so the minimum IBC degree on these examples is exactly two.
"""
function rotation_degree_report(name)
    name in ("Rotation2","Rotation4") || throw(ArgumentError("rotation benchmark"))
    P=benchmark(name); x=P.x
    q=x[1]^2+x[2]^2-rat(3//2)
    invariant=iszero(compose(q,P.x,P.f)-q)
    invariant || error("quadratic invariant")
    # First coordinate pair: max on X0 is (11/10)^2+(1/10)^2=61/50;
    # min on Xu is (3/2)^2=9/4.
    initial_margin=rat(3//2)-rat(61//50) # 7/25
    unsafe_margin=rat(9//4)-rat(3//2)    # 3/4
    initial_margin>0 && unsafe_margin>0 || error("rotation separation")
    n=length(x); R=zeros(QQ,n,n)
    for i in 1:n; R[i,isodd(i) ? i+1 : i-1]=isodd(i) ? -1 : 1; end
    orbit_sum=sum(R^j for j in 0:3)
    spectral_hypotheses=(R^4==Matrix{QQ}(I,n,n) && det(R-Matrix{QQ}(I,n,n))!=0 && orbit_sum==zeros(QQ,n,n))
    spectral_hypotheses || error("orbit obstruction hypotheses")
    affine=exact_rotation(name); affine["verification"]["verified"] || error("affine VBC witness")
    return Dict("problem"=>name,"verified"=>true,
        "vbc"=>Dict("minimum_degree"=>1,"witness_components"=>length(affine["B"]),"coupling"=>"cyclic"),
        "ibc"=>Dict("minimum_degree"=>2,"affine_excluded_by_orbit_average_theorem"=>true,
            "quadratic_witness"=>"x1^2+x2^2-3/2","quadratic_invariant"=>invariant,
            "initial_margin"=>string(initial_margin),"unsafe_margin"=>string(unsafe_margin)),
        "obstruction_hypotheses_verified"=>spectral_hypotheses)
end

"""Dense monomial/SOS sizing used only as transparent complexity accounting.
It is not a runtime prediction. The propagation degree is the generic upper
bound d*deg(f); actual cancellations/sparsity can make a problem smaller.
"""
function complexity_profile(n::Integer,d::Integer,m::Integer;dynamics_degree::Integer=2)
    n>0 && d>=0 && m>0 && dynamics_degree>0 || throw(ArgumentError("complexity dimensions"))
    coeff_per=binomial(n+d,d)
    propagation_degree=d*dynamics_degree
    gram_order=cld(propagation_degree,2)
    gram_dim=binomial(n+gram_order,gram_order)
    gram_entries=gram_dim*(gram_dim+1)÷2
    return Dict("states"=>n,"certificate_degree"=>d,"components"=>m,
        "dense_coefficients_per_component"=>coeff_per,"dense_certificate_coefficients"=>m*coeff_per,
        "dynamics_degree"=>dynamics_degree,"propagation_degree_upper_bound"=>propagation_degree,
        "degree_matched_dense_gram_dimension"=>gram_dim,"symmetric_gram_entries"=>gram_entries,
        "interpretation"=>"Combinatorial sizing only; multiplier degrees, sparsity, component count, and solver structure also determine cost.")
end

function complementarity_report()
    gap=exact_implication_gap()
    r2=rotation_degree_report("Rotation2")
    r4=rotation_degree_report("Rotation4")
    profiles=[complexity_profile(6,d,1;dynamics_degree=2) for d in (1,2,4)]
    return Dict("schema"=>"dt-vbc-ibc-complementarity-report-v1","verified"=>true,
        "title"=>"Duality and Complementarity of Vector and Interpolation-Inspired Barrier Certificates for Safety Verification: Toward Reduced Conservatism and Complexity",
        "structural_message"=>"Exact conversion within the globally scaled path subclass preserves degree. Complementary formulations outside that subclass can remove different structural obstructions before degree is increased.",
        "rotation_degree_separation"=>[r2,r4],"implication_gap"=>gap,
        "complexity_accounting"=>profiles,
        "limits"=>["No universal dominance claim.","The implication-gap affine VBC obstruction assumes a fixed domain and constant nonnegative comparison matrix.","Degree four is an upper bound, not a proved minimum, for the global VBC recovery in ImplicationGap1D.","Combinatorial sizing is not a runtime or scalability theorem."])
end
