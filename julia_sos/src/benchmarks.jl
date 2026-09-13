# Versioned, exact definitions. See docs/BENCHMARK_PROVENANCE.md.
const EXTRA_CASES = ("BB_rotation", "Logistic_adapted", "Rotation2", "Rotation4")
function benchmark(name::AbstractString)
    name in EXTRA_CASES || return legacy_benchmark(name)
    n = name == "Rotation4" ? 4 : 2
    @polyvar x[1:n]
    if name == "BB_rotation"
        # Exact published autonomous BarrierBench entry: no controller added.
        f = [-x[2]/100, x[1]/100]
        X=box([(-6//5,6//5),(-6//5,6//5)])
        X0=box([(1//10,2//5),(1//10,11//20)])
        Xu=box([(9//20,1//2),(3//5,1)])
    elseif name == "Logistic_adapted"
        # Remove inputs from the published controlled logistic entry.
        # This is NOT a reproduction of that control-synthesis task.
        f=[16//5*x[1]*(1-x[1]),14//5*x[2]*(1-x[2])]
        X=box([(0,1),(0,1)])
        X0=box([(1//10,3//10),(1//5,2//5)])
        Xu=box([(4//5,1),(4//5,1)])
    else
        # Author-constructed obstruction, not a literature benchmark.
        f=[isodd(i) ? -x[i+1] : x[i-1] for i in 1:n]
        X=box(fill((-2,2),n))
        X0=box([isodd(i) ? (9//10,11//10) : (-1//10,1//10) for i in 1:n])
        Xu=box([i==1 ? (3//2,17//10) : (-1//10,1//10) for i in 1:n])
    end
    return (;name=String(name),x,f=[qpoly(p,x) for p in f],X,X0,Xu)
end
function domain_audit(P)
    P.name in EXTRA_CASES || return legacy_domain_audit(P)
    R=benchmark(P.name)
    same=P.X==R.X && P.X0==R.X0 && P.Xu==R.Xu &&
         length(P.f)==length(R.f) && length(P.x)==length(R.x) &&
         all(iszero(P.f[i]-compose(R.f[i],R.x,P.x)) for i in eachindex(P.f))
    same || return Dict("status"=>"DOMAIN_NOT_ESTABLISHED","proof"=>"Canonical data changed.")
    if P.name=="BB_rotation"
        return Dict("status"=>"EXACT_INVARIANT","proof"=>"Absolute image bounds 3/250 in both coordinates are within +/-6/5.","image_abs_bounds"=>["3//250","3//250"],"initial_box_invariant"=>false)
    elseif P.name=="Logistic_adapted"
        @assert rat(16//5)/4<1 && rat(14//5)/4<1
        return Dict("status"=>"EXACT_INVARIANT","proof"=>"For x in [0,1], 0 <= r*x*(1-x) <= r/4.","image_bounds"=>[["0//1","4//5"],["0//1","7//10"]],"initial_box_invariant"=>false)
    else
        return Dict("status"=>"EXACT_INVARIANT","proof"=>"Signed coordinate permutation maps the box onto itself; f^4=identity.","initial_box_invariant"=>false,"period_divides"=>4)
    end
end
