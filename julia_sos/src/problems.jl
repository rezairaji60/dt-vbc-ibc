rat(x::Rational) = BigInt(numerator(x)) // BigInt(denominator(x))
rat(x::Integer) = BigInt(x) // BigInt(1)
function rat(x::AbstractFloat)
    isfinite(x) || throw(ArgumentError("nonfinite coefficient"))
    return rationalize(BigInt, x; tol=1e-12)
end
function rat(x::AbstractString)
    v = split(x, "//")
    length(v) == 1 && return parse(BigInt, v[1]) // BigInt(1)
    length(v) == 2 || throw(ArgumentError("invalid rational"))
    return parse(BigInt, v[1]) // parse(BigInt, v[2])
end
box(v) = [(rat(a), rat(b)) for (a,b) in v]
function benchmark(name::AbstractString)
    @polyvar x[1:2]
    if name == "S1"
        f = [18//25*x[1] + 1//10*x[2] - 3//25*x[1]^3,
             -2//25*x[1] + 17//25*x[2] - 2//25*x[2]^3]
        X = box([(-7//5,7//5),(-7//5,7//5)])
        X0 = box([(-1//5,1//5),(-1//5,1//5)])
        Xu = box([(19//20,5//4),(19//20,5//4)])
    elseif name in ("S2_original", "S2_repaired")
        f = [21//20*x[1] + 9//50*x[2] - 1//10*x[1]^3,
             1//50*x[1] + 23//25*x[2]]
        a = name == "S2_original" ? 7//5 : 3//2
        X = box([(-a,a),(-6//5,6//5)])
        X0 = box([(-3//25,3//25),(-3//25,3//25)])
        Xu = box([(17//20,11//10),(-1//10,1//10)])
    else
        throw(ArgumentError("unknown benchmark: $name"))
    end
    return (; name=String(name), x, f=[qpoly(p,x) for p in f], X, X0, Xu)
end
compose(p,x,f) = MP.subs(p, x => f)
function qpoly(p,x)
    pp = p + 0*x[1]
    return sum((rat(MP.coefficient(t))*MP.monomial(t) for t in MP.terms(pp)); init=rat(0)*x[1])
end
exponents(m,x) = [MP.degree(m,xi) for xi in x]
function polydata(p,x)
    pp = p + 0*x[1]
    return [Dict("c"=>string(MP.coefficient(t)), "e"=>exponents(MP.monomial(t),x)) for t in MP.terms(pp)]
end
function frompolydata(data,x)
    p = rat(0)*x[1]
    for t in data
        e = Int.(t["e"])
        length(e) == length(x) || throw(ArgumentError("invalid exponent dimension"))
        all(e .>= 0) || throw(ArgumentError("negative exponent"))
        p += rat(String(t["c"]))*prod(x[i]^e[i] for i in eachindex(x))
    end
    return p
end
generators(x,K) = [(x[i]-K[i][1])*(K[i][2]-x[i]) for i in eachindex(x)]
function domain_audit(P)
    if P.name == "S1"
        # a(x)=.72-.12x^2 and d(y)=.68-.08y^2 are positive on X.
        amin = rat(18//25)-rat(3//25)*rat(7//5)^2
        dmin = rat(17//25)-rat(2//25)*rat(7//5)^2
        @assert amin > 0 && dmin > 0
        bounds = [rat(41//50)*rat(7//5), rat(19//25)*rat(7//5)]
        @assert all(bounds .< rat(7//5))
        return Dict("status"=>"EXACT_INVARIANT", "image_abs_bounds"=>string.(bounds),
                    "initial_box_invariant"=>true, "proof"=>"Nonnegative diagonal factors and absolute row-sum bounds.")
    elseif P.name == "S2_repaired"
        # df1/dx1 = 1.05-.3x1^2 > 0, df1/dx2=.18; f is odd.
        a,b = rat(3//2),rat(6//5)
        @assert rat(21//20)-rat(3//10)*a^2 > 0
        v = [rat(21//20)*a+rat(9//50)*b-rat(1//10)*a^3,
             rat(1//50)*a+rat(23//25)*b]
        @assert v[1] < a && v[2] < b
        return Dict("status"=>"EXACT_INVARIANT", "image_abs_bounds"=>string.(v),
                    "initial_box_invariant"=>false, "proof"=>"Coordinatewise monotonicity and odd symmetry; exact corner maxima.")
    else
        a,b = rat(7//5),rat(6//5)
        image = [rat(21//20)*a+rat(9//50)*b-rat(1//10)*a^3,
                 rat(1//50)*a+rat(23//25)*b]
        @assert image[1] > a
        return Dict("status"=>"NOT_INVARIANT", "point"=>string.([a,b]), "image"=>string.(image),
                    "initial_box_invariant"=>false, "proof"=>"Exact counterexample to f(X) subset X.")
    end
end
function write_json(path, data)
    mkpath(dirname(path))
    open(path,"w") do io
        JSON3.write(io,data)
        println(io)
    end
    return path
end
