function path_matrix(lambda; reciprocal=false)
    isempty(lambda) && throw(ArgumentError("empty frame sequence"))
    all(t -> t > 0, lambda) || throw(ArgumentError("scales must be positive"))
    m=length(lambda)
    A=zeros(QQ,m,m)
    for i in 1:m
        A[i,min(i+1,m)] = reciprocal ? inv(rat(lambda[i])) : rat(lambda[i])
    end
    return A
end
function checked_matrix(A,m)
    size(A)==(m,m) || throw(DimensionMismatch("comparison matrix"))
    all(a -> isfinite(a) && a>=0,A) || throw(ArgumentError("A must be nonnegative"))
    return A
end
"""All returned polynomial targets must be nonnegative on the ENTIRE named box.
This uses robust anchored separation and full-X propagation. These are explicit
strengthenings of the weak/disjunctive/unsafe-excluded manuscript definitions.
"""
function obligations(P, family::Symbol, B, parameter; separation=1//1000)
    x,f=P.x,P.f; m=length(B); sep=separation
    m>0 || throw(ArgumentError("empty certificate"))
    bf=[compose(p,x,f) for p in B]
    out=NamedTuple[]
    add(n,p,k,kind)=push!(out,(;name=n,p=p+0*x[1],box=k,kind))
    if family in (:forward_vbc,:backward_vbc)
        A=checked_matrix(parameter,m)
        if family==:forward_vbc
            for i in 1:m
                add("initial_$i",-B[i]-sep,P.X0,"separation")
                add("propagation_$i",sum(A[i,j]*B[j] for j in 1:m)-bf[i],P.X,"propagation")
            end
            add("unsafe_anchor",B[1]-sep,P.Xu,"separation")
        else
            for i in 1:m
                add("unsafe_$i",-B[i]-sep,P.Xu,"separation")
                add("propagation_$i",sum(A[i,j]*bf[j] for j in 1:m)-B[i],P.X,"propagation")
            end
            add("initial_anchor",B[1]-sep,P.X0,"separation")
        end
    elseif family in (:forward_ibc,:backward_ibc)
        l=parameter
        length(l)==m && all(t->t>0,l) || throw(ArgumentError("invalid frame scales"))
        if family==:forward_ibc
            add("initial_anchor",-B[1]-sep,P.X0,"separation")
            for i in 1:m
                add("unsafe_$i",B[i]-sep,P.Xu,"separation")
                add("propagation_$i",B[i]-l[i]*bf[min(i+1,m)],P.X,"propagation")
            end
        else
            add("unsafe_anchor",-B[1]-sep,P.Xu,"separation")
            for i in 1:m
                add("initial_$i",B[i]-sep,P.X0,"separation")
                add("propagation_$i",l[i]*bf[i]-B[min(i+1,m)],P.X,"propagation")
            end
        end
    else
        throw(ArgumentError("unknown formulation: $family"))
    end
    return out
end
