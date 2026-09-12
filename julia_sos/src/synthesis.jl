"""Explicit Putinar SOS identity: p = reserve + sum_j g_j*z_j'Q_j*z_j.
PSD decision matrices and coefficient identities replace ALL sampled inequalities.
order is a relaxation order: each product has total degree at most 2*order.
"""
function putinar!(model,p,x,K;order=3,reserve=1//1000000)
    2*order>=MP.maxdegree(p) || throw(ArgumentError("relaxation order too small"))
    gs=generators(x,K); blocks=Any[]; rhs=0*x[1]
    for idx in 0:length(gs)
        d=idx==0 ? order : order-1
        d>=0 || throw(ArgumentError("invalid multiplier degree"))
        z=MP.monomials(x,0:d); n=length(z)
        Q=@variable(model,[1:n,1:n],PSD)
        s=sum(Q[i,j]*z[i]*z[j] for i in 1:n,j in 1:n)
        rhs+=(idx==0 ? 1 : gs[idx])*s
        push!(blocks,(;Q,z,index=idx))
    end
    con=@constraint(model,p==Float64(reserve)+rhs)
    return (;blocks,con,reserve=rat(reserve))
end
function normalized_template!(model,x,m,d,normalization)
    z=MP.monomials(x,0:d); n=length(z)
    C=@variable(model,[1:m,1:n])
    if normalization==:symmetric_l1
        T=@variable(model,[1:m,1:n],lower_bound=0)
        for i in 1:m,j in 1:n
            @constraint(model,C[i,j]<=T[i,j])
            @constraint(model,-C[i,j]<=T[i,j])
        end
        for i in 1:m; @constraint(model,sum(T[i,:])<=1); end
    elseif normalization==:legacy_positive_trace
        d==2 && length(x)==2 || throw(ArgumentError("legacy mode is quadratic 2D only"))
        j1=findfirst(==(x[1]^2),z); j2=findfirst(==(x[2]^2),z)
        j0=findfirst(mm->all(MP.degree(mm,xi)==0 for xi in x),z)
        for i in 1:m
            @constraint(model,C[i,j1]+C[i,j2]==1)
            @constraint(model,C[i,j1]>=0.05)
            @constraint(model,C[i,j2]>=0.05)
            @constraint(model,-2<=C[i,j0]<=0.5)
        end
    else
        throw(ArgumentError("unknown normalization"))
    end
    B=[sum(C[i,j]*z[j] for j in 1:n) for i in 1:m]
    return B,C,z
end
function synthesize(P,family::Symbol,parameter;degree=2,order=3,separation=1//1000,
                    reserve=1//1000000,normalization=:symmetric_l1,fixed_B=nothing,
                    optimizer=CSDP.Optimizer)
    m=family in (:forward_vbc,:backward_vbc) ? size(parameter,1) : length(parameter)
    model=SOSModel(optimizer);set_silent(model)
    C=nothing;z=nothing
    if fixed_B===nothing
        B,C,z=normalized_template!(model,P.x,m,degree,normalization)
    else
        length(fixed_B)==m || throw(DimensionMismatch("fixed certificate"))
        B=fixed_B
    end
    obs=obligations(P,family,B,parameter;separation=separation)
    proofs=[putinar!(model,o.p,P.x,o.box;order=order,reserve=reserve) for o in obs]
    @objective(model,Min,0)
    elapsed=@elapsed optimize!(model)
    meta=Dict{String,Any}("problem"=>P.name,"family"=>String(family),"degree"=>degree,
        "relaxation_order"=>order,"components"=>m,"normalization"=>String(normalization),
        "sos_reserve"=>string(rat(reserve)),"separation"=>string(rat(separation)),
        "termination_status"=>string(termination_status(model)),
        "primal_status"=>string(primal_status(model)),"runtime_seconds"=>elapsed,
        "scalar_variables"=>num_variables(model),"domain"=>domain_audit(P))
    if !has_values(model)
        meta["status"]="NO_CERTIFIED_CANDIDATE"
        meta["interpretation"]="Solver outcome for this fixed relaxation only; not certificate nonexistence."
        return meta,nothing
    end
    Bq=fixed_B===nothing ? [sum(rat(value(C[i,j]))*z[j] for j in eachindex(z)) for i in 1:m] : [qpoly(p,P.x) for p in B]
    bundle=bundle_base(P,family,Bq,parameter,separation)
    bundle["origin"]="numerical_SOS_candidate_with_exact_replay"
    shifts=QQ[]
    try
        for (o,pf) in zip(obs,proofs)
            grams=Any[]
            for block in pf.blocks
                Q,shift=certify_gram(value.(block.Q));push!(shifts,shift)
                push!(grams,gramdata(Q,block.z,block.index,P.x))
            end
            push!(bundle["proofs"],Dict("name"=>o.name,"reserve"=>string(pf.reserve),"grams"=>grams))
        end
        result=verify_bundle(bundle)
        bundle["verification"]=result
        meta["status"]=result["status"]
        meta["exact_verified"]=result["verified"]
        meta["max_gram_shift"]=string(maximum(shifts;init=rat(0)))
        meta["max_residual_bound"]=string(maximum(rat(String(c["absolute_error_bound"])) for c in result["checks"]))
        meta["all_coefficients_finite"]=true
    catch err
        meta["status"]="UNVERIFIED_CANDIDATE"
        meta["validation_error"]=sprint(showerror,err)
    end
    bundle["solver_metadata"]=meta
    return meta,bundle
end
