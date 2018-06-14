import Base.copy

export GLPKOptimizerLP, GLPKOptimizerMIP

using LinQuadOptInterface

const LQOI = LinQuadOptInterface
const MOI  = LQOI.MOI

# Many functions in this module are adapted from GLPKMathProgInterface.jl. This is the copyright notice:
## Copyright (c) 2013: Carlo Baldassi
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.

const SUPPORTED_OBJECTIVES = [
    LQOI.SinVar,
    LQOI.Linear
]
const SUPPORTED_CONSTRAINTS_LP = [
    (LQOI.Linear, LQOI.EQ),
    (LQOI.Linear, LQOI.LE),
    (LQOI.Linear, LQOI.GE),
    (LQOI.Linear, LQOI.IV),
    (LQOI.SinVar, LQOI.EQ),
    (LQOI.SinVar, LQOI.LE),
    (LQOI.SinVar, LQOI.GE),
    (LQOI.SinVar, LQOI.IV),
    (LQOI.VecVar, MOI.Nonnegatives),
    (LQOI.VecVar, MOI.Nonpositives),
    (LQOI.VecVar, MOI.Zeros),
    (LQOI.VecLin, MOI.Nonnegatives),
    (LQOI.VecLin, MOI.Nonpositives),
    (LQOI.VecLin, MOI.Zeros)
]
const SUPPORTED_CONSTRAINTS_MIP = vcat(SUPPORTED_CONSTRAINTS_LP, [
    (LQOI.SinVar, MOI.ZeroOne),
    (LQOI.SinVar, MOI.Integer),
    # (VecVar, LQOI.SOS1),
    # (VecVar, LQOI.SOS2),
])

abstract type GLPKOptimizer <: LQOI.LinQuadOptimizer end

import GLPK.Prob

const Model = GLPK.Prob
Model(env) = Model()

LQOI.LinearQuadraticModel(::Type{M},env) where M<:GLPKOptimizer = GLPK.Prob()

mutable struct GLPKOptimizerLP <: GLPKOptimizer
    LQOI.@LinQuadOptimizerBase
    method::Symbol
    param::Union{GLPK.SimplexParam, GLPK.InteriorParam}
    GLPKOptimizerLP(::Void) = new()
end
function GLPKOptimizerLP(presolve = false, method = :Simplex;kwargs...)
    if !(method in [:Simplex,:Exact,:InteriorPoint])
        error("""
        Unknown method for GLPK LP solver: $method
            Allowed methods:
                :Simplex
                :Exact
                :InteriorPoint""")
    end
    if method == :Simplex || method == :Exact
        param = GLPK.SimplexParam()
        if presolve
            param.presolve = GLPK.ON
        end
    elseif method == :InteriorPoint
        param = GLPK.InteriorParam()
        if presolve
            warn("Ignored option: presolve")
        end
    else
        error("This is a bug")
    end
    param.msg_lev = GLPK.MSG_ERR
    for (k,v) in kwargs
        i = findfirst(x->x==k, fieldnames(typeof(param)))
        if i > 0
            t = typeof(param).types[i]
            setfield!(param, i, convert(t, v))
        else
            warn("Ignored option: $(string(k))")
        end
    end

    m = GLPKOptimizerLP(nothing)
    m.param = param
    m.method = method
    MOI.empty!(m)

    return m
end

mutable struct GLPKOptimizerMIP <: GLPKOptimizer

    LQOI.@LinQuadOptimizerBase

    param::GLPK.IntoptParam
    smplxparam::GLPK.SimplexParam
    # lazycb::Union{Function,Void}
    # cutcb::Union{Function,Void}
    # heuristiccb::Union{Function,Void}
    # infocb::Union{Function,Void}
    objbound::Float64
    # cbdata::MathProgCallbackData
    binaries::Vector{Int}
    userlimit::Bool

    GLPKOptimizerMIP(::Void) = new()
end


function GLPKOptimizerMIP(presolve = false; kwargs...)

    env = nothing
    lpm = GLPKOptimizerMIP(nothing)
    lpm.param = GLPK.IntoptParam()
    lpm.smplxparam = GLPK.SimplexParam()
    lpm.objbound = -Inf
    lpm.binaries = Int[]
    lpm.userlimit = false
    MOI.empty!(lpm)

    lpm.param.msg_lev = GLPK.MSG_ERR
    lpm.smplxparam.msg_lev = GLPK.MSG_ERR
    if presolve
        lpm.param.presolve = GLPK.ON
    end

    # lpm.param.cb_func = cfunction(_internal_callback, Void, (Ptr{Void}, Ptr{Void}))
    # lpm.param.cb_info = pointer_from_objref(lpm.cbdata)

    for (k,v) in kwargs
        if k in [:cb_func, :cb_info]
            warn("ignored option: $(string(k)); use the MathProgBase callback interface instead")
            continue
        end
        i = findfirst(x->x==k, fieldnames(typeof(lpm.param)))
        s = findfirst(x->x==k, fieldnames(typeof(lpm.smplxparam)))
        if !(i > 0 || s > 0)
            warn("Ignored option: $(string(k))")
            continue
        end
        if i > 0
            t = typeof(lpm.param).types[i]
            setfield!(lpm.param, i, convert(t, v))
        end
        if s > 0
            t = typeof(lpm.smplxparam).types[s]
            setfield!(lpm.smplxparam, s, convert(t, v))
        end
    end

    return lpm
end

function MOI.empty!(m::GLPKOptimizer)
    MOI.empty!(m,nothing)
end

LQOI.supported_objectives(s::GLPKOptimizer) = SUPPORTED_OBJECTIVES
LQOI.supported_constraints(s::GLPKOptimizerMIP) = SUPPORTED_CONSTRAINTS_MIP
LQOI.supported_constraints(s::GLPKOptimizerLP) = SUPPORTED_CONSTRAINTS_LP
#=
    inner wrapper
=#

#=
    Main
=#

function LQOI.change_variable_bounds!(instance::GLPKOptimizer, colvec, valvec, sensevec)
    m = instance.inner
    for i in eachindex(colvec)
        colub = Inf
        collb = -Inf
        bt = GLPK.DB
        if sensevec[i] == Cchar('E')
            colub = valvec[i]
            collb = valvec[i]
            bt = GLPK.FX
        elseif sensevec[i] == Cchar('L')
            collb = valvec[i]
            colub = Inf
            u = GLPK.get_col_ub(m, colvec[i])
            if u < 1e100
                bt = GLPK.DB
                colub = u
            else
                bt = GLPK.LO
            end
        elseif sensevec[i] == Cchar('U')
            colub = valvec[i]
            collb = -Inf
            l = GLPK.get_col_lb(m, colvec[i])
            if l > -1e100
                bt = GLPK.DB
                collb = l
            else
                bt = GLPK.UP
            end
        else
            error("invalid bound type")
        end
        if colub > 1e100 && collb < -1e100
            bt = GLPK.FR
            colub = Inf
            collb = -Inf
        elseif colub == collb
            bt = GLPK.FX
        end
        GLPK.set_col_bnds(m, colvec[i], bt, collb, colub)
    end

end


function LQOI.get_variable_lowerbound(instance::GLPKOptimizer, col)
    GLPK.get_col_lb(instance.inner, col)
end

function LQOI.get_variable_upperbound(instance::GLPKOptimizer, col)
    GLPK.get_col_ub(instance.inner, col)
end

function LQOI.get_number_linear_constraints(instance::GLPKOptimizer)
    GLPK.get_num_rows(instance.inner)
end

function LQOI.add_linear_constraints!(instance::GLPKOptimizer,
        A::LQOI.CSRMatrix{Float64}, senses::Vector{Cchar}, rhs::Vector{Float64})
    m = instance.inner
    nrows = length(rhs)
    if nrows <= 0
        error("no row to be added")
    elseif nrows == 1
        addrow!(m, A.columns, A.coefficients, senses[1], rhs[1])
    else
        push!(A.row_pointers, length(A.columns)+1)
        for i in 1:nrows
            indices = A.row_pointers[i]:A.row_pointers[i+1]-1
            addrow!(m, A.columns[indices], A.coefficients[indices], senses[i], rhs[i])
        end
        pop!(A.row_pointers)
    end
end

function LQOI.add_ranged_constraints!(instance::GLPKOptimizer, A::LQOI.CSRMatrix{Float64}, lowerbound::Vector{Float64}, upperbound::Vector{Float64})
    row1 = GLPK.get_num_rows(instance.inner)
    LQOI.add_linear_constraints!(instance, A,
        fill(Cchar('R'), length(lowerbound)), lowerbound)
    row2 = GLPK.get_num_rows(instance.inner)
    for (i,row) in enumerate((row1+1):row2)
        GLPK.set_row_bnds(instance.inner, row, GLPK.DB, lowerbound[i], upperbound[i])
    end
end

function LQOI.modify_ranged_constraints!(instance::GLPKOptimizer, rows, lowerbounds, upperbounds)
    for (row, lb, ub) in zip(rows, lowerbounds, upperbounds)
        LQOI.change_rhs_coefficient!(instance, row, lb)
        GLPK.set_row_bnds(instance.inner, row, GLPK.DB, lb, ub)
    end
end

function LQOI.get_range(m::GLPKOptimizer, row::Int)
    GLPK.get_row_lb(m.inner, row), GLPK.get_row_ub(m.inner, row)
end

function addrow!(lp::GLPK.Prob, colidx::Vector, colcoef::Vector, sense::Cchar, rhs::Real)
    if length(colidx) != length(colcoef)
        error("colidx and colcoef have different legths")
    end
    GLPK.add_rows(lp, 1)
    m = GLPK.get_num_rows(lp)
    GLPK.set_mat_row(lp, m, colidx, colcoef)

    if sense == Cchar('E')
        bt = GLPK.FX
        rowlb = rhs
        rowub = rhs
    elseif sense == Cchar('G')
        bt = GLPK.LO
        rowlb = rhs
        rowub = Inf
    elseif sense == Cchar('L')
        bt = GLPK.UP
        rowlb = -Inf
        rowub = rhs
    elseif sense == Cchar('R')
        # start with lower
        bt = GLPK.DB
        rowlb = rhs
        rowub = Inf
    else
        error("row type $(sense) not valid")
        bt = GLPK.FR
    end
    GLPK.set_row_bnds(lp, m, bt, rowlb, rowub)
    return
end

function LQOI.get_rhs(instance::GLPKOptimizer, row)
    m = instance.inner
    sense = GLPK.get_row_type(m, row)
    if sense == GLPK.LO
        return GLPK.get_row_lb(m, row)
    elseif sense == GLPK.FX
        return GLPK.get_row_lb(m, row)
    elseif sense == GLPK.DB
        return GLPK.get_row_lb(m, row)
    else
        return GLPK.get_row_ub(m, row)
    end
end

function LQOI.get_linear_constraint(instance::GLPKOptimizer, idx)
    lp = instance.inner
    colidx, coefs = GLPK.get_mat_row(lp, idx)
    # note: we return 1-indexed columns here
    return colidx, coefs
end

function LQOI.change_rhs_coefficient!(instance::GLPKOptimizer, idx::Integer, rhs::Real)
    lp = instance.inner

    l = GLPK.get_row_lb(lp, idx)
    u = GLPK.get_row_ub(lp, idx)

    rowub = Inf
    rowlb = -Inf

    if l == u
        bt = GLPK.FX
        rowlb = rhs
        rowub = rhs
    elseif l > -Inf && u < Inf
        bt = GLPK.FX
        rowlb = rhs
        rowub = rhs
    elseif l > -Inf
        bt = GLPK.LO
        rowlb = rhs
        rowub = Inf
    elseif u < Inf
        bt = GLPK.UP
        rowlb = -Inf
        rowub = rhs
    else
        error("not valid rhs")
    end

    GLPK.set_row_bnds(lp, idx, bt, rowlb, rowub)
end

function LQOI.change_objective_coefficient!(instance::GLPKOptimizer, col, coef)
    lp = instance.inner
    GLPK.set_obj_coef(lp, col, coef)
end

function LQOI.change_matrix_coefficient!(instance::GLPKOptimizer, row, col, coef)
    lp = instance.inner
    colidx, coefs = GLPK.get_mat_row(lp, row)
    idx = findfirst(colidx, col)
    if idx > 0
        coefs[idx] = coef
    else
        push!(colidx, col)
        push!(coefs, coef)
    end
    GLPK.set_mat_row(lp, row, colidx, coefs)
    return nothing
end

function LQOI.delete_linear_constraints!(instance::GLPKOptimizer, rowbeg, rowend)

    m = instance.inner

    idx = collect(rowbeg:rowend)

    GLPK.std_basis(m)
    GLPK.del_rows(m, length(idx), idx)

    nothing
end

# TODO fix types
function LQOI.change_variable_types!(instance::GLPKOptimizer, colvec, vartype)

    lp = instance.inner
    coltype = GLPK.CV
    for i in eachindex(colvec)
        if vartype[i] == Cint('I')
            coltype = GLPK.IV
        elseif vartype[i] == Cint('C')
            coltype = GLPK.CV
        elseif vartype[i] == Cint('B')
            coltype = GLPK.IV
            GLPK.set_col_bnds(lp, colvec[i], GLPK.DB, 0.0, 1.0)
        else
            error("invalid variable type: $(vartype[i])")
        end
        GLPK.set_col_kind(lp, colvec[i], coltype)
    end
end

# TODO fix types
function LQOI.change_linear_constraint_sense!(instance::GLPKOptimizer, rowvec, sensevec)
    for (row, sense) in zip(rowvec, sensevec)
        changesense!(instance, row, sense)
    end
    nothing
end

function changesense!(instance::GLPKOptimizer, row, sense)
    m = instance.inner
    oldsense = GLPK.get_row_type(m, row)
    newsense = translatesense(sense)
    if oldsense == newsense
        return nothing
    end

    if newsense == GLPK.FX
        rowub = rowlb
    elseif oldsense == GLPK.DB
        if newsense == GLPK.UP
            rowlb = -Inf
        else
            rowub = Inf
        end
    else
        rowlb = get_row_lb(m, row)
        rowub = get_row_ub(m, row)
        if newsense == GLPK.UP
            rowub = rowlb
            rowlb = -Inf
        else
            rowlb = rowub
            rowub = Inf
        end
    end

    GLPK.set_row_bnds(m, row, newsense, rowlb, rowub)

    nothing
end

function translatesense(sense)
    if sense == Cchar('E')
        return GLPK.FX
    elseif sense == Cchar('R')
        return GLPK.DB
    elseif sense == Cchar('L')
        return GLPK.UP
    elseif sense == Cchar('G')
        return GLPK.LO
    else
        error("invalid sense")
    end
end

LQOI.add_sos_constraint!(instance::GLPKOptimizer, colvec, valvec, typ) = GLPK.add_sos!(instance.inner, typ, colvec, valvec)

LQOI.delete_sos!(instance::GLPKOptimizer, idx1, idx2) = error("cant del SOS")

# TODO improve getting processes
function LQOI.get_sos_constraint(instance::GLPKOptimizer, idx)
    indices, weights, types = GLPK.getsos(instance.inner, idx)

    return indices, weights, types == Cchar('1') ? :SOS1 : :SOS2
end



# LQOI.lqs_copyquad(m, intvec,intvec, floatvec) #?
# LQOI.lqs_copyquad!(instance::GLPKOptimizer, I, J, V) = error("GLPK does no support quadratics")

function LQOI.set_linear_objective!(instance::GLPKOptimizer, colvec, coefvec)
    ncols = GLPK.get_num_cols(instance.inner)
    new_colvec = collect(1:ncols)
    new_coefvec = zeros(ncols)
    for (ind,val) in enumerate(colvec)
        new_coefvec[val] = coefvec[ind]
    end
    m = instance.inner
    for (col, coef) in zip(new_colvec, new_coefvec)
        GLPK.set_obj_coef(m, col, coef)
    end
    nothing
end

function LQOI.change_objective_sense!(instance::GLPKOptimizer, sense)
    m = instance.inner
    if sense == :min
        GLPK.set_obj_dir(m, GLPK.MIN)
    elseif sense == :max
        GLPK.set_obj_dir(m, GLPK.MAX)
    else
        error("Unrecognized objective sense $sense")
    end
end

function LQOI.get_linear_objective!(instance::GLPKOptimizer, x::Vector{Float64})
    m = instance.inner
    n = GLPK.get_num_cols(m)
    for col = 1:length(x)
        x[col] = GLPK.get_obj_coef(m, col)
    end
end

function LQOI.get_objectivesense(instance::GLPKOptimizer)

    s = GLPK.get_obj_dir(instance.inner)
    if s == GLPK.MIN
        return MOI.MinSense
    elseif s == GLPK.MAX
        return MOI.MaxSense
    else
        error("Internal library error")
    end
end

LQOI.get_number_variables(instance::GLPKOptimizer) = GLPK.get_num_cols(instance.inner)

function LQOI.add_variables!(instance::GLPKOptimizer, int)
    n = GLPK.get_num_cols(instance.inner)
    GLPK.add_cols(instance.inner, int)
    for i in 1:int
        GLPK.set_col_bnds(instance.inner, n+i, GLPK.FR, -Inf, Inf)
    end
    nothing
end

function LQOI.delete_variables!(instance::GLPKOptimizer, col, col2)
    idx = collect(col:col2)
    GLPK.std_basis(instance.inner)
    GLPK.del_cols(instance.inner, length(idx), idx)
end

LQOI.add_mip_starts!(instance::GLPKOptimizer, colvec, valvec) = nothing

LQOI.solve_mip_problem!(instance::GLPKOptimizer) = opt!(instance)

# LQOI.solve_quadratic_problem!(instance::GLPKOptimizer) = error("Quadratic solving not supported")

LQOI.solve_linear_problem!(instance::GLPKOptimizer) = opt!(instance)

function LQOI.get_termination_status(model::GLPKOptimizerMIP)

    if model.userlimit
        return MOI.OtherLimit
    end
    s = GLPK.mip_status(model.inner)
    if s == GLPK.UNDEF
        if model.param.presolve == GLPK.OFF && GLPK.get_status(model.inner) == GLPK.NOFEAS
            return MOI.InfeasibleNoResult
        else
            return MOI.OtherError
        end
    end
    if s == GLPK.OPT
        return MOI.Success
    elseif s == GLPK.INFEAS
        return MOI.InfeasibleNoResult
    elseif s == GLPK.UNBND
        return MOI.UnboundedNoResult
    elseif s == GLPK.FEAS
        return MOI.SlowProgress
    elseif s == GLPK.NOFEAS
        return MOI.OtherError
    elseif s == GLPK.UNDEF
        return MOI.OtherError
    else
        error("internal library error")
    end
end

function LQOI.get_termination_status(model::GLPKOptimizerLP)
    s = lp_status(model)
    if s == GLPK.OPT
        return MOI.Success
    elseif s == GLPK.INFEAS
        return MOI.InfeasibleNoResult
    elseif s == GLPK.UNBND
        return MOI.UnboundedNoResult
    elseif s == GLPK.FEAS
        return MOI.SlowProgress
    elseif s == GLPK.NOFEAS
        return MOI.InfeasibleOrUnbounded
    elseif s == GLPK.UNDEF
        return MOI.OtherError
    else
        error("Internal library error")
    end
end

function lp_status(lpm::GLPKOptimizerLP)
    if lpm.method == :Simplex || lpm.method == :Exact
        get_status = GLPK.get_status
    elseif lpm.method == :InteriorPoint
        get_status = GLPK.ipt_status
    else
        error("bug")
    end
    s = get_status(lpm.inner)
end

function LQOI.get_primal_status(model::GLPKOptimizerMIP)
    m = model.inner
    s = GLPK.mip_status(model.inner)
    out = MOI.UnknownResultStatus
    if s in [GLPK.OPT]#, GLPK.FEAS]
        out = MOI.FeasiblePoint
    end
    return out
end

function LQOI.get_primal_status(model::GLPKOptimizerLP)
    m = model.inner
    s = lp_status(model)
    out = MOI.UnknownResultStatus
    if s in [GLPK.OPT]#, GLPK.FEAS]
        out = MOI.FeasiblePoint
    end
    return out
end

function LQOI.get_dual_status(model::GLPKOptimizerMIP)
    return MOI.UnknownResultStatus
end

function LQOI.get_dual_status(model::GLPKOptimizerLP)
    m = model.inner
    s = lp_status(model)
    out = MOI.UnknownResultStatus
    if s in [GLPK.OPT]#, GLPK.FEAS]
        out = MOI.FeasiblePoint
    end
    return out
end

function LQOI.get_variable_primal_solution!(instance::GLPKOptimizerMIP, place)
    lp = instance.inner
    for c in eachindex(place)
        place[c] = GLPK.mip_col_val(lp, c)
    end
end

function LQOI.get_variable_primal_solution!(lpm::GLPKOptimizerLP, place)
    lp = lpm.inner
    if lpm.method == :Simplex || lpm.method == :Exact
        get_col_prim = GLPK.get_col_prim
    elseif lpm.method == :InteriorPoint
        get_col_prim = GLPK.ipt_col_prim
    else
        error("bug")
    end

    for c in eachindex(place)
        place[c] = get_col_prim(lp, c)
    end
    return nothing
end

function LQOI.get_linear_primal_solution!(instance::GLPKOptimizerMIP, place)
    lp = instance.inner
    for c in eachindex(place)
        place[c] = GLPK.mip_row_val(lp, c)
    end
end
function LQOI.get_linear_primal_solution!(lpm::GLPKOptimizerLP, place)
    lp = lpm.inner
    if lpm.method == :Simplex || lpm.method == :Exact
        get_row_prim = GLPK.get_row_prim
    elseif lpm.method == :InteriorPoint
        get_row_prim = GLPK.ipt_row_prim
    else
        error("bug")
    end
    for r in eachindex(place)
        place[r] = get_row_prim(lp, r)
    end
    return nothing
end

# function LQOI.get_variable_dual_solution!(instance::GLPKOptimizerMIP, place)
# end

function LQOI.get_variable_dual_solution!(lpm::GLPKOptimizerLP, place)
    lp = lpm.inner
    if lpm.method == :Simplex || lpm.method == :Exact
        get_col_dual = GLPK.get_col_dual
    elseif lpm.method == :InteriorPoint
        get_col_dual = GLPK.ipt_col_dual
    else
        error("bug")
    end
    for c in eachindex(place)
        place[c] = get_col_dual(lp, c)
    end
    return nothing
end

# function LQOI.get_linear_dual_solution!(instance::GLPKOptimizerMIP, place)
 # end

function LQOI.get_linear_dual_solution!(lpm::GLPKOptimizerLP, place)
    lp = lpm.inner
    if lpm.method == :Simplex || lpm.method == :Exact
        get_row_dual = GLPK.get_row_dual
    elseif lpm.method == :InteriorPoint
        get_row_dual = GLPK.ipt_row_dual
    else
        error("bug")
    end
    for r in eachindex(place)
        place[r] = get_row_dual(lp, r)
    end
    return nothing
end

LQOI.get_objective_value(instance::GLPKOptimizerMIP) = GLPK.mip_obj_val(instance.inner)

function LQOI.get_objective_value(lpm::GLPKOptimizerLP)
    if lpm.method == :Simplex || lpm.method == :Exact
        get_obj_val = GLPK.get_obj_val
    elseif lpm.method == :InteriorPoint
        get_obj_val = GLPK.ipt_obj_val
    else
        error("bug")
    end
    return get_obj_val(lpm.inner)
end

LQOI.get_objective_bound(instance::GLPKOptimizerMIP) = instance.objbound

LQOI.get_relative_mip_gap(instance::GLPKOptimizer) = abs(GLPK.mip_obj_val(instance.inner)-instance.objbound)/(1e-9+GLPK.mip_obj_val(instance.inner))

# LQOI.get_iteration_count(m)
# LQOI.get_iteration_count(instance::GLPKOptimizer)  = -1

# LQOI.lqs_getbaritcnt(m)
# LQOI.lqs_getbaritcnt(instance::GLPKOptimizer) = -1

# LQOI.get_node_count(m)
# LQOI.get_node_count(instance::GLPKOptimizer) = -1

LQOI.get_farkas_dual!(instance::GLPKOptimizer, place) = getinfeasibilityray(instance, place)

LQOI.get_unbounded_ray!(instance::GLPKOptimizer, place) = getunboundedray(instance, place)

function MOI.free!(instance::GLPKOptimizer) end

"""
    writeproblem(m: :MOI.AbstractOptimizer, filename::String)
Writes the current problem data to the given file.
Supported file types are solver-dependent.
"""
writeproblem(instance::GLPKOptimizer, filename::String, flags::String="") = GLPK.write_model(instance.inner, filename)


LQOI.make_problem_type_continuous(instance::GLPKOptimizer) = GLPK._make_problem_type_continuous(instance.inner)


#=
    old helpers
=#
function opt!(lpm::GLPKOptimizerLP)
    # write_lp(lpm.inner, "model.lp")
    if lpm.method == :Simplex
        solve = GLPK.simplex
    elseif lpm.method == :Exact
        solve = GLPK.exact
    elseif lpm.method == :InteriorPoint
        solve = GLPK.interior
    else
        error("bug")
    end
    return solve(lpm.inner, lpm.param)
end

function opt!(lpm::GLPKOptimizerMIP)
    # write_lp(lpm.inner, "model.lp")
    vartype = getvartype(lpm)
    lb = getvarLB(lpm)
    ub = getvarUB(lpm)
    old_lb = copy(lb)
    old_ub = copy(ub)
    for c in 1:length(vartype)
        vartype[c] in [:Int,:Bin] && (lb[c] = ceil(lb[c]); ub[c] = floor(ub[c]))
        vartype[c] == :Bin && (lb[c] = max(lb[c],0.0); ub[c] = min(ub[c],1.0))
    end
    #lpm.cbdata.vartype = vartype
    try
        setvarLB!(lpm, lb)
        setvarUB!(lpm, ub)
        if lpm.param.presolve == GLPK.OFF
            ret_ps = GLPK.simplex(lpm.inner, lpm.smplxparam)
            ret_ps != 0 && return ret_ps
        end
        ret = GLPK.intopt(lpm.inner, lpm.param)
        if ret == GLPK.EMIPGAP || ret == GLPK.ETMLIM || ret == GLPK.ESTOP
            lpm.userlimit = true
        end
    finally
        setvarLB!(lpm, old_lb)
        setvarUB!(lpm, old_ub)
    end
end

function getvarLB(lpm::GLPKOptimizer)
    lp = lpm.inner
    n = GLPK.get_num_cols(lp)
    lb = Array{Float64}(n)
    for c = 1:n
        l = GLPK.get_col_lb(lp, c)
        if l <= -realmax(Float64)
            l = -Inf
        end
        lb[c] = l
    end
    return lb
end

function setvarLB!(lpm::GLPKOptimizer, collb)
    lp = lpm.inner
    n = GLPK.get_num_cols(lp)
    if nonnull(collb) && length(collb) != n
        error("invalid size of collb")
    end
    for c = 1:n
        u = GLPK.get_col_ub(lp, c)
        if u >= realmax(Float64)
            u = Inf
        end
        if nonnull(collb) && collb[c] != -Inf
            l = collb[c]
            if u < Inf
                if l != u
                    GLPK.set_col_bnds(lp, c, GLPK.DB, l, u)
                else
                    GLPK.set_col_bnds(lp, c, GLPK.FX, l, u)
                end
            else
                GLPK.set_col_bnds(lp, c, GLPK.LO, l, 0.0)
            end
        else
            if u < Inf
                GLPK.set_col_bnds(lp, c, GLPK.UP, 0.0, u)
            else
                GLPK.set_col_bnds(lp, c, GLPK.FR, 0.0, 0.0)
            end
        end
    end
end

function getvarUB(lpm::GLPKOptimizer)
    lp = lpm.inner
    n = GLPK.get_num_cols(lp)
    ub = Array{Float64}(n)
    for c = 1:n
        u = GLPK.get_col_ub(lp, c)
        if u >= realmax(Float64)
            u = Inf
        end
        ub[c] = u
    end
    return ub
end

function setvarUB!(lpm::GLPKOptimizer, colub)
    lp = lpm.inner
    n = GLPK.get_num_cols(lp)
    if nonnull(colub) && length(colub) != n
        error("invalid size of colub")
    end
    for c = 1:n
        l = GLPK.get_col_lb(lp, c)
        if l <= -realmax(Float64)
            l = -Inf
        end
        if nonnull(colub) && colub[c] != Inf
            u = colub[c]
            if l > -Inf
                if l != u
                    GLPK.set_col_bnds(lp, c, GLPK.DB, l, u)
                else
                    GLPK.set_col_bnds(lp, c, GLPK.FX, l, u)
                end
            else
                GLPK.set_col_bnds(lp, c, GLPK.UP, 0.0, u)
            end
        else
            if l > -Inf
                GLPK.set_col_bnds(lp, c, GLPK.LO, l, 0.0)
            else
                GLPK.set_col_bnds(lp, c, GLPK.FR, 0.0, 0.0)
            end
        end
    end
end

const vartype_map = Dict(
    GLPK.CV => :Cont,
    GLPK.IV => :Int,
    GLPK.BV => :Bin
)

function getvartype(lpm::GLPKOptimizer)
    lp = lpm.inner
    ncol = GLPK.get_num_cols(lp)
    coltype = Array{Symbol}(ncol)
    for i in 1:ncol
        ct = GLPK.get_col_kind(lp, i)
        coltype[i] = vartype_map[ct]
        if i in lpm.binaries
            coltype[i] = :Bin
        elseif coltype[i] == :Bin # GLPK said it was binary, but we didn't tell it
            coltype[i] = :Int
        end
    end
    return coltype
end
nonnull(x) = (x != nothing && !isempty(x))

# The functions getinfeasibilityray and getunboundedray are adapted from code
# taken from the LEMON C++ optimization library. This is the copyright notice:
#
### Copyright (C) 2003-2010
### Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
### (Egervary Research Group on Combinatorial Optimization, EGRES).
###
### Permission to use, modify and distribute this software is granted
### provided that this copyright notice appears in all copies. For
### precise terms see the accompanying LICENSE file.
###
### This software is provided "AS IS" with no warranty of any kind,
### express or implied, and with no claim as to its suitability for any
### purpose.

function getinfeasibilityray(lpm::GLPKOptimizerLP, ray)
    lp = lpm.inner

    if lpm.method == :Simplex || lpm.method == :Exact
    elseif lpm.method == :InteriorPoint
        error("getinfeasibilityray is not available when using the InteriorPoint method")
    else
        error("bug")
    end

    m = GLPK.get_num_rows(lp)

    # ray = zeros(m)
    @assert length(ray) == m

    ur = GLPK.get_unbnd_ray(lp)
    if ur != 0
        if ur <= m
            k = ur
            get_stat = GLPK.get_row_stat
            get_bind = GLPK.get_row_bind
            get_prim = GLPK.get_row_prim
            get_ub = GLPK.get_row_ub
        else
            k = ur - m
            get_stat = GLPK.get_col_stat
            get_bind = GLPK.get_col_bind
            get_prim = GLPK.get_col_prim
            get_ub = GLPK.get_col_ub
        end

        get_stat(lp, k) == GLPK.BS || error("unbounded ray is primal (use getunboundedray)")

        ray[get_bind(lp, k)] = (get_prim(lp, k) > get_ub(lp, k)) ? -1 : 1

        GLPK.btran(lp, ray)
    else
        eps = 1e-7
        for i = 1:m
            idx = GLPK.get_bhead(lp, i)
            if idx <= m
                k = idx
                get_prim = GLPK.get_row_prim
                get_ub = GLPK.get_row_ub
                get_lb = GLPK.get_row_lb
            else
                k = idx - m
                get_prim = GLPK.get_col_prim
                get_ub = GLPK.get_col_ub
                get_lb = GLPK.get_col_lb
            end

            res = get_prim(lp, k)
            if res > get_ub(lp, k) + eps
                ray[i] = -1
            elseif res < get_lb(lp, k) - eps
                ray[i] = 1
            else
                continue # ray[i] == 0
            end

            if idx <= m
                ray[i] *= GLPK.get_rii(lp, k)
            else
                ray[i] /= GLPK.get_sjj(lp, k)
            end
        end

        GLPK.btran(lp, ray)

        for i = 1:m
            ray[i] /= GLPK.get_rii(lp, i)
        end
    end

    return nothing
end

function getunboundedray(lpm::GLPKOptimizerLP, ray)
    lp = lpm.inner

    if lpm.method == :Simplex || lpm.method == :Exact
    elseif lpm.method == :InteriorPoint
        error("getunboundedray is not available when using the InteriorPoint method")
    else
        error("bug")
    end

    m = GLPK.get_num_rows(lp)
    n = GLPK.get_num_cols(lp)

    # ray = zeros(n)
    @assert length(ray) == n

    ur = GLPK.get_unbnd_ray(lp)
    if ur != 0
        if ur <= m
            k = ur
            get_stat = GLPK.get_row_stat
            get_dual = GLPK.get_row_dual
        else
            k = ur - m
            get_stat = GLPK.get_col_stat
            get_dual = GLPK.get_col_dual
            ray[k] = 1
        end

        get_stat(lp, k) != GLPK.BS || error("unbounded ray is dual (use getinfeasibilityray)")

        for (ri, rv) in zip(GLPK.eval_tab_col(lp, ur)...)
            ri > m && (ray[ri - m] = rv)
        end

        if (GLPK.get_obj_dir(lp) == GLPK.MAX) $ (get_dual(lp, k) > 0)
            scale!(ray, -1.0)
        end
    else
        for i = 1:n
            ray[i] = GLPK.get_col_prim(lp, i)
        end
    end

    return nothing
end
