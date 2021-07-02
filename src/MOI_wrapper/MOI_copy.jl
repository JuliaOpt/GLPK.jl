struct _OptimizerCache
    "Column lower bounds"
    cl::Vector{Float64}
    "Column upper bounds"
    cu::Vector{Float64}
    "Column bound types"
    bounds::Vector{BoundEnum}
    "Column types"
    types::Vector{TypeEnum}
    "Row lower bounds"
    rl::Vector{Float64}
    "Row upper bounds"
    ru::Vector{Float64}
    "A matrix in 1-indexed sparse triplet form"
    I::Vector{Cint}
    J::Vector{Cint}
    V::Vector{Float64}
    function _OptimizerCache(N::Int)
        return new(
            fill(-Inf, N),
            fill(Inf, N),
            fill(NONE, N),
            fill(CONTINUOUS, N),
            Float64[],
            Float64[],
            Cint[],
            Cint[],
            Float64[],
        )
    end
end

"""
    _validate_constraint_types(dest::Optimizer, src::MOI.ModelLike)

Throw an error if unsupported constraint or objective types are present in
`src`.
"""
function _validate_constraint_types(dest::Optimizer, src::MOI.ModelLike)
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(dest, F, S)
            throw(
                MOI.UnsupportedConstraint{F,S}(
                    "GLPK.Optimizer does not support constraints of type $F-in-$S.",
                ),
            )
        end
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            if !MOI.supports(dest, attr, MOI.ConstraintIndex{F,S})
                throw(MOI.UnsupportedAttribute(attr))
            end
        end
    end
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if !MOI.supports(dest, attr)
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if !MOI.supports(dest, attr, MOI.VariableIndex)
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    return
end

function _init_index_map(src::MOI.ModelLike)
    variables = MOI.get(src, MOI.ListOfVariableIndices())
    map = MOIU.IndexMap()
    N = 0
    for x in variables
        N += 1
        map[x] = MOI.VariableIndex(N)
    end
    return variables, map
end

_add_set_data(cache, i, s::MOI.LessThan{Float64}) = (cache.cu[i] = s.upper)
_add_set_data(cache, i, s::MOI.GreaterThan{Float64}) = (cache.cl[i] = s.lower)

function _add_set_data(cache, i, s::MOI.EqualTo{Float64})
    cache.cl[i] = s.value
    cache.cu[i] = s.value
    cache.bounds[i] = EQUAL_TO
    return
end

function _add_set_data(cache, i, s::MOI.Interval{Float64})
    cache.cl[i] = s.lower
    cache.cu[i] = s.upper
    cache.bounds[i] = INTERVAL
    return
end

function _extract_bound_data(src, map, cache, s::Type{S}) where {S}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.SingleVariable,S}())
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        s = MOI.get(src, MOI.ConstraintSet(), ci)
        column = map[f.variable].value
        _add_set_data(cache, column, s)
        map[ci] = MOI.ConstraintIndex{MOI.SingleVariable,S}(column)
    end
    return
end

function _extract_type_data(src, map, cache, ::Type{S}) where {S}
    for ci in MOI.get(src, MOI.ListOfConstraintIndices{MOI.SingleVariable,S}())
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        column = map[f.variable].value
        cache.types[column] = S == MOI.Integer ? INTEGER : BINARY
        map[ci] = MOI.ConstraintIndex{MOI.SingleVariable,S}(column)
    end
    return
end

function _extract_row_data(src, map, cache, ::Type{S}) where {S}
    nnz = length(cache.I)
    row = nnz == 0 ? 1 : cache.I[end] + 1
    for ci in MOI.get(
        src,
        MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},S}(),
    )::Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S}}
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        if !MOIU.is_canonical(f)
            f = MOIU.canonical(f)
        end
        l, u = _bounds(MOI.get(src, MOI.ConstraintSet(), ci))
        push!(cache.rl, l === nothing ? -Inf : l - f.constant)
        push!(cache.ru, u === nothing ? Inf : u - f.constant)
        resize!(cache.I, nnz + length(f.terms))
        resize!(cache.J, nnz + length(f.terms))
        resize!(cache.V, nnz + length(f.terms))
        for term in f.terms
            nnz += 1
            cache.I[nnz] = row
            cache.J[nnz] = Cint(map[term.variable].value::Int64)
            cache.V[nnz] = term.coefficient
        end
        map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S}(row)
        row += 1
    end
    return
end

function _add_all_variables(model::Optimizer, cache::_OptimizerCache)
    N = length(cache.cl)
    glp_add_cols(model, N)
    sizehint!(model.variable_info, N)
    for i in 1:N
        bound = get_moi_bound_type(cache.cl[i], cache.cu[i], cache.bounds[i])
        CleverDicts.add_item(
            model.variable_info,
            VariableInfo(MOI.VariableIndex(i), i, bound, cache.types[i], ""),
        )
        glp_bound_type = get_glp_bound_type(cache.cl[i], cache.cu[i])
        glp_set_col_bnds(model, i, glp_bound_type, cache.cl[i], cache.cu[i])
        if cache.types[i] == BINARY
            model.num_binaries += 1
            glp_set_col_kind(model, i, GLP_IV)
        elseif cache.types[i] == INTEGER
            model.num_integers += 1
            glp_set_col_kind(model, i, GLP_IV)
        end
    end
    return
end

function _add_all_constraints(dest::Optimizer, cache::_OptimizerCache)
    N = length(cache.rl)
    if N == 0
        return  # GLPK doesn't like it if we add 0 rows...
    end
    glp_add_rows(dest, N)
    glp_load_matrix(
        dest,
        length(cache.I),
        offset(cache.I),
        offset(cache.J),
        offset(cache.V),
    )
    sizehint!(dest.affine_constraint_info, N)
    for (i, l, u) in zip(1:N, cache.rl, cache.ru)
        if l == -Inf
            glp_set_row_bnds(dest, i, GLP_UP, -GLP_DBL_MAX, u)
            CleverDicts.add_item(
                dest.affine_constraint_info,
                ConstraintInfo(i, MOI.LessThan{Float64}(u)),
            )
        elseif u == Inf
            glp_set_row_bnds(dest, i, GLP_LO, l, GLP_DBL_MAX)
            CleverDicts.add_item(
                dest.affine_constraint_info,
                ConstraintInfo(i, MOI.GreaterThan{Float64}(l)),
            )
        else
            @assert l ≈ u  # Must be an equal-to constraint!
            glp_set_row_bnds(dest, i, GLP_FX, l, u)
            CleverDicts.add_item(
                dest.affine_constraint_info,
                ConstraintInfo(i, MOI.EqualTo{Float64}(l)),
            )
        end
    end
    return
end

function MOI.copy_to(
    dest::Optimizer,
    src::MOI.ModelLike;
    copy_names::Bool = false,
    kwargs...,
)
    @assert MOI.is_empty(dest)
    _validate_constraint_types(dest, src)
    # Initialize the problem storage
    variables, map = _init_index_map(src)
    cache = _OptimizerCache(length(variables))
    # Extract the problem data
    #   Variable bounds:
    _extract_bound_data(src, map, cache, MOI.GreaterThan{Float64})
    _extract_bound_data(src, map, cache, MOI.LessThan{Float64})
    _extract_bound_data(src, map, cache, MOI.EqualTo{Float64})
    _extract_bound_data(src, map, cache, MOI.Interval{Float64})
    #   Variable types:
    _extract_type_data(src, map, cache, MOI.Integer)
    _extract_type_data(src, map, cache, MOI.ZeroOne)
    #   Affine constraints:
    _extract_row_data(src, map, cache, MOI.GreaterThan{Float64})
    _extract_row_data(src, map, cache, MOI.LessThan{Float64})
    _extract_row_data(src, map, cache, MOI.EqualTo{Float64})
    # Add the problem data
    _add_all_variables(dest, cache)
    _add_all_constraints(dest, cache)
    # Copy model attributes:
    MOIU.pass_attributes(dest, src, copy_names, map)
    MOIU.pass_attributes(dest, src, copy_names, map, variables)
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        indices = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        # TODO(odow): fix copy_names = false.
        MOIU.pass_attributes(dest, src, false, map, indices)
    end
    return map
end
