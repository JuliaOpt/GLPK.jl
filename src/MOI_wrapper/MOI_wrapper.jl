import MathOptInterface

const MOI = MathOptInterface
const MOIU = MOI.Utilities
const CleverDicts = MOI.Utilities.CleverDicts

@enum(TypeEnum, CONTINUOUS, BINARY, INTEGER)
@enum(
    BoundEnum,
    NONE,
    LESS_THAN,
    GREATER_THAN,
    LESS_AND_GREATER_THAN,
    INTERVAL,
    EQUAL_TO
)
@enum(ObjectiveEnum, SINGLE_VARIABLE, SCALAR_AFFINE)
@enum(MethodEnum, SIMPLEX, INTERIOR, EXACT)
@enum(CallbackState, CB_NONE, CB_GENERIC, CB_LAZY, CB_USER_CUT, CB_HEURISTIC)

mutable struct VariableInfo
    index::MOI.VariableIndex
    column::Int
    bound::BoundEnum
    type::TypeEnum
    name::String
end

function VariableInfo(index::MOI.VariableIndex, column::Int)
    return VariableInfo(index, column, NONE, CONTINUOUS, "")
end

struct ConstraintKey
    value::Int64
end
CleverDicts.key_to_index(k::ConstraintKey) = k.value
CleverDicts.index_to_key(::Type{ConstraintKey}, index) = ConstraintKey(index)

_HASH(x) = CleverDicts.key_to_index(x)
_INVERSE_HASH_V(x) = CleverDicts.index_to_key(MOI.VariableIndex, x)
_INVERSE_HASH_C(x) = CleverDicts.index_to_key(ConstraintKey, x)

mutable struct ConstraintInfo
    row::Int
    set::MOI.AbstractSet
    name::String
    ConstraintInfo(set) = new(0, set, "")
    ConstraintInfo(row, set) = new(row, set, "")
end

# Dummy callback function for internal use only. Responsible for updating the
# objective bound, saving the mip gap, and calling the user's callback.
function _internal_callback(tree::Ptr{Cvoid}, info::Ptr{Cvoid})
    callback_data = unsafe_pointer_to_objref(info)::CallbackData
    model = callback_data.model
    callback_data.tree = tree
    node = glp_ios_best_node(tree)
    if node != 0
        model.objective_bound = glp_ios_node_bound(tree, node)
        model.relative_gap = glp_ios_mip_gap(tree)
    end
    model.callback_function(callback_data)
    return nothing
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    # The low-level GLPK problem.
    inner::Ptr{glp_prob}
    method::MethodEnum

    interior_param::glp_iptcp
    intopt_param::glp_iocp
    simplex_param::glp_smcp
    solver_status::Cint
    last_solved_by_mip::Bool
    num_binaries::Int
    num_integers::Int

    objective_bound::Float64
    relative_gap::Float64
    solve_time::Float64
    callback_data::Any

    # A flag to keep track of MOI.Silent, which over-rides the print_level
    # parameter.
    silent::Bool

    # A flag to keep track of MOI.FEASIBILITY_SENSE, since GLPK only stores
    # MIN_SENSE or MAX_SENSE. This allows us to differentiate between MIN_SENSE
    # and FEASIBILITY_SENSE.
    is_feasibility::Bool

    variable_info::CleverDicts.CleverDict{
        MOI.VariableIndex,
        VariableInfo,
        typeof(_HASH),
        typeof(_INVERSE_HASH_V),
    }
    affine_constraint_info::CleverDicts.CleverDict{
        ConstraintKey,
        ConstraintInfo,
        typeof(_HASH),
        typeof(_INVERSE_HASH_C),
    }

    # Mappings from variable and constraint names to their indices. These are
    # lazily built on-demand, so most of the time, they are `nothing`.
    name_to_variable::Union{
        Nothing,
        Dict{String,Union{Nothing,MOI.VariableIndex}},
    }
    name_to_constraint_index::Union{
        Nothing,
        Dict{String,Union{Nothing,MOI.ConstraintIndex}},
    }

    optimize_not_called::Bool

    # These two flags allow us to distinguish between FEASIBLE_POINT and
    # INFEASIBILITY_CERTIFICATE when querying VariablePrimal and ConstraintDual.
    want_infeasibility_certificates::Bool
    unbounded_ray::Union{Vector{Float64},Nothing}
    infeasibility_cert::Union{Vector{Float64},Nothing}

    # Callback fields.
    has_generic_callback::Bool
    callback_state::CallbackState
    lazy_callback::Union{Nothing,Function}
    user_cut_callback::Union{Nothing,Function}
    heuristic_callback::Union{Nothing,Function}

    """
    Optimizer(;
        want_infeasibility_certificates::Bool = true,
        method::MethodEnum = GLPK.SIMPLEX,
    )

    Create a new Optimizer object.

    ## Arguments

     * `want_infeasibility_certificates::Bool`: flag to control whether to
       attempt to generate an infeasibility certificate in the case of primal or
       dual infeasibility. Defaults to `true`. You should set this to `false` if
       you want GLPK to report primal or dual infeasiblity, but you don't need
       a certificate.

     * `method::MethodEnum`: Solution method to use. Default is `GLPK.SIMPLEX`.
       Other options are `GLPK.EXACT` and `GLPK.INTERIOR`.
    """
    function Optimizer(;
        want_infeasibility_certificates::Bool = true,
        method::MethodEnum = SIMPLEX,
        kwargs...,
    )
        model = new()
        model.inner = glp_create_prob()
        model.method = method
        model.want_infeasibility_certificates = want_infeasibility_certificates

        model.interior_param = glp_iptcp()
        glp_init_iptcp(model.interior_param)
        model.intopt_param = glp_iocp()
        glp_init_iocp(model.intopt_param)
        model.simplex_param = glp_smcp()
        glp_init_smcp(model.simplex_param)

        MOI.set(model, MOI.RawOptimizerAttribute("msg_lev"), GLP_MSG_ERR)
        if length(kwargs) > 0
            @warn(
                "Passing parameters as keyword arguments is deprecated. Use " *
                "`JuMP.set_optimizer_attribute` or `MOI.RawOptimizerAttribute(key)` " *
                "instead."
            )
        end
        for (key, val) in kwargs
            MOI.set(model, MOI.RawOptimizerAttribute(String(key)), val)
        end
        model.silent = false
        model.variable_info =
            CleverDicts.CleverDict{MOI.VariableIndex,VariableInfo}(
                _HASH,
                _INVERSE_HASH_V,
            )
        model.affine_constraint_info =
            CleverDicts.CleverDict{ConstraintKey,ConstraintInfo}(
                _HASH,
                _INVERSE_HASH_C,
            )

        MOI.empty!(model)

        finalizer(model) do m
            return glp_delete_prob(m)
        end
        return model
    end
end

Base.cconvert(::Type{Ptr{glp_prob}}, m::Optimizer) = m
Base.unsafe_convert(::Type{Ptr{glp_prob}}, m::Optimizer) = m.inner

mutable struct CallbackData
    c_callback::Base.CFunction
    tree::Ptr{Cvoid}
    exception::Union{Nothing,Exception}
end

Base.broadcastable(x::CallbackData) = Ref(x)

# Dummy callback function for internal use only. Responsible for updating the
# objective bound, saving the mip gap, and calling the user's callback.
#
# !!! Very Important Note
#
# If Julia throws an exception from within the callback, the GLPK model does not
# gracefully exit! Instead, it throws a `glp_delete_prob: operation not
# supported` error when Julia tries to finalize `prob`. This is very annoying to
# debug.
#
# As a work-around, we catch all Julia exceptions with a try-catch, terminate
# the callback with `ios_terminate`, store the exception in `cb_data.exception`,
# and then re-throw it once we have gracefully exited from the callback.
#
# See also: the note in `_solve_mip_problem`.
function _set_callback(model::Optimizer, callback_function::Function)
    internal_callback =
        (tree::Ptr{Cvoid}, info::Ptr{Cvoid}) -> begin
            cb_data = unsafe_pointer_to_objref(info)::CallbackData
            node = glp_ios_best_node(tree)
            if node != 0
                model.objective_bound = glp_ios_node_bound(tree, node)
                model.relative_gap = glp_ios_mip_gap(tree)
            end
            try
                cb_data.tree = tree
                callback_function(cb_data)
            catch ex
                glp_ios_terminate(tree)
                cb_data.exception = ex
            end
            return Cint(0)
        end
    c_callback = @cfunction($internal_callback, Cint, (Ptr{Cvoid}, Ptr{Cvoid}))
    model.callback_data = CallbackData(c_callback, C_NULL, nothing)
    model.intopt_param.cb_func = c_callback.ptr
    model.intopt_param.cb_info = pointer_from_objref(model.callback_data)
    return
end

Base.show(io::IO, model::Optimizer) = print(io, "A GLPK model")

function MOI.empty!(model::Optimizer)
    glp_erase_prob(model)
    model.solver_status = GLP_UNDEF
    model.last_solved_by_mip = false
    model.num_binaries = 0
    model.num_integers = 0
    model.objective_bound = NaN
    model.relative_gap = NaN
    model.solve_time = NaN
    model.is_feasibility = true
    model.optimize_not_called = true
    empty!(model.variable_info)
    empty!(model.affine_constraint_info)
    model.name_to_variable = nothing
    model.name_to_constraint_index = nothing
    model.unbounded_ray = nothing
    model.infeasibility_cert = nothing
    model.has_generic_callback = false
    model.callback_state = CB_NONE
    model.lazy_callback = nothing
    model.user_cut_callback = nothing
    model.heuristic_callback = nothing
    _set_callback(model, cb_data -> nothing)
    return
end

function MOI.is_empty(model::Optimizer)
    model.is_feasibility == false && return false
    !isempty(model.variable_info) && return false
    !isempty(model.affine_constraint_info) && return false
    model.name_to_variable !== nothing && return false
    model.name_to_constraint_index !== nothing && return false
    model.unbounded_ray !== nothing && return false
    model.infeasibility_cert !== nothing && return false
    model.has_generic_callback && return false
    model.callback_state != CB_NONE && return false
    model.lazy_callback !== nothing && return false
    model.user_cut_callback !== nothing && return false
    model.heuristic_callback !== nothing && return false
    return true
end

MOI.get(::Optimizer, ::MOI.SolverName) = "GLPK"

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.SingleVariable},
    ::Type{F},
) where {
    F<:Union{
        MOI.EqualTo{Float64},
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
        MOI.Interval{Float64},
        MOI.ZeroOne,
        MOI.Integer,
    },
}
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{F},
) where {
    F<:Union{
        MOI.EqualTo{Float64},
        MOI.LessThan{Float64},
        MOI.GreaterThan{Float64},
    },
}
    return true
end

const _SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintName,
    ::Type{<:MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}}},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.Name) = true
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

"""
    _set_parameter(param_store, key::Symbol, value)::Bool

Set the field name `key` in a `param_store` type (that is one of `InteriorParam`,
`IntoptParam`, or `SimplexParam`) to `value`.

Returns a `Bool` indicating if the parameter was set.
"""
function _set_parameter(param_store, key::Symbol, value)
    if key == :cb_func || key == :cb_info
        error(
            "Invalid option: $(string(key)). Use the MOI attribute " *
            "`GLPK.CallbackFunction` instead.",
        )
    elseif key in fieldnames(typeof(param_store))
        field_type = typeof(getfield(param_store, key))
        setfield!(param_store, key, convert(field_type, value))
        return true
    end
    return false
end

function MOI.set(model::Optimizer, param::MOI.RawOptimizerAttribute, value)
    key = Symbol(param.name)
    set_interior = _set_parameter(model.interior_param, key, value)
    set_intopt = _set_parameter(model.intopt_param, key, value)
    set_simplex = _set_parameter(model.simplex_param, key, value)
    if !set_interior && !set_intopt && !set_simplex
        throw(MOI.UnsupportedAttribute(param))
    end
    return
end

function MOI.get(model::Optimizer, param::MOI.RawOptimizerAttribute)
    name = Symbol(param.name)
    if (model.method == SIMPLEX || model.method == EXACT) &&
       name in fieldnames(glp_smcp)
        return getfield(model.simplex_param, name)
    elseif model.method == INTERIOR && name in fieldnames(glp_iptcp)
        return getfield(model.interior_param, name)
    elseif name in fieldnames(glp_iocp)
        return getfield(model.intopt_param, name)
    end
    return throw(MOI.UnsupportedAttribute(param))
end

_limit_sec_to_ms(::Nothing) = typemax(Int32)
_limit_sec_to_ms(x::Real) = ceil(Int32, min(typemax(Int32), 1_000 * x))
function MOI.set(
    model::Optimizer,
    ::MOI.TimeLimitSec,
    limit::Union{Nothing,Real},
)
    MOI.set(model, MOI.RawOptimizerAttribute("tm_lim"), _limit_sec_to_ms(limit))
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    # convert internal ms to sec
    return MOI.get(model, MOI.RawOptimizerAttribute("tm_lim")) / 1_000
end

MOI.supports_incremental_interface(::Optimizer, ::Bool) = true

function MOI.get(model::Optimizer, ::MOI.ListOfVariableAttributesSet)
    return MOI.AbstractVariableAttribute[MOI.VariableName()]
end

function MOI.get(model::Optimizer, ::MOI.ListOfModelAttributesSet)
    obj_func_type = MOI.get(model, MOI.ObjectiveFunctionType())
    attributes = [MOI.ObjectiveSense(), MOI.ObjectiveFunction{obj_func_type}()]
    if MOI.get(model, MOI.Name()) != ""
        push!(attributes, MOI.Name())
    end
    return attributes
end

function MOI.get(
    ::Optimizer,
    ::MOI.ListOfConstraintAttributesSet,
)
    return MOI.AbstractConstraintAttribute[MOI.ConstraintName()]
end

function MOI.get(
    ::Optimizer,
    ::MOI.ListOfConstraintAttributesSet{
        MOI.SingleVariable,
        <:MOI.AbstractScalarSet,
    },
)
    return MOI.AbstractConstraintAttribute[]
end

function _indices_and_coefficients(
    indices::Vector{Cint},
    coefficients::Vector{Float64},
    model::Optimizer,
    f::MOI.ScalarAffineFunction{Float64},
)
    i = 1
    for term in f.terms
        indices[i] = Cint(column(model, term.variable))
        coefficients[i] = term.coefficient
        i += 1
    end
    return indices, coefficients
end

function _indices_and_coefficients(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{Float64},
)
    f_canon = MOI.Utilities.canonical(f)
    nnz = length(f_canon.terms)
    indices, coefficients = zeros(Cint, nnz), zeros(Cdouble, nnz)
    _indices_and_coefficients(indices, coefficients, model, f_canon)
    return indices, coefficients
end

_sense_and_rhs(s::MOI.LessThan{Float64}) = (Cchar('L'), s.upper)
_sense_and_rhs(s::MOI.GreaterThan{Float64}) = (Cchar('G'), s.lower)
_sense_and_rhs(s::MOI.EqualTo{Float64}) = (Cchar('E'), s.value)

###
### Variables
###

# Short-cuts to return the VariableInfo associated with an index.
function _info(model::Optimizer, key::MOI.VariableIndex)
    if haskey(model.variable_info, key)
        return model.variable_info[key]
    end
    return throw(MOI.InvalidIndex(key))
end

column(model, x::MOI.VariableIndex) = _info(model, x).column

function MOI.add_variable(model::Optimizer)
    # Initialize `VariableInfo` with a dummy `VariableIndex` and a column,
    # because we need `add_item` to tell us what the `VariableIndex` is.
    index = CleverDicts.add_item(
        model.variable_info,
        VariableInfo(MOI.VariableIndex(0), 0),
    )
    info = _info(model, index)
    # Now, set `.index` and `.column`.
    info.index = index
    info.column = length(model.variable_info)
    glp_add_cols(model, 1)
    glp_set_col_bnds(model, info.column, GLP_FR, 0.0, 0.0)
    return index
end

function MOI.add_variables(model::Optimizer, N::Int)
    indices = Vector{MOI.VariableIndex}(undef, N)
    num_variables = length(model.variable_info)
    glp_add_cols(model, N)
    for i in 1:N
        # Initialize `VariableInfo` with a dummy `VariableIndex` and a column,
        # because we need `add_item` to tell us what the `VariableIndex` is.
        index = CleverDicts.add_item(
            model.variable_info,
            VariableInfo(MOI.VariableIndex(0), 0),
        )
        info = _info(model, index)
        # Now, set `.index` and `.column`.
        info.index = index
        info.column = num_variables + i
        glp_set_col_bnds(model, info.column, GLP_FR, 0.0, 0.0)
        indices[i] = index
    end
    return indices
end

function MOI.is_valid(model::Optimizer, v::MOI.VariableIndex)
    return haskey(model.variable_info, v)
end

function MOI.delete(model::Optimizer, v::MOI.VariableIndex)
    info = _info(model, v)
    glp_std_basis(model)
    c = Cint[info.column]
    glp_del_cols(model, 1, offset(c))
    delete!(model.variable_info, v)
    for other_info in values(model.variable_info)
        if other_info.column > info.column
            other_info.column -= 1
        end
    end
    model.name_to_variable = nothing
    model.name_to_constraint_index = nothing
    return
end

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, name::String)
    if model.name_to_variable === nothing
        _rebuild_name_to_variable(model)
    end
    if haskey(model.name_to_variable, name)
        variable = model.name_to_variable[name]
        if variable === nothing
            error("Duplicate name detected: $(name)")
        end
        return variable
    end
    return nothing
end

function _rebuild_name_to_variable(model::Optimizer)
    model.name_to_variable = Dict{String,Union{Nothing,MOI.VariableIndex}}()
    for (index, info) in model.variable_info
        if isempty(info.name)
            continue
        end
        if haskey(model.name_to_variable, info.name)
            model.name_to_variable[info.name] = nothing
        else
            model.name_to_variable[info.name] = index
        end
    end
    return
end

function MOI.get(model::Optimizer, ::MOI.VariableName, v::MOI.VariableIndex)
    return _info(model, v).name
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariableName,
    v::MOI.VariableIndex,
    name::String,
)
    info = _info(model, v)
    info.name = name
    if !isempty(name) && isascii(name)
        # Note: GLPK errors if we try to set non-ascii column names.
        glp_set_col_name(model, info.column, name)
    end
    model.name_to_variable = nothing
    return
end

###
### Objectives
###

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    if sense == MOI.MIN_SENSE
        glp_set_obj_dir(model, GLP_MIN)
        model.is_feasibility = false
    elseif sense == MOI.MAX_SENSE
        glp_set_obj_dir(model, GLP_MAX)
        model.is_feasibility = false
    else
        @assert sense == MOI.FEASIBILITY_SENSE
        glp_set_obj_dir(model, GLP_MIN)
        model.is_feasibility = true
    end
    return
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveSense)
    sense = glp_get_obj_dir(model)
    if model.is_feasibility
        return MOI.FEASIBILITY_SENSE
    elseif sense == GLP_MAX
        return MOI.MAX_SENSE
    else
        @assert sense == GLP_MIN
        return MOI.MIN_SENSE
    end
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunction{F}) where {F}
    obj = MOI.get(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
    )
    return convert(F, obj)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction{F},
    f::F,
) where {F<:MOI.ScalarAffineFunction{Float64}}
    num_vars = length(model.variable_info)
    obj = zeros(Float64, num_vars)
    for term in f.terms
        col = column(model, term.variable)
        obj[col] += term.coefficient
    end
    for (col, coef) in enumerate(obj)
        glp_set_obj_coef(model, col, coef)
    end
    glp_set_obj_coef(model, 0, f.constant)
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    dest = zeros(length(model.variable_info))
    for col in 1:length(dest)
        dest[col] = glp_get_obj_coef(model, col)
    end
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (index, info) in model.variable_info
        coefficient = dest[info.column]
        iszero(coefficient) && continue
        push!(terms, MOI.ScalarAffineTerm(coefficient, index))
    end
    constant = glp_get_obj_coef(model, 0)
    return MOI.ScalarAffineFunction(terms, constant)
end

function MOI.modify(
    model::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.ScalarConstantChange{Float64},
)
    glp_set_obj_coef(model, 0, chg.new_constant)
    return
end

##
##  SingleVariable-in-Set constraints.
##

function _info(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,<:Any},
)
    var_index = MOI.VariableIndex(c.value)
    if haskey(model.variable_info, var_index)
        return _info(model, var_index)
    end
    return throw(MOI.InvalidIndex(c))
end

function column(model, c::MOI.ConstraintIndex{MOI.SingleVariable,<:Any})
    return _info(model, c).column
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    if haskey(model.variable_info, MOI.VariableIndex(c.value))
        info = _info(model, c)
        return info.bound == LESS_THAN || info.bound == LESS_AND_GREATER_THAN
    end
    return false
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    if haskey(model.variable_info, MOI.VariableIndex(c.value))
        info = _info(model, c)
        return info.bound == GREATER_THAN || info.bound == LESS_AND_GREATER_THAN
    end
    return false
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.Interval{Float64}},
)
    return haskey(model.variable_info, MOI.VariableIndex(c.value)) &&
           _info(model, c).bound == INTERVAL
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    return haskey(model.variable_info, MOI.VariableIndex(c.value)) &&
           _info(model, c).bound == EQUAL_TO
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.ZeroOne},
)
    return haskey(model.variable_info, MOI.VariableIndex(c.value)) &&
           _info(model, c).type == BINARY
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.Integer},
)
    return haskey(model.variable_info, MOI.VariableIndex(c.value)) &&
           _info(model, c).type == INTEGER
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable,<:Any},
)
    MOI.throw_if_not_valid(model, c)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable,<:Any},
    ::MOI.SingleVariable,
)
    return throw(MOI.SettingSingleVariableFunctionNotAllowed())
end

_bounds(s::MOI.GreaterThan{Float64}) = (s.lower, nothing)
_bounds(s::MOI.LessThan{Float64}) = (nothing, s.upper)
_bounds(s::MOI.EqualTo{Float64}) = (s.value, s.value)
_bounds(s::MOI.Interval{Float64}) = (s.lower, s.upper)

function _throw_if_existing_lower(
    bound::BoundEnum,
    ::TypeEnum,
    new_set::Type{<:MOI.AbstractSet},
    variable::MOI.VariableIndex,
)
    existing_set = if bound == LESS_AND_GREATER_THAN || bound == GREATER_THAN
        MOI.GreaterThan{Float64}
    elseif bound == INTERVAL
        MOI.Interval{Float64}
    elseif bound == EQUAL_TO
        MOI.EqualTo{Float64}
    else
        nothing  # Also covers `NONE` and `LESS_THAN`.
    end
    if existing_set !== nothing
        throw(MOI.LowerBoundAlreadySet{existing_set,new_set}(variable))
    end
end

function _throw_if_existing_upper(
    bound::BoundEnum,
    ::TypeEnum,
    new_set::Type{<:MOI.AbstractSet},
    variable::MOI.VariableIndex,
)
    existing_set = if bound == LESS_AND_GREATER_THAN || bound == LESS_THAN
        MOI.LessThan{Float64}
    elseif bound == INTERVAL
        MOI.Interval{Float64}
    elseif bound == EQUAL_TO
        MOI.EqualTo{Float64}
    else
        nothing  # Also covers `NONE` and `GREATER_THAN`.
    end
    if existing_set !== nothing
        throw(MOI.UpperBoundAlreadySet{existing_set,new_set}(variable))
    end
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.SingleVariable,
    s::S,
) where {S<:_SCALAR_SETS}
    info = _info(model, f.variable)
    if S <: MOI.LessThan{Float64}
        _throw_if_existing_upper(info.bound, info.type, S, f.variable)
        info.bound =
            info.bound == GREATER_THAN ? LESS_AND_GREATER_THAN : LESS_THAN
    elseif S <: MOI.GreaterThan{Float64}
        _throw_if_existing_lower(info.bound, info.type, S, f.variable)
        info.bound =
            info.bound == LESS_THAN ? LESS_AND_GREATER_THAN : GREATER_THAN
    elseif S <: MOI.EqualTo{Float64}
        _throw_if_existing_lower(info.bound, info.type, S, f.variable)
        _throw_if_existing_upper(info.bound, info.type, S, f.variable)
        info.bound = EQUAL_TO
    else
        @assert S <: MOI.Interval{Float64}
        _throw_if_existing_lower(info.bound, info.type, S, f.variable)
        _throw_if_existing_upper(info.bound, info.type, S, f.variable)
        info.bound = INTERVAL
    end
    index = MOI.ConstraintIndex{MOI.SingleVariable,typeof(s)}(f.variable.value)
    MOI.set(model, MOI.ConstraintSet(), index, s)
    return index
end

const GLP_DBL_MAX = prevfloat(Inf)

function get_glp_bound_type(lower, upper)
    if lower == upper
        GLP_FX
    elseif lower <= -GLP_DBL_MAX
        upper >= GLP_DBL_MAX ? GLP_FR : GLP_UP
    else
        upper >= GLP_DBL_MAX ? GLP_LO : GLP_DB
    end
end
function get_moi_bound_type(lower, upper, type)
    if type == INTERVAL
        INTERVAL
    elseif type == EQUAL_TO
        EQUAL_TO
    elseif lower == upper
        LESS_AND_GREATER_THAN
    elseif lower <= -GLP_DBL_MAX
        upper >= GLP_DBL_MAX ? NONE : LESS_THAN
    else
        upper >= GLP_DBL_MAX ? GREATER_THAN : LESS_AND_GREATER_THAN
    end
end

function _set_variable_bound(
    model::Optimizer,
    column::Int,
    lower::Union{Nothing,Float64},
    upper::Union{Nothing,Float64},
)
    if lower === nothing
        lower = glp_get_col_lb(model, column)
    end
    if upper === nothing
        upper = glp_get_col_ub(model, column)
    end
    bound_type = get_glp_bound_type(lower, upper)
    glp_set_col_bnds(model, column, bound_type, lower, upper)
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    info = _info(model, c)
    _set_variable_bound(model, info.column, nothing, Inf)
    if info.bound == LESS_AND_GREATER_THAN
        info.bound = GREATER_THAN
    else
        info.bound = NONE
    end
    model.name_to_constraint_index = nothing
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    info = _info(model, c)
    _set_variable_bound(model, info.column, -Inf, nothing)
    info.bound = info.bound == LESS_AND_GREATER_THAN ? LESS_THAN : NONE
    model.name_to_constraint_index = nothing
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.Interval{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    info = _info(model, c)
    _set_variable_bound(model, info.column, -Inf, Inf)
    info.bound = NONE
    model.name_to_constraint_index = nothing
    return
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    info = _info(model, c)
    _set_variable_bound(model, info.column, -Inf, Inf)
    info.bound = NONE
    model.name_to_constraint_index = nothing
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    lower = glp_get_col_lb(model, column(model, c))
    return MOI.GreaterThan(lower)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    upper = glp_get_col_ub(model, column(model, c))
    return MOI.LessThan(upper)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    lower = glp_get_col_lb(model, column(model, c))
    return MOI.EqualTo(lower)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.Interval{Float64}},
)
    MOI.throw_if_not_valid(model, c)
    col = column(model, c)
    lower = glp_get_col_lb(model, col)
    upper = glp_get_col_ub(model, col)
    return MOI.Interval(lower, upper)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,S},
    s::S,
) where {S<:_SCALAR_SETS}
    MOI.throw_if_not_valid(model, c)
    lower, upper = _bounds(s)
    _set_variable_bound(model, column(model, c), lower, upper)
    return
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.SingleVariable,
    ::MOI.ZeroOne,
)
    info = _info(model, f.variable)
    # See https://github.com/JuliaOpt/GLPKMathProgInterface.jl/pull/15
    # for why this is necesary. GLPK interacts weirdly with binary variables and
    # bound modification. So let's set binary variables as "Integer" with [0,1]
    # bounds that we enforce just before solve.
    glp_set_col_kind(model, info.column, GLP_IV)
    info.type = BINARY
    model.num_binaries += 1
    return MOI.ConstraintIndex{MOI.SingleVariable,MOI.ZeroOne}(f.variable.value)
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.ZeroOne},
)
    MOI.throw_if_not_valid(model, c)
    info = _info(model, c)
    glp_set_col_kind(model, info.column, GLP_CV)
    info.type = CONTINUOUS
    model.num_binaries -= 1
    model.name_to_constraint_index = nothing
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.ZeroOne},
)
    MOI.throw_if_not_valid(model, c)
    return MOI.ZeroOne()
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.SingleVariable,
    ::MOI.Integer,
)
    info = _info(model, f.variable)
    glp_set_col_kind(model, info.column, GLP_IV)
    info.type = INTEGER
    model.num_integers += 1
    return MOI.ConstraintIndex{MOI.SingleVariable,MOI.Integer}(f.variable.value)
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.Integer},
)
    MOI.throw_if_not_valid(model, c)
    info = _info(model, c)
    glp_set_col_kind(model, info.column, GLP_CV)
    info.type = CONTINUOUS
    model.num_integers -= 1
    model.name_to_constraint_index = nothing
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.Integer},
)
    MOI.throw_if_not_valid(model, c)
    return MOI.Integer()
end

###
### ScalarAffineFunction-in-Set
###

function _info(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}},
)
    key = ConstraintKey(c.value)
    if haskey(model.affine_constraint_info, key)
        return model.affine_constraint_info[key]
    end
    return throw(MOI.InvalidIndex(c))
end

function MOI.is_valid(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
) where {S}
    key = ConstraintKey(c.value)
    if haskey(model.affine_constraint_info, key)
        info = model.affine_constraint_info[key]
        return typeof(info.set) == S
    else
        return false
    end
end

"""
    _add_affine_constraint(
        problem::Union{Optimizer, Ptr{glp_prob}},
        columns::Vector{Cint},
        coefficients::Vector{Float64},
        sense::Cchar,
        rhs::Float64,
    )

Helper function to add a row to the problem. Sense must be one of `'E'`
(ax == b), `'G'` (ax >= b), `'L'` (ax <= b).
"""
function _add_affine_constraint(
    problem::Union{Optimizer,Ptr{glp_prob}},
    indices::Vector{Cint},
    coefficients::Vector{Float64},
    sense::Cchar,
    rhs::Float64,
)
    if length(indices) != length(coefficients)
        error("columns and coefficients have different lengths.")
    end
    glp_add_rows(problem, 1)
    row = glp_get_num_rows(problem)
    glp_set_mat_row(
        problem,
        row,
        length(indices),
        offset(indices),
        offset(coefficients),
    )
    # According to http://most.ccib.rutgers.edu/glpk.pdf page 22, the `lb`
    # argument is ignored for constraint types with no lower bound (GLP_UP) and
    # the `ub` argument is ignored for constraint types with no upper bound
    # (GLP_LO). We pass ±GLP_DBL_MAX for those unused bounds since (a) we have
    # to pass something, and (b) it is consistent with the other usages of
    # ±GLP_DBL_MAX to represent infinite bounds in the rest of the GLPK
    # interface.
    if sense == Cchar('E')
        glp_set_row_bnds(problem, row, GLP_FX, rhs, rhs)
    elseif sense == Cchar('G')
        glp_set_row_bnds(problem, row, GLP_LO, rhs, GLP_DBL_MAX)
    else
        @assert sense == Cchar('L')
        glp_set_row_bnds(problem, row, GLP_UP, -GLP_DBL_MAX, rhs)
    end
    return
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarAffineFunction{Float64},
    s::Union{
        MOI.GreaterThan{Float64},
        MOI.LessThan{Float64},
        MOI.EqualTo{Float64},
    },
)
    if !iszero(f.constant)
        throw(
            MOI.ScalarFunctionConstantNotZero{Float64,typeof(f),typeof(s)}(
                f.constant,
            ),
        )
    end
    key = CleverDicts.add_item(model.affine_constraint_info, ConstraintInfo(s))
    model.affine_constraint_info[key].row = length(model.affine_constraint_info)
    indices, coefficients = _indices_and_coefficients(model, f)
    sense, rhs = _sense_and_rhs(s)
    _add_affine_constraint(model, indices, coefficients, sense, rhs)
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(key.value)
end

function MOI.delete(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:Any},
)
    row = _info(model, c).row
    glp_std_basis(model)
    x = Cint[row]
    glp_del_rows(model, 1, offset(x))
    for info in values(model.affine_constraint_info)
        if info.row > row
            info.row -= 1
        end
    end
    key = ConstraintKey(c.value)
    delete!(model.affine_constraint_info, key)
    model.name_to_constraint_index = nothing
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
) where {S}
    row = _info(model, c).row
    sense = glp_get_row_type(model, row)
    if sense == GLP_LO || sense == GLP_FX || sense == GLP_DB
        return S(glp_get_row_lb(model, row))
    else
        return S(glp_get_row_ub(model, row))
    end
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S},
    s::S,
) where {S<:Union{MOI.LessThan,MOI.GreaterThan,MOI.EqualTo}}
    row = _info(model, c).row
    if S <: MOI.LessThan
        glp_set_row_bnds(model, row, GLP_UP, -GLP_DBL_MAX, s.upper)
    elseif S <: MOI.GreaterThan
        glp_set_row_bnds(model, row, GLP_LO, s.lower, GLP_DBL_MAX)
    else
        @assert S <: MOI.EqualTo
        glp_set_row_bnds(model, row, GLP_FX, s.value, s.value)
    end
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:Any},
)
    row = Cint(_info(model, c).row)
    nnz = glp_get_mat_row(model, row, C_NULL, C_NULL)
    indices, coefficients = zeros(Cint, nnz), zeros(Cdouble, nnz)
    glp_get_mat_row(model, row, offset(indices), offset(coefficients))
    terms = MOI.ScalarAffineTerm{Float64}[]
    for (col, val) in zip(indices, coefficients)
        if iszero(val)
            continue
        end
        push!(
            terms,
            MOI.ScalarAffineTerm(
                val,
                model.variable_info[CleverDicts.LinearIndex(col)].index,
            ),
        )
    end
    return MOI.ScalarAffineFunction(terms, 0.0)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:Any},
)
    return _info(model, c).name
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintName,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:Any},
    name::String,
)
    info = _info(model, c)
    old_name = info.name
    info.name = name
    if !isempty(name) && isascii(name)
        # Note: GLPK errors if we try to set non-ascii row names.
        glp_set_row_name(model, info.row, name)
    end
    model.name_to_constraint_index = nothing
    return
end

function MOI.get(model::Optimizer, ::Type{MOI.ConstraintIndex}, name::String)
    if model.name_to_constraint_index === nothing
        _rebuild_name_to_constraint_index(model)
    end
    if haskey(model.name_to_constraint_index, name)
        constr = model.name_to_constraint_index[name]
        if constr === nothing
            error("Duplicate constraint name detected: $(name)")
        end
        return constr
    end
    return nothing
end

function MOI.get(
    model::Optimizer,
    C::Type{MOI.ConstraintIndex{F,S}},
    name::String,
) where {F,S}
    index = MOI.get(model, MOI.ConstraintIndex, name)
    if typeof(index) == C
        return index::MOI.ConstraintIndex{F,S}
    end
    return nothing
end

function _rebuild_name_to_constraint_index(model::Optimizer)
    model.name_to_constraint_index =
        Dict{String,Union{Nothing,MOI.ConstraintIndex}}()
    for (key, info) in model.affine_constraint_info
        if isempty(info.name)
            continue
        end
        _set_name_to_constraint_index(
            model,
            info.name,
            MOI.ConstraintIndex{
                MOI.ScalarAffineFunction{Float64},
                typeof(info.set),
            }(
                key.value,
            ),
        )
    end
    return
end

function _set_name_to_constraint_index(
    model::Optimizer,
    name::String,
    index::MOI.ConstraintIndex,
)
    if haskey(model.name_to_constraint_index, name)
        model.name_to_constraint_index[name] = nothing
    else
        model.name_to_constraint_index[name] = index
    end
    return
end

###
### Optimize methods.
###

function _solve_linear_problem(model::Optimizer)
    model.last_solved_by_mip = false
    if model.method == SIMPLEX
        model.solver_status = glp_simplex(model, model.simplex_param)
    elseif model.method == EXACT
        model.solver_status = glp_exact(model, model.simplex_param)
    else
        @assert model.method == INTERIOR
        model.solver_status = glp_interior(model, model.interior_param)
    end
    return
end

"""
    _round_bounds_to_integer(model)

GLPK does not allow integer variables with fractional bounds. Therefore, we
round the bounds of binary and integer variables to integer values prior to
solving.

Returns a tuple `(column, lower, upper)` for the bounds that need to be reset.
"""
function _round_bounds_to_integer(model::Optimizer)
    bounds_to_reset = Tuple{Int,Float64,Float64}[]
    for (key, info) in model.variable_info
        if info.type == BINARY || info.type == INTEGER
            lb = glp_get_col_lb(model, info.column)
            ub = glp_get_col_ub(model, info.column)
            new_lb = ceil(lb)
            new_ub = floor(ub)
            if info.type == BINARY
                new_lb = max(0.0, new_lb)
                new_ub = min(1.0, new_ub)
            end
            if new_lb != lb || new_ub != ub
                push!(bounds_to_reset, (info.column, lb, ub))
                _set_variable_bound(model, info.column, new_lb, new_ub)
            end
        end
    end
    return bounds_to_reset
end

function _solve_mip_problem(model::Optimizer)
    bounds_to_reset = _round_bounds_to_integer(model)
    # Because we're muddling with the presolve in this function, cache the
    # original setting so that it can be reset.
    presolve_cache = model.intopt_param.presolve
    try
        # glp_intopt requires a starting basis for the LP relaxation. There are
        # two ways to get this. If presolve=GLP_ON, then the presolve will find
        # a basis. If presolve=GLP_OFF, then we should solve the problem via
        # glp_simplex first.
        if model.intopt_param.presolve == GLP_OFF
            glp_simplex(model, model.simplex_param)
            if glp_get_status(model) != GLP_OPT
                # We didn't find an optimal solution to the LP relaxation, so
                # let's turn presolve on and let intopt figure out what the
                # problem is.
                model.intopt_param.presolve = GLP_ON
            end
        end
        model.solver_status = glp_intopt(model, model.intopt_param)
        model.last_solved_by_mip = true

        # !!! Very Important Note
        #
        #  This next bit is _very_ important! See the note associated with
        #  set_callback.
        if model.callback_data.exception !== nothing
            throw(model.callback_data.exception)
        end

    finally
        for (column, lower, upper) in bounds_to_reset
            _set_variable_bound(model, column, lower, upper)
        end
        model.intopt_param.presolve = presolve_cache
    end
    return
end

include("infeasibility_certificates.jl")

function _check_moi_callback_validity(model::Optimizer)
    has_moi_callback =
        model.lazy_callback !== nothing ||
        model.user_cut_callback !== nothing ||
        model.heuristic_callback !== nothing
    if has_moi_callback && model.has_generic_callback
        error(
            "Cannot use `GLPK.CallbackFunction` as well as " *
            "`MOI.AbstractCallbackFunction`.",
        )
    end
    return has_moi_callback
end

function MOI.optimize!(model::Optimizer)
    start_time = time()
    model.optimize_not_called = false
    model.infeasibility_cert = nothing
    model.unbounded_ray = nothing

    # Initialize callbacks if necessary.
    if _check_moi_callback_validity(model)
        MOI.set(model, CallbackFunction(), _default_moi_callback(model))
        model.has_generic_callback = false
    end

    if model.num_binaries > 0 || model.num_integers > 0
        _solve_mip_problem(model)
    else
        _solve_linear_problem(model)
    end
    if _certificates_potentially_available(model)
        (status, _) = _get_status(model)
        if status == MOI.DUAL_INFEASIBLE
            ray = zeros(glp_get_num_cols(model))
            if _get_unbounded_ray(model, ray)
                model.unbounded_ray = ray
            end
        elseif status == MOI.INFEASIBLE
            ray = zeros(glp_get_num_rows(model))
            if _get_infeasibility_ray(model, ray)
                model.infeasibility_cert = ray
            end
        end
    end
    model.solve_time = time() - start_time
    return
end

function _throw_if_optimize_in_progress(model::Optimizer, attr)
    if model.callback_state != CB_NONE
        throw(MOI.OptimizeInProgress(attr))
    end
end

# GLPK has a complicated status reporting system because it can be solved via
# multiple different solution algorithms. Regardless of the algorithm, the
# return value is stored in `model.solver_status`.
#
# Note that the first status (`Int32(0)`) should map to a `SUCCESS` status,
# because it doesn't imply anything about the solution. If `solver_status` is
# `Int32(0)`, then a solution-specific status can be queried with `_get_status`.

const _RAW_SIMPLEX_STRINGS =
    Dict{Int32,Tuple{MOI.TerminationStatusCode,String}}(
        GLP_EBADB => (
            MOI.INVALID_MODEL,
            "Unable to start the search, because the initial basis specified in the problem object is invalid—the number of basic (auxiliary and structural) variables is not the same as the number of rows in the problem object.",
        ),
        GLP_ESING => (
            MOI.NUMERICAL_ERROR,
            "Unable to start the search, because the basis matrix corresponding to the initial basis is singular within the working precision.",
        ),
        GLP_ECOND => (
            MOI.NUMERICAL_ERROR,
            "Unable to start the search, because the basis matrix corresponding to the initial basis is ill-conditioned, i.e. its condition number is too large.",
        ),
        GLP_EBOUND => (
            MOI.INVALID_MODEL,
            "Unable to start the search, because some double-bounded (auxiliary or structural) variables have incorrect bounds.",
        ),
        GLP_EFAIL => (
            MOI.NUMERICAL_ERROR,
            "The search was prematurely terminated due to the solver failure.",
        ),
        GLP_EOBJLL => (
            MOI.OBJECTIVE_LIMIT,
            "The search was prematurely terminated, because the objective function being maximized has reached its lower limit and continues decreasing (the dual simplex only).",
        ),
        GLP_EOBJUL => (
            MOI.OBJECTIVE_LIMIT,
            "The search was prematurely terminated, because the objective function being minimized has reached its upper limit and continues increasing (the dual simplex only).",
        ),
        GLP_EITLIM => (
            MOI.ITERATION_LIMIT,
            "The search was prematurely terminated, because the simplex iteration limit has been exceeded.",
        ),
        GLP_ETMLIM => (
            MOI.TIME_LIMIT,
            "The search was prematurely terminated, because the time limit has been exceeded.",
        ),
        GLP_ENOPFS => (
            MOI.INFEASIBLE,
            "The LP problem instance has no primal feasible solution (only if the LP presolver is used).",
        ),
        GLP_ENODFS => (
            MOI.DUAL_INFEASIBLE,
            "The LP problem instance has no dual feasible solution (only if the LP presolver is used).",
        ),
    )

const _RAW_EXACT_STRINGS = Dict{Int32,Tuple{MOI.TerminationStatusCode,String}}(
    GLP_EBADB => (
        MOI.INVALID_MODEL,
        "Unable to start the search, because the initial basis specified in the problem object is invalid—the number of basic (auxiliary and structural) variables is not the same as the number of rows in the problem object.",
    ),
    GLP_ESING => (
        MOI.NUMERICAL_ERROR,
        "Unable to start the search, because the basis matrix corresponding to the initial basis is exactly singular.",
    ),
    GLP_EBOUND => (
        MOI.INVALID_MODEL,
        "Unable to start the search, because some double-bounded (auxiliary or structural) variables have incorrect bounds.",
    ),
    GLP_EFAIL =>
        (MOI.INVALID_MODEL, "The problem instance has no rows/columns."),
    GLP_EITLIM => (
        MOI.ITERATION_LIMIT,
        "The search was prematurely terminated, because the simplex iteration limit has been exceeded.",
    ),
    GLP_ETMLIM => (
        MOI.TIME_LIMIT,
        "The search was prematurely terminated, because the time limit has been exceeded.",
    ),
)

const _RAW_INTERIOR_STRINGS =
    Dict{Int32,Tuple{MOI.TerminationStatusCode,String}}(
        GLP_EFAIL =>
            (MOI.INVALID_MODEL, "The problem instance has no rows/columns."),
        GLP_ENOCVG =>
            (MOI.SLOW_PROGRESS, "Very slow convergence or divergence."),
        GLP_EITLIM => (MOI.ITERATION_LIMIT, "Iteration limit exceeded."),
        GLP_EINSTAB => (
            MOI.NUMERICAL_ERROR,
            "Numerical instability on solving Newtonian system.",
        ),
    )

const _RAW_INTOPT_STRINGS = Dict{Int32,Tuple{MOI.TerminationStatusCode,String}}(
    GLP_EBOUND => (
        MOI.INVALID_MODEL,
        "Unable to start the search, because some double-bounded (auxiliary or structural) variables have incorrect bounds.",
    ),
    GLP_ENOPFS => (
        MOI.INFEASIBLE,
        "Unable to start the search, because LP relaxation of the MIP problem instance has no primal feasible solution. (This code may appear only if the presolver is enabled.)",
    ),
    GLP_ENODFS => (
        MOI.DUAL_INFEASIBLE,
        "Unable to start the search, because LP relaxation of the MIP problem instance has no dual feasible solution. In other word, this code means that if the LP relaxation has at least one primal feasible solution, its optimal solution is unbounded, so if the MIP problem has at least one integer feasible solution, its (integer) optimal solution is also unbounded. (This code may appear only if the presolver is enabled.)",
    ),
    GLP_EFAIL => (
        MOI.INVALID_MODEL,
        "The search was prematurely terminated due to the solver failure.",
    ),
    GLP_EMIPGAP => (
        MOI.OPTIMAL,
        "The search was prematurely terminated, because the relative mip gap tolerance has been reached.",
    ),
    GLP_ETMLIM => (
        MOI.TIME_LIMIT,
        "The search was prematurely terminated, because the time limit has been exceeded.",
    ),
    GLP_ESTOP => (
        MOI.INTERRUPTED,
        "The search was prematurely terminated by application. (This code may appear only if the advanced solver interface is used.)",
    ),
)

const _RAW_SOLUTION_STATUS =
    Dict{Int32,Tuple{MOI.TerminationStatusCode,String}}(
        GLP_OPT => (MOI.OPTIMAL, "Solution is optimal"),
        GLP_FEAS => (MOI.LOCALLY_SOLVED, "Solution is feasible"),
        GLP_INFEAS => (MOI.LOCALLY_INFEASIBLE, "Solution is infeasible"),
        GLP_NOFEAS =>
            (MOI.INFEASIBLE, "No feasible primal-dual solution exists."),
        GLP_UNBND => (MOI.DUAL_INFEASIBLE, "Problem has unbounded solution"),
        GLP_UNDEF => (MOI.OTHER_ERROR, "Solution is undefined"),
    )

function MOI.get(model::Optimizer, attr::MOI.RawStatusString)
    _throw_if_optimize_in_progress(model, attr)
    if model.solver_status == Int32(0)
        (_, msg) = _get_status(model)
        return msg
    elseif model.last_solved_by_mip
        return _RAW_INTOPT_STRINGS[model.solver_status][2]
    elseif model.method == SIMPLEX
        return _RAW_SIMPLEX_STRINGS[model.solver_status][2]
    elseif model.method == EXACT
        return _RAW_EXACT_STRINGS[model.solver_status][2]
    else
        @assert model.method == INTERIOR
        return _RAW_INTERIOR_STRINGS[model.solver_status][2]
    end
end

function _get_status(model::Optimizer)
    status_code = if model.last_solved_by_mip
        glp_mip_status(model)
    elseif model.method == SIMPLEX || model.method == EXACT
        glp_get_status(model)
    else
        @assert model.method == INTERIOR
        glp_ipt_status(model)
    end
    return _RAW_SOLUTION_STATUS[status_code]
end

"""
    _certificates_potentially_available(model::Optimizer)

Return true if an infeasiblity certificate or an unbounded ray is potentially
available (i.e., the model has been solved using either the Simplex or Exact
methods).
"""
function _certificates_potentially_available(model::Optimizer)
    if !model.want_infeasibility_certificates
        return false
    elseif model.last_solved_by_mip
        return false
    else
        return model.method == SIMPLEX || model.method == EXACT
    end
end

function MOI.get(model::Optimizer, attr::MOI.TerminationStatus)
    _throw_if_optimize_in_progress(model, attr)
    if model.optimize_not_called
        return MOI.OPTIMIZE_NOT_CALLED
    elseif model.solver_status != Int32(0)
        # The solver did not exit successfully for some reason.
        if model.last_solved_by_mip
            return _RAW_INTOPT_STRINGS[model.solver_status][1]
        elseif model.method == SIMPLEX
            return _RAW_SIMPLEX_STRINGS[model.solver_status][1]
        elseif model.method == INTERIOR
            return _RAW_INTERIOR_STRINGS[model.solver_status][1]
        else
            @assert model.method == EXACT
            return _RAW_EXACT_STRINGS[model.solver_status][1]
        end
    else
        (status, _) = _get_status(model)
        return status
    end
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    _throw_if_optimize_in_progress(model, attr)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    (status, _) = _get_status(model)
    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        return MOI.FEASIBLE_POINT
    elseif status == MOI.LOCALLY_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    elseif status == MOI.DUAL_INFEASIBLE
        if model.unbounded_ray !== nothing
            return MOI.INFEASIBILITY_CERTIFICATE
        end
    else
        @assert status == MOI.INFEASIBLE || status == MOI.OTHER_ERROR
    end
    return MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    _throw_if_optimize_in_progress(model, attr)
    if attr.result_index != 1 || model.last_solved_by_mip
        return MOI.NO_SOLUTION
    end
    (status, _) = _get_status(model)
    if status == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif status == MOI.INFEASIBLE
        if model.infeasibility_cert !== nothing
            return MOI.INFEASIBILITY_CERTIFICATE
        end
    end
    return MOI.NO_SOLUTION
end

function _get_col_dual(model::Optimizer, column::Int)
    @assert !model.last_solved_by_mip
    if model.method == SIMPLEX || model.method == EXACT
        return _dual_multiplier(model) * glp_get_col_dual(model, column)
    else
        @assert model.method == INTERIOR
        return _dual_multiplier(model) * glp_ipt_col_dual(model, column)
    end
end

function _get_col_primal(model::Optimizer, column::Int)
    if model.last_solved_by_mip
        return glp_mip_col_val(model, column)
    elseif model.method == SIMPLEX || model.method == EXACT
        return glp_get_col_prim(model, column)
    else
        @assert model.method == INTERIOR
        return glp_ipt_col_prim(model, column)
    end
end

function _get_row_primal(model::Optimizer, row::Int)
    if model.last_solved_by_mip
        return glp_mip_row_val(model, row)
    elseif model.method == SIMPLEX || model.method == EXACT
        return glp_get_row_prim(model, row)
    else
        @assert model.method == INTERIOR
        return glp_ipt_row_prim(model, row)
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    x::MOI.VariableIndex,
)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    if model.unbounded_ray !== nothing
        return model.unbounded_ray[column(model, x)]
    else
        return _get_col_primal(model, column(model, x))
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.SingleVariable,<:Any},
)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    return MOI.get(model, MOI.VariablePrimal(), MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:Any},
)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    return _get_row_primal(model, _info(model, c).row)
end

function _dual_multiplier(model::Optimizer)
    return MOI.get(model, MOI.ObjectiveSense()) == MOI.MIN_SENSE ? 1.0 : -1.0
end

function _farkas_variable_dual(model::Optimizer, col::Int)
    nnz = glp_get_mat_col(model, col, C_NULL, C_NULL)
    vind = Vector{Cint}(undef, nnz)
    vval = Vector{Cdouble}(undef, nnz)
    nnz = glp_get_mat_col(model, Cint(col), offset(vind), offset(vval))
    return sum(
        model.infeasibility_cert[row] * val for (row, val) in zip(vind, vval)
    )
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    col = column(model, c)
    if model.infeasibility_cert !== nothing
        dual = _farkas_variable_dual(model, col)
        return min(dual, 0.0)
    end
    reduced_cost = if model.method == SIMPLEX || model.method == EXACT
        glp_get_col_dual(model, col)
    else
        @assert model.method == INTERIOR
        glp_ipt_col_dual(model, col)
    end
    sense = MOI.get(model, MOI.ObjectiveSense())
    # The following is a heuristic for determining whether the reduced cost
    # (i.e., the column dual) applies to the lower or upper bound. It can be
    # wrong by at most `tol_dj`.
    if sense == MOI.MIN_SENSE && reduced_cost < 0
        # If minimizing, the reduced cost must be negative (ignoring
        # tolerances).
        return reduced_cost
    elseif sense == MOI.MAX_SENSE && reduced_cost > 0
        # If minimizing, the reduced cost must be positive (ignoring
        # tolerances). However, because of the MOI dual convention, we return a
        # negative value.
        return -reduced_cost
    else
        # The reduced cost, if non-zero, must related to the lower bound.
        return 0.0
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    col = column(model, c)
    if model.infeasibility_cert !== nothing
        dual = _farkas_variable_dual(model, col)
        return max(dual, 0.0)
    end
    reduced_cost = if model.method == SIMPLEX || model.method == EXACT
        glp_get_col_dual(model, col)
    else
        @assert model.method == INTERIOR
        glp_ipt_col_dual(model, col)
    end
    sense = MOI.get(model, MOI.ObjectiveSense())
    # The following is a heuristic for determining whether the reduced cost
    # (i.e., the column dual) applies to the lower or upper bound. It can be
    # wrong by at most `tol_dj`.
    if sense == MOI.MIN_SENSE && reduced_cost > 0
        # If minimizing, the reduced cost must be negative (ignoring
        # tolerances).
        return reduced_cost
    elseif sense == MOI.MAX_SENSE && reduced_cost < 0
        # If minimizing, the reduced cost must be positive (ignoring
        # tolerances). However, because of the MOI dual convention, we return a
        # negative value.
        return -reduced_cost
    else
        # The reduced cost, if non-zero, must related to the lower bound.
        return 0.0
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.SingleVariable,S},
) where {S<:Union{MOI.EqualTo,MOI.Interval}}
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    col = column(model, c)
    if model.infeasibility_cert !== nothing
        return _farkas_variable_dual(model, col)
    end
    return _get_col_dual(model, col)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:Any},
)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    row = _info(model, c).row
    if model.infeasibility_cert !== nothing
        return -model.infeasibility_cert[row]
    else
        @assert !model.last_solved_by_mip
        if model.method == SIMPLEX || model.method == EXACT
            return _dual_multiplier(model) * glp_get_row_dual(model, row)
        else
            @assert model.method == INTERIOR
            return _dual_multiplier(model) * glp_ipt_row_dual(model, row)
        end
    end
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    if model.last_solved_by_mip
        return glp_mip_obj_val(model)
    elseif model.method == SIMPLEX || model.method == EXACT
        return glp_get_obj_val(model)
    else
        @assert model.method == INTERIOR
        return glp_ipt_obj_val(model)
    end
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveBound)
    _throw_if_optimize_in_progress(model, attr)
    if !model.last_solved_by_mip
        return MOI.get(model, MOI.ObjectiveSense()) == MOI.MIN_SENSE ? -Inf :
               Inf
    end
    # @mlubin and @ccoey observed some cases where mip_status == OPT and objval
    # and objbound didn't match. In that case, they return mip_obj_val, but
    # objbound may still be incorrect in cases where GLPK terminates early.
    if glp_mip_status(model) == GLP_OPT
        return glp_mip_obj_val(model)
    end
    return model.objective_bound
end

function MOI.get(model::Optimizer, attr::MOI.DualObjectiveValue)
    _throw_if_optimize_in_progress(model, attr)
    MOI.check_result_index_bounds(model, attr)
    return MOI.Utilities.get_fallback(model, attr, Float64)
end

function MOI.get(model::Optimizer, attr::MOI.RelativeGap)
    _throw_if_optimize_in_progress(model, attr)
    if !model.last_solved_by_mip
        error(
            "RelativeGap is only available for models with integer variables.",
        )
    end
    return model.relative_gap
end

function MOI.get(model::Optimizer, attr::MOI.SolveTimeSec)
    _throw_if_optimize_in_progress(model, attr)
    return model.solve_time
end

function MOI.get(model::Optimizer, attr::MOI.ResultCount)
    _throw_if_optimize_in_progress(model, attr)
    (status, _) = _get_status(model)
    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.LOCALLY_INFEASIBLE)
        return 1
    elseif status in
           (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE, MOI.LOCALLY_INFEASIBLE)
        if _certificates_potentially_available(model)
            return 1
        end
    end
    return 0
end

function MOI.get(model::Optimizer, ::MOI.Silent)
    return model.silent
end

function MOI.set(model::Optimizer, ::MOI.Silent, flag::Bool)
    model.silent = flag
    output_flag =
        flag ? GLP_OFF : MOI.get(model, MOI.RawOptimizerAttribute("msg_lev"))
    MOI.set(model, MOI.RawOptimizerAttribute("msg_lev"), output_flag)
    return
end

function MOI.get(model::Optimizer, ::MOI.Name)
    name = glp_get_prob_name(model)
    return name == C_NULL ? "" : unsafe_string(name)
end

function MOI.set(model::Optimizer, ::MOI.Name, name::String)
    glp_set_prob_name(model, name)
    return
end

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.variable_info)
function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return sort!(collect(keys(model.variable_info)), by = x -> x.value)
end

MOI.get(model::Optimizer, ::MOI.RawSolver) = model

function MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{F,S}) where {F,S}
    # TODO: this could be more efficient.
    return length(MOI.get(model, MOI.ListOfConstraintIndices{F,S}()))
end

_bound_enums(::Type{<:MOI.LessThan}) = (LESS_THAN, LESS_AND_GREATER_THAN)
_bound_enums(::Type{<:MOI.GreaterThan}) = (GREATER_THAN, LESS_AND_GREATER_THAN)
_bound_enums(::Type{<:MOI.Interval}) = (INTERVAL,)
_bound_enums(::Type{<:MOI.EqualTo}) = (EQUAL_TO,)
_bound_enums(::Any) = (nothing,)

_type_enums(::Type{MOI.ZeroOne}) = (BINARY,)
_type_enums(::Type{MOI.Integer}) = (INTEGER,)
_type_enums(::Any) = (nothing,)

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.SingleVariable,S},
) where {S}
    indices = MOI.ConstraintIndex{MOI.SingleVariable,S}[]
    for (key, info) in model.variable_info
        if info.bound in _bound_enums(S) || info.type in _type_enums(S)
            push!(indices, MOI.ConstraintIndex{MOI.SingleVariable,S}(key.value))
        end
    end
    return sort!(indices, by = x -> x.value)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64},S},
) where {S}
    indices = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S}[]
    for (key, info) in model.affine_constraint_info
        if typeof(info.set) == S
            push!(
                indices,
                MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},S}(
                    key.value,
                ),
            )
        end
    end
    return sort!(indices, by = x -> x.value)
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    constraints = Set{Tuple{DataType,DataType}}()
    for info in values(model.variable_info)
        if info.bound == NONE
        elseif info.bound == LESS_THAN
            push!(constraints, (MOI.SingleVariable, MOI.LessThan{Float64}))
        elseif info.bound == GREATER_THAN
            push!(constraints, (MOI.SingleVariable, MOI.GreaterThan{Float64}))
        elseif info.bound == LESS_AND_GREATER_THAN
            push!(constraints, (MOI.SingleVariable, MOI.LessThan{Float64}))
            push!(constraints, (MOI.SingleVariable, MOI.GreaterThan{Float64}))
        elseif info.bound == EQUAL_TO
            push!(constraints, (MOI.SingleVariable, MOI.EqualTo{Float64}))
        elseif info.bound == INTERVAL
            push!(constraints, (MOI.SingleVariable, MOI.Interval{Float64}))
        end
        if info.type == CONTINUOUS
        elseif info.type == BINARY
            push!(constraints, (MOI.SingleVariable, MOI.ZeroOne))
        elseif info.type == INTEGER
            push!(constraints, (MOI.SingleVariable, MOI.Integer))
        end
    end
    for info in values(model.affine_constraint_info)
        push!(
            constraints,
            (MOI.ScalarAffineFunction{Float64}, typeof(info.set)),
        )
    end
    return collect(constraints)
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType)
    return MOI.ScalarAffineFunction{Float64}
end

# TODO(odow): is there a way to modify a single element, rather than the whole
# row?
function MOI.modify(
    model::Optimizer,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:Any},
    chg::MOI.ScalarCoefficientChange{Float64},
)
    row = Cint(_info(model, c).row)
    col = column(model, chg.variable)
    nnz = glp_get_mat_row(model, row, C_NULL, C_NULL)
    indices, coefficients = zeros(Cint, nnz), zeros(Cdouble, nnz)
    glp_get_mat_row(model, row, offset(indices), offset(coefficients))
    index = something(findfirst(isequal(col), indices), 0)
    if index > 0
        coefficients[index] = chg.new_coefficient
    else
        push!(indices, col)
        push!(coefficients, chg.new_coefficient)
    end
    glp_set_mat_row(
        model,
        row,
        length(indices),
        offset(indices),
        offset(coefficients),
    )
    return
end

function MOI.modify(
    model::Optimizer,
    c::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    chg::MOI.ScalarCoefficientChange{Float64},
)
    glp_set_obj_coef(model, column(model, chg.variable), chg.new_coefficient)
    return
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:_SCALAR_SETS},
    f::MOI.ScalarAffineFunction{Float64},
)
    if !iszero(f.constant)
        throw(MOI.ScalarFunctionConstantNotZero(f.constant))
    end
    row = Cint(_info(model, c).row)
    indices, coefficients = _indices_and_coefficients(model, f)
    glp_set_mat_row(
        model,
        row,
        length(indices),
        offset(indices),
        offset(coefficients),
    )
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintBasisStatus,
    c::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},<:_SCALAR_SETS},
)
    _throw_if_optimize_in_progress(model, attr)
    row = _info(model, c).row
    cbasis = glp_get_row_stat(model, row)
    if cbasis == GLP_BS
        return MOI.BASIC
    elseif cbasis == GLP_NL ||
           cbasis == GLP_NU ||
           cbasis == GLP_NF ||
           cbasis == GLP_NS
        return MOI.NONBASIC
    else
        error("CBasis value of $(cbasis) isn't defined.")
    end
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariableBasisStatus,
    x::MOI.VariableIndex,
)
    _throw_if_optimize_in_progress(model, attr)
    col = column(model, x)
    vbasis = glp_get_col_stat(model, col)
    if vbasis == GLP_BS
        return MOI.BASIC
    elseif vbasis == GLP_NL
        return MOI.NONBASIC_AT_LOWER
    elseif vbasis == GLP_NU
        return MOI.NONBASIC_AT_UPPER
    elseif vbasis == GLP_NF
        return MOI.NONBASIC
    else
        @assert vbasis == GLP_NS
        return MOI.SUPERBASIC
    end
end
