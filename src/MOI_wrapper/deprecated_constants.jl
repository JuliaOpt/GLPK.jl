struct DeprecatedConstant{T}
    x::T
end

for sym in names(@__MODULE__, all = true)
    sym_string = string(sym)
    if !startswith(sym_string, "GLP_")
        continue
    end
    old_sym = Symbol(sym_string[5:end])
    @eval const $old_sym = DeprecatedConstant($sym)
end

function MOI.set(
    model::Optimizer,
    param::MOI.RawOptimizerAttribute,
    value::DeprecatedConstant,
)
    @warn(
        "The GLPK constants have been renamed from `GLPK.XXX` to " *
        "`GLPK.GLP_XXX` in order to better match the C API. For example, " *
        "`GLPK.MSG_OFF` is now `GLPK.GLP_MSG_OFF`. Support for the old " *
        "constants will be removed in a future release.",
        maxlog = 1
    )
    return MOI.set(model, param, value.x)
end
