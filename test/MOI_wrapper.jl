module TestMOIWrapper

using GLPK
using Test

const MOI = GLPK.MathOptInterface

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

# TODO: also run the test for this model:
# MOI.Bridges.full_bridge_optimizer(
#     MOI.Utilities.CachingOptimizer(
#         MOI.Utilities.UniversalFallback(
#             MOI.Utilities.Model{Float64}(),
#         ),
#         GLPK.Optimizer(),
#     ),
#     Float64,
# )
function test_MOI_Test()
    MOI.Test.runtests(
        MOI.Bridges.full_bridge_optimizer(GLPK.Optimizer(), Float64),
        MOI.Test.Config();
        exclude = [
            # GLPK returns INVALID_MODEL instead of INFEASIBLE
            "test_constraint_ZeroOne_bounds_3",
            # Upstream issue: https://github.com/jump-dev/MathOptInterface.jl/issues/1398#issuecomment-872633881
            "test_constraint_ConstraintDualStart",
            "test_constraint_ConstraintPrimalStart",
            # Upstream issue: https://github.com/jump-dev/MathOptInterface.jl/issues/1431
            "test_model_LowerBoundAlreadySet",
            "test_model_UpperBoundAlreadySet",
        ]
    )
    return
end

function test_parameter_setting()
    solver = GLPK.Optimizer(tm_lim = 1, ord_alg = 2, alien = 3)
    @test solver.simplex_param.tm_lim == 1
    @test solver.intopt_param.tm_lim == 1
    @test solver.interior_param.ord_alg == 2
    @test solver.intopt_param.alien == 3
    return
end

function test_unbounded_integer_model()
    model = GLPK.Optimizer()
    MOI.Utilities.loadfromstring!(
        model,
        """
variables: x, y
minobjective: -5.0x + y
x in Integer()
x in LessThan(1.0)
""",
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    return
end

function test_infeasible_integer_model()
    model = GLPK.Optimizer()
    MOI.Utilities.loadfromstring!(
        model,
        """
variables: x
minobjective: -5.0x
x in Integer()
x in LessThan(1.0)
c3: 1.0x in GreaterThan(2.0)
""",
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    return
end

function test_issue_70()
    model = GLPK.Optimizer()
    x = MOI.add_variable(model)
    f = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0], [x]), 0.0)
    s = MOI.LessThan(2.0)
    c = MOI.add_constraint(model, f, s)
    row = GLPK._info(model, c).row
    @test GLPK.glp_get_row_type(model.inner, row) == GLPK.GLP_UP
    @test GLPK.glp_get_row_lb(model.inner, row) == -GLPK.GLP_DBL_MAX
    @test GLPK.glp_get_row_ub(model.inner, row) == 2.0
    # Modify the constraint set and verify that the internal constraint
    # has the correct bounds afterwards
    MOI.set(model, MOI.ConstraintSet(), c, MOI.LessThan(1.0))
    @test GLPK.glp_get_row_type(model.inner, row) == GLPK.GLP_UP
    @test GLPK.glp_get_row_lb(model.inner, row) == -GLPK.GLP_DBL_MAX
    @test GLPK.glp_get_row_ub(model.inner, row) == 1.0
    return
end

function test_infeasible_bounds()
    model = GLPK.Optimizer()
    x = MOI.add_variable(model)
    MOI.add_constraint(model, MOI.SingleVariable(x), MOI.Interval(1.0, -1.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INVALID_MODEL
    return
end

function test_RawOptimizerAttribute()
    model = GLPK.Optimizer(method = GLPK.SIMPLEX)
    exception = ErrorException(
        "Invalid option: cb_func. Use the MOI attribute `GLPK.CallbackFunction` instead.",
    )
    @test_throws exception MOI.set(
        model,
        MOI.RawOptimizerAttribute("cb_func"),
        (cb) -> nothing,
    )
    MOI.set(model, MOI.RawOptimizerAttribute("tm_lim"), 100)
    @test MOI.get(model, MOI.RawOptimizerAttribute("tm_lim")) == 100
    param = MOI.RawOptimizerAttribute("bad")
    @test_throws MOI.UnsupportedAttribute(param) MOI.set(model, param, 1)
    @test_throws MOI.UnsupportedAttribute(param) MOI.get(model, param)

    model = GLPK.Optimizer(method = GLPK.INTERIOR)
    exception = ErrorException(
        "Invalid option: cb_func. Use the MOI attribute `GLPK.CallbackFunction` instead.",
    )
    @test_throws exception MOI.set(
        model,
        MOI.RawOptimizerAttribute("cb_func"),
        (cb) -> nothing,
    )
    MOI.set(model, MOI.RawOptimizerAttribute("tm_lim"), 100)
    @test MOI.get(model, MOI.RawOptimizerAttribute("tm_lim")) == 100
    @test_throws MOI.UnsupportedAttribute(param) MOI.set(
        model,
        MOI.RawOptimizerAttribute("bad"),
        1,
    )
    @test_throws MOI.UnsupportedAttribute(param) MOI.get(
        model,
        MOI.RawOptimizerAttribute("bad"),
    )

    model = GLPK.Optimizer(method = GLPK.EXACT)
    exception = ErrorException(
        "Invalid option: cb_func. Use the MOI attribute `GLPK.CallbackFunction` instead.",
    )
    @test_throws exception MOI.set(
        model,
        MOI.RawOptimizerAttribute("cb_func"),
        (cb) -> nothing,
    )
    MOI.set(model, MOI.RawOptimizerAttribute("tm_lim"), 100)
    @test MOI.get(model, MOI.RawOptimizerAttribute("tm_lim")) == 100
    @test_throws MOI.UnsupportedAttribute(param) MOI.set(
        model,
        MOI.RawOptimizerAttribute("bad"),
        1,
    )
    @test_throws MOI.UnsupportedAttribute(param) MOI.get(
        model,
        MOI.RawOptimizerAttribute("bad"),
    )

    model = GLPK.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("mip_gap"), 0.001)
    @test MOI.get(model, MOI.RawOptimizerAttribute("mip_gap")) == 0.001
    return
end

function test_TimeLimitSec_issue_110()
    model = GLPK.Optimizer(method = GLPK.SIMPLEX)
    MOI.set(model, MOI.TimeLimitSec(), nothing)
    @test MOI.get(model, MOI.RawOptimizerAttribute("tm_lim")) == typemax(Int32)
    MOI.set(model, MOI.TimeLimitSec(), 100)
    @test MOI.get(model, MOI.RawOptimizerAttribute("tm_lim")) == 100000
    @test MOI.get(model, MOI.TimeLimitSec()) == 100
    # conversion between ms and sec
    MOI.set(model, MOI.RawOptimizerAttribute("tm_lim"), 100)
    @test isapprox(MOI.get(model, MOI.TimeLimitSec()), 0.1)
    return
end

function test_RelativeGap()
    model = GLPK.Optimizer()
    MOI.Utilities.loadfromstring!(
        model,
        """
    variables: x
    minobjective: 1.0x
    x in Integer()
    x in GreaterThan(1.5)
""",
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.RelativeGap()) == 0.0

    model = GLPK.Optimizer()
    MOI.Utilities.loadfromstring!(
        model,
        """
    variables: x
    minobjective: 1.0x
    x in GreaterThan(1.5)
""",
    )
    MOI.optimize!(model)
    @test_throws ErrorException MOI.get(model, MOI.RelativeGap())
    return
end

function test_issue_102()
    model = GLPK.Optimizer()
    x = MOI.add_variable(model)
    MOI.add_constraint(model, MOI.SingleVariable(x), MOI.GreaterThan(0.0))
    MOI.add_constraint(model, MOI.SingleVariable(x), MOI.Integer())
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 3.0),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) == 3.0
    @test MOI.get(model, MOI.ObjectiveBound()) == 3.0
    return
end

function test_issue_116()
    model = GLPK.Optimizer(method = GLPK.EXACT)
    x = MOI.add_variables(model, 2)
    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, x[1]), MOI.ScalarAffineTerm(1.0, x[2])],
            0.0,
        ),
        MOI.LessThan(1.0),
    )
    c2 = MOI.add_constraint(model, MOI.SingleVariable(x[1]), MOI.EqualTo(1.0))
    c3 = MOI.add_constraint(model, MOI.SingleVariable(x[2]), MOI.EqualTo(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(model, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
    cd1 = MOI.get(model, MOI.ConstraintDual(), c1)
    @test cd1 <= 1e-6
    @test MOI.get(model, MOI.ConstraintDual(), c2) ≈ -cd1 atol = 1e-6
    @test MOI.get(model, MOI.ConstraintDual(), c3) ≈ -cd1 atol = 1e-6
    return
end

function test_default_parameters()
    model = GLPK.Optimizer()
    @test MOI.get(model, MOI.RawOptimizerAttribute("msg_lev")) ==
          GLPK.GLP_MSG_ERR
    @test MOI.get(model, MOI.RawOptimizerAttribute("presolve")) == GLPK.GLP_OFF
    model = GLPK.Optimizer(msg_lev = GLPK.GLP_MSG_ALL, presolve = true)
    @test MOI.get(model, MOI.RawOptimizerAttribute("msg_lev")) ==
          GLPK.GLP_MSG_ALL
    @test MOI.get(model, MOI.RawOptimizerAttribute("presolve")) == GLPK.GLP_ON
    return
end

function test_duals_with_equal_bounds()
    model = GLPK.Optimizer()
    x = MOI.add_variable(model)
    xl = MOI.add_constraint(model, MOI.SingleVariable(x), MOI.GreaterThan(1.0))
    xu = MOI.add_constraint(model, MOI.SingleVariable(x), MOI.LessThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ConstraintDual(), xl) == 1.0
    @test MOI.get(model, MOI.ConstraintDual(), xu) == 0.0
    return
end

function test_pr_121()
    model = GLPK.Optimizer()
    ci = MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }(
        1,
    )
    @test_throws MOI.InvalidIndex(ci) MOI.get(model, MOI.ConstraintSet(), ci)
    @test_throws MOI.InvalidIndex(ci) MOI.get(
        model,
        MOI.ConstraintFunction(),
        ci,
    )
    @test_throws MOI.InvalidIndex(ci) MOI.delete(model, ci)
    return
end

function test_nonascii_names()
    model = GLPK.Optimizer()
    x = MOI.add_variable(model)
    MOI.set(model, MOI.VariableName(), x, "ω")
    @test MOI.get(model, MOI.VariableName(), x) == "ω"
    c = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
        MOI.GreaterThan(0.0),
    )
    MOI.set(model, MOI.ConstraintName(), c, "ω")
    @test MOI.get(model, MOI.ConstraintName(), c) == "ω"
    return
end

function test_deprecated_constants()
    model = GLPK.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("msg_lev"), GLPK.MSG_OFF)
    @test MOI.get(model, MOI.RawOptimizerAttribute("msg_lev")) ==
          GLPK.GLP_MSG_OFF
    return
end

function test_copy_to()
    dest = GLPK.Optimizer()
    src = MOI.Utilities.Model{Float64}()
    MOI.Utilities.loadfromstring!(
        src,
        """
    variables: a, b, c, d
    minobjective: a + b + c + d
    a >= 1.0
    b <= 2.0
    c == 3.0
    d in Interval(-4.0, 4.0)
    a in Integer()
    b in ZeroOne()
    c7: a + b >= -1.1
    c8: a + b <= 2.2
    c8: c + d == 2.2
""",
    )
    index_map = MOI.copy_to(dest, src; copy_names = true)
    @test length(index_map) == 13
    for (k, v) in index_map
        if v isa MOI.VariableIndex
            @test k == v
        elseif v isa MOI.ConstraintIndex{MOI.SingleVariable}
            @test k == v
        else
            # The order of the linear constraints may change. But they should be
            # ordered from 1 to 3.
            @test typeof(k) == typeof(v)
            @test 1 <= v.value <= 3
        end
    end
    v = MOI.get(dest, MOI.ListOfVariableIndices())
    @test length(v) == 4
    names = MOI.get.(dest, MOI.VariableName(), v)
    @test names == ["a", "b", "c", "d"]
    return
end

function test_want_infeasibility_certificates()
    model = GLPK.Optimizer(want_infeasibility_certificates = false)
    x = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, MOI.SingleVariable.(x), MOI.LessThan(0.0))
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-2.0, -1.0], x), 0.0),
        MOI.LessThan(-1.0),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_large_time_limits()
    model = GLPK.Optimizer()
    MOI.set(model, MOI.TimeLimitSec(), 1e9)
    @test MOI.get(model, MOI.TimeLimitSec()) == typemax(Cint) / 1_000
    return
end

function test_fractional_time_limits()
    model = GLPK.Optimizer()
    MOI.set(model, MOI.TimeLimitSec(), 1.2345)
    @test MOI.get(model, MOI.TimeLimitSec()) == 1.235
    return
end

function test_empty_problem()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer(),
    )
    MOI.Utilities.reset_optimizer(model)
    MOI.add_variable(model)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    return
end

function test_empty_problem_infeasible()
    model = GLPK.Optimizer()
    x = MOI.add_variable(model)
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(0.0, x)], 0.0),
        MOI.GreaterThan(1.0),
    )
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(0.0, x)], 0.0),
        MOI.EqualTo(0.0),
    )
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

function test_empty_problem_unbounded()
    model = GLPK.Optimizer()
    x = MOI.add_variable(model)
    MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(0.0, x)], 0.0),
        MOI.EqualTo(0.0),
    )
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    return
end

end  # module

TestMOIWrapper.runtests()
