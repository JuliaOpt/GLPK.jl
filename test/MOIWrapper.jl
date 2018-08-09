using LinQuadOptInterface

const MOI  = LinQuadOptInterface.MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities

@testset "Unit Tests" begin
    config = MOIT.TestConfig()
    solver = GLPK.Optimizer()

    # MOIT.basic_constraint_tests(solver, config)

    MOIT.unittest(solver, config, [
        # These are excluded because GLPK does not support quadratics.
        "solve_qp_edge_cases",
        "solve_qcp_edge_cases"
    ])

    MOIT.modificationtest(solver, config, [
        # This is excluded because LQOI does not support setting the constraint
        # function.
        "solve_func_scalaraffine_lessthan"
    ])
end

@testset "Linear tests" begin
    solver = GLPK.Optimizer()
    MOIT.contlineartest(solver, MOIT.TestConfig(), [
        # GLPK returns InfeasibleOrUnbounded
        "linear8a",
        # Requires infeasiblity certificate for variable bounds
        "linear12"
    ])
end

@testset "Linear Conic tests" begin
    MOIT.lintest(GLPK.Optimizer(), MOIT.TestConfig(infeas_certificates=false))
end

@testset "Integer Linear tests" begin
    MOIT.intlineartest(GLPK.Optimizer(), MOIT.TestConfig(), [
        # int2 is excluded because SOS constraints are not supported.
        "int2"
    ])
end

@testset "ModelLike tests" begin
    solver = GLPK.Optimizer()
    @testset "nametest" begin
        MOIT.nametest(solver)
    end
    @testset "validtest" begin
        MOIT.validtest(solver)
    end
    @testset "emptytest" begin
        MOIT.emptytest(solver)
    end
    @testset "orderedindicestest" begin
        # MOIT.orderedindicestest(solver)
    end
    @testset "copytest" begin
        MOIT.copytest(solver, GLPK.Optimizer())
    end
end

@testset "Parameter setting" begin
    solver = GLPK.Optimizer(tm_lim=1, ord_alg=2, alien=3)
    @test solver.simplex.tm_lim == 1
    @test solver.intopt.tm_lim == 1
    @test solver.interior.ord_alg == 2
    @test solver.intopt.alien == 3
end

@testset "Callbacks" begin
    @testset "Lazy cut" begin
        model = GLPK.Optimizer()
        MOI.Utilities.loadfromstring!(model, """
            variables: x, y
            maxobjective: y
            c1: x in Integer()
            c2: y in Integer()
            c3: x in Interval(0.0, 2.0)
            c4: y in Interval(0.0, 2.0)
        """)
        x = MOI.get(model, MOI.VariableIndex, "x")
        y = MOI.get(model, MOI.VariableIndex, "y")

        # We now define our callback function that takes the callback handle.
        # Note that we can access model, x, and y because this function is
        # defined inside the same scope.
        cb_calls = Int32[]
        function callback_function(cb_data::GLPK.CallbackData)
            reason = GLPK.ios_reason(cb_data.tree)
            push!(cb_calls, reason)
            if reason == GLPK.IROWGEN
                GLPK.load_variable_primal!(cb_data)
                x_val = MOI.get(model, MOI.VariablePrimal(), x)
                y_val = MOI.get(model, MOI.VariablePrimal(), y)
                # We have two constraints, one cutting off the top
                # left corner and one cutting off the top right corner, e.g.
                # (0,2) +---+---+ (2,2)
                #       |xx/ \xx|
                #       |x/   \x|
                #       |/     \|
                # (0,1) +   +   + (2,1)
                #       |       |
                # (0,0) +---+---+ (2,0)
                TOL = 1e-6  # Allow for some impreciseness in the solution
                if y_val - x_val > 1 + TOL
                    GLPK.add_lazy_constraint!(cb_data,
                        MOI.ScalarAffineFunction{Float64}(
                            MOI.ScalarAffineTerm.([-1.0, 1.0], [x, y]),
                            0.0
                        ),
                        MOI.LessThan{Float64}(1.0)
                    )
                elseif y_val + x_val > 3 + TOL
                    GLPK.add_lazy_constraint!(cb_data,
                        MOI.ScalarAffineFunction{Float64}(
                            MOI.ScalarAffineTerm.([1.0, 1.0], [x, y]),
                            0.0
                        ),
                        MOI.LessThan{Float64}(3.0)
                    )
                end
            end
        end
        MOI.set!(model, GLPK.CallbackFunction(), callback_function)
        MOI.optimize!(model)
        @test MOI.get(model, MOI.VariablePrimal(), x) == 1
        @test MOI.get(model, MOI.VariablePrimal(), y) == 2
        @test length(cb_calls) > 0
        @test GLPK.ISELECT in cb_calls
        @test GLPK.IPREPRO in cb_calls
        @test GLPK.IROWGEN in cb_calls
        @test GLPK.IBINGO in cb_calls
        @test !(GLPK.IHEUR in cb_calls)
    end
end
