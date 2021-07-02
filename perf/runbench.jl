using SparseArrays
using LinearAlgebra
using Random
using GLPK
using MathOptInterface
# using ProfileView
const MOI = MathOptInterface

using TimerOutputs

struct RandomLP
    rows::Int
    cols::Int
    dens::Float64
end

function generate_moi_problem(model, At, b, c;
    var_bounds = true, scalar = true)
    cols, rows = size(At)
    x = MOI.add_variables(model, cols)
    A_cols = rowvals(At)
    A_vals = nonzeros(At)
    if var_bounds
        for col in 1:cols
            MOI.add_constraint(model, MOI.SingleVariable(x[col]),
                MOI.LessThan(10.0))
            MOI.add_constraint(model, MOI.SingleVariable(x[col]),
                MOI.GreaterThan(-10.0))
        end
    end
    if scalar
        for row in 1:rows
            MOI.add_constraint(model, MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(A_vals[i], x[A_cols[i]]) for i in nzrange(At, row)], 0.0),
                MOI.LessThan(b[row]))
        end
    else
        for row in 1:rows
            MOI.add_constraint(model, MOI.VectorAffineFunction(
                [MOI.VectorAffineTerm(1,
                    MOI.ScalarAffineTerm(A_vals[i], x[A_cols[i]])
                ) for i in nzrange(At, row)], [-b[row]]),
                MOI.Nonpositives(1))
        end
    end
    objective = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(c[i], x[i]) for i in findall(!iszero, c)],
            0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objective)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return x
end

function random_data(seed, data)

    rows = data.rows
    cols = data.cols
    density = data.dens

    p_neg_element = 0.0 # 0.25

    rng = Random.MersenneTwister(seed)

    f_A(r, n) = ifelse.(rand(r, n) .> p_neg_element, 1, -1) .* (15 .+ 30 .* rand(r, n))

    At = sprand(rng, cols, rows, density, f_A)
    b = 50 * rand(rng, rows)

    # not using signs now
    sign = ifelse.(rand(rng, rows) .> 0.2, 'L', 'G')

    f_c(r, n) = 20 .* 2 .* (rand(r, n) .- 0.5)
    c = sprand(rng, cols, 0.5, f_c)

    return At, b, c
end

function bridged_cache_and_solver()
    model = MOI.Bridges.full_bridge_optimizer(MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Utilities.MANUAL), Float64)
    GLPK_ = GLPK.Optimizer()
    MOI.set(GLPK_, MOI.Silent(), true)
    return model, GLPK_
end
function cache_and_solver()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Utilities.MANUAL)
    GLPK_ = GLPK.Optimizer()
    MOI.set(GLPK_, MOI.Silent(), true)
    return model, GLPK_
end
function bridged_cached_solver()
    model = MOI.Bridges.full_bridge_optimizer(MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer()), Float64)
    MOI.set(model, MOI.Silent(), true)
    return model
end
function cached_solver()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        GLPK.Optimizer())
    MOI.set(model, MOI.Silent(), true)
    return model
end

function time_build_and_solve(to_build, to_solve, At, b, c, scalar = true)
    @timeit "build" x = generate_moi_problem(to_build, At, b, c, scalar = scalar)
    if to_build !== to_solve
        @timeit "copy" MOI.copy_to(to_solve, to_build, copy_names = false)
    end
    MOI.set(to_solve, MOI.TimeLimitSec(), 0.0010)
    @time @timeit "opt" MOI.optimize!(to_solve)
    val = MOI.get(to_solve, MOI.SolveTimeSec())
    println(val)
    @show MOI.get(to_solve, MOI.ObjectiveValue())
    @show MOI.get(to_solve, MOI.TerminationStatus())
end

function solve_GLPK(seed, data; time_limit_sec=Inf)

    reset_timer!()

    @timeit "data"  At, b, c = random_data(1, data)
    for i in 1:seed
        # mod(i,5) == 0 && GC.gc()
        GC.gc()
        bridged_cache, pure_solver = bridged_cache_and_solver()
        @timeit "bc + s" time_build_and_solve(bridged_cache, pure_solver, At, b, c)

        GC.gc()
        cache, pure_solver2 = cache_and_solver()
        @timeit "c + s" time_build_and_solve(cache, pure_solver2, At, b, c)

        GC.gc()
        full_solver = bridged_cached_solver()
        @timeit "bcs" time_build_and_solve(full_solver, full_solver, At, b, c)

        GC.gc()
        full_solver = bridged_cached_solver()
        @timeit "bcs + v" time_build_and_solve(full_solver, full_solver, At, b, c, false)

        GC.gc()
        cache_solver = cached_solver()
        @timeit "cs" time_build_and_solve(cache_solver, cache_solver, At, b, c)

    end

    print_timer()

end

solve_GLPK(2, RandomLP(11, 11, 0.5); time_limit_sec=5)
solve_GLPK(20, RandomLP(10000, 10000, 0.005); time_limit_sec=5)
