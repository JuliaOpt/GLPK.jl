if get(ENV, "GITHUB_ACTIONS", "") == "true"
    import Pkg
    Pkg.add(Pkg.PackageSpec(name = "MathOptInterface", rev = "master"))
end

using GLPK
using Test

@testset "C API" begin
    @testset "glpk_tst_$(i).jl" for i in 1:6
        include("glpk_tst_$i.jl")
    end
end

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
    include("MOI_callbacks.jl")
end
