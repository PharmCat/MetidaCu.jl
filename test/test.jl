# MetidaCu
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>
using Metida
using MetidaCu
using Test, CSV, DataFrames, StatsModels, CategoricalArrays

df0 = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","df0.csv"); types = [String, String, String, String, Float64, Float64]) |> DataFrame
transform!(df0, :subject => categorical, renamecols=false)
transform!(df0, :period => categorical, renamecols=false)
transform!(df0, :sequence => categorical, renamecols=false)
transform!(df0, :formulation=> categorical, renamecols=false)
@testset "  Basic test                                               " begin
    lmm = LMM(@formula(var~sequence+period+formulation), df0;
    random = VarEffect(@covstr(formulation|subject), CSH),
    repeated = VarEffect(@covstr(formulation|subject), Metida.DIAG),
    )
    fit!(lmm; solver = :cuda, rholinkf = :sigm, f_tol=1e-11, x_tol=1e-11)
    #10.065239006121315
    #10.065759784473597
    #10.06534259559994
    @test lmm.result.reml       ≈ 10.065238620495808 atol=1E-3
end
