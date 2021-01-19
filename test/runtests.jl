# MetidaCu
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>
using Metida
#using MetidaNLopt
using Test, CSV, DataFrames, StatsModels

df0 = CSV.File(dirname(pathof(Metida))*"\\..\\test\\csv\\df0.csv") |> DataFrame
transform!(df0, :subject => categorical, renamecols=false)
transform!(df0, :period => categorical, renamecols=false)
transform!(df0, :sequence => categorical, renamecols=false)
transform!(df0, :formulation=> categorical, renamecols=false)
@testset "  Basic test                                               " begin
    lmm = LMM(@formula(var~sequence+period+formulation), df0;
    random = VarEffect(@covstr(formulation), CSH),
    repeated = VarEffect(@covstr(formulation), Metida.DIAG),
    subject = :subject)
    fit!(lmm; solver = :cuda)
    @test lmm.result.reml       ≈ 10.065991599800707 atol=1E-6
end
