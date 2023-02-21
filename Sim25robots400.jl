push!(LOAD_PATH, ".")
using  Distributed

M  = 25; T = 1.0; H = 3; L = 30; MAX_ITER = 100
nP = M
nW = nworkers()
(nP+1 > nW) ? addprocs(nP-nW+1) : nothing 

import Pkg
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)

using Optim, Random, Distributions, CSV, DataFrames, MAT, JLD2
using Plots, Dates,  Statistics, Colors, ColorSchemes, StatsPlots

@everywhere begin
    using  SharedArrays, Ipopt, JuMP, GaussianProcesses, LinearAlgebra, Optim
end

@everywhere include("robots.jl")
include("computing.jl")
include("pxadmm.jl")
include("connectivity.jl")

ENV["GKSwstype"]="nul"

x_min =  0.; x_max = 400.
y_min =  0.; y_max = 400.
color =  cgrad(:turbo, M, categorical = true, scale = :lin)


## Load data
df   = CSV.read("SOM.csv", DataFrame, header = 0)
data = Matrix{Float64}(df)
loca_train    = Matrix(data[:,1:2]') 
obsr_train    = data[:,end]
inDim, numDa  = size(loca_train)

# Create a ground-truth model from the data
GPtruth       = GPE(loca_train, obsr_train, 
                    MeanConst(Statistics.mean(obsr_train)), 
                    SEArd(ones(inDim), 1/2*log(Statistics.var(obsr_train))),
                   -2.)
GaussianProcesses.optimize!(GPtruth, domean = true, kern = true, noise = true)
println([-log.(GPtruth.kernel.iℓ2)/2; log(GPtruth.kernel.σ2)/2; GPtruth.logNoise.value])


s_max = 20.; R = 80.; r = 6.
pBounds  = polyBound(s_max, x_min, x_max, y_min, y_max)


init    = init_position(pBounds, R, r, M)
robo    = [robot(i, T, H, R, r, 0., pBounds, init[:,i]) for i in 1:M]
NB      = find_nears(robo, M) 
mGP     = Vector{GPBase}(undef, M)

for i in 1:M
    robo[i].meas   = measure!(robo[i].posn, GPtruth)  
    robo[i].β      = Statistics.mean([robo[j].meas for j in [NB[i]; 1]])  
    mGP[i]         = ElasticGPE(inDim, mean = MeanConst(robo[i].β), kernel = SEArd(ones(inDim), 0.), logNoise = -2.)
end

testSize = [200, 200]
hor_gr   = range(x_min, stop = x_max, length = testSize[1])
ver_gr   = range(x_min, stop = y_max, length = testSize[2])
test     = meshgrid(hor_gr, ver_gr)

vectemp  = myGP_predict(GPtruth, test)[1] #Take only mean
temp     = reshape(vectemp, testSize[1], testSize[2])'
gr(size=(700,600))
Fig0 = heatmap(hor_gr, ver_gr,  temp, c = :turbo, aspect_ratio = 1, tickfontsize = 16, 
                                clim =(2.5, 6.5), label = "", xlims = (x_min,x_max), ylims = (y_min,y_max))
png(Fig0, "Fig-25robots-400/GroundTruth")





## Run simulation
println("Now start the simulation")
timer        = zeros(L)
Pred         = zeros(inDim, H, M)
J            = ones(M, L)
Eig2         = zeros(L)
RMSE         = zeros(M,L)
ResE         = ones(MAX_ITER, M, L)
[Pred[:,h,i] = robo[i].posn for h in 1:H, i in 1:M]

# random_move!(robo, mGP, GPtruth)

for k in 1:L
    println("Time instance $k")
    global Pred, J, ResE, NB

    # Train
    t0 = time_ns()
    dstbRetrain!(robo, mGP, NB, k)
    dt = (time_ns()-t0)/1e9
    println("Training time: $dt (s)")

    NB      = find_nears(robo, M)
    Eig2[k] = Index!(NB)
    pserSet = pserCon(robo, J[:, (k>1) ? k-1 : 1])

    Fig, RMSE[:,k] = myPlot(robo, mGP, vectemp, testSize, NB, color)
    png(Fig, "Fig-25robots-400/step $k"); #display(Fig)

    # Execute PxADMM
    t0 = time_ns()
    Pred, J[:,k], ResE[:,:,k] = dstbProxADMM!(robo, Pred, NB, pserSet; MAX_ITER = MAX_ITER)
    dt = (time_ns()-t0)/1e9
    println("Predicting time: $dt (s)")

    # Robots move to new locations and take measurement
    for i in 1:M
        robo[i].posn = Pred[:,1,i]
        robo[i].meas = measure!(robo[i].posn, GPtruth)
    end

    Distributed.clear!(CachingPool(workers()))
end

function nonzero(A::Vector{Float64}, value::Float64)
    for i in 1:length(A) if A[i] == value return i-1 end end
    return length(A)
end

gr(size=(1000,600))
FigRMSE = errorline(1:L, RMSE[:,1:L], linestyles = :solid, linewidth=2, secondarylinewidth=2, xlims = (0,L+0.5), errorstyle=:stick, 
secondarycolor=:blue,  legendfontsize = 16, tickfontsize = 20, framestyle = :box, label = "")
# errorline!(1:L, RMSE, linestyles = :solid, linewidth=2, xlims = (0,L+0.5), errorstyle=:ribbon, label="")
scatter!(1:L, [mean(RMSE[:,i]) for i in 1:L], label="Mean Errors")
png(FigRMSE, "Fig-25robots-400/RMSE")

pResE    = zeros(M,MAX_ITER)
[pResE[i,:] = sum(ResE[:,i,k] for k in 1:L)/L for i in 1:M]
id       = minimum([nonzero(pResE[i,:],0.) for i in 1:M]) - 7
FigpResE = errorline(1:id, pResE[:,1:id], secondarylinewidth=2, secondarycolor=:blue, errorstyle=:stick, framestyle = :box, yticks = 10 .^(-3.:1.:2.), label = "",
            legendfontsize = 16, tickfontsize = 20, xlims = (0, id-0.5), ylims = (1e-3, 2e2),  yscale=:log10, linestyles = :solid, linewidth=2)
scatter!(1:id, [mean(pResE[:,k]) for k in 1:id], label="Mean Errors")
png(FigpResE, "Fig-25robots-400/ResE")

                
png(plot(1:L, Eig2[1:L], linestyles = :dot, linewidth=3, xlims = (0,L+0.5), ylims = (0, 1.1*maximum(Eig2)), 
                tickfontsize = 20, markershape = :circle, markersize = 5, label="", framestyle = :box), "Fig-25robots-400/Eig2")

matwrite("Data25robo400.mat", Dict("RMSE" => RMSE, "ResE" => ResE, "Eig2" => Eig2))
save_object("25robots.jld2", robo)

