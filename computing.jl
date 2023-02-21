function dstbRetrain!(robo::Vector{robot}, mGP::Vector{GPBase}, NeiB::Vector{Vector{Int64}}, ts::Int64)
    M    = length(robo)
    Dim  = length(robo[1].posn)

    # Hyperparameter
    for i in 1:M # Add previous data
        loca = Matrix{Float64}(undef, Dim, 0)
        obsr = Vector{Float64}(undef, 0)
        for j in NeiB[i]
            for r in 1:M
                for c in 1:ts
                    if  robo[i].data[r,c][1]   == -1 && robo[j].data[r,c][1] != -1
                        robo[i].data[r,c][1:2] =   robo[j].data[r,c][1:2]
                        robo[i].data[r,c][3]   =   robo[j].data[r,c][3]
                        loca = [loca  robo[j].data[r,c][1:2]]
                        obsr = [obsr; robo[j].data[r,c][3]]
                    end
                end
            end
        end
        append!(mGP[i], loca, obsr)
        robo[i].loca = [robo[i].loca  loca]
        robo[i].obsr = [robo[i].obsr; obsr]
    end

    for i in 1:M # Add current data
        loca = Matrix{Float64}(undef, Dim, 0)
        obsr = Vector{Float64}(undef, 0)
        for j in [NeiB[i]; i]
            robo[i].data[j,ts][1:2] = robo[j].posn
            robo[i].data[j,ts][3]   = robo[j].meas
            loca = [loca  robo[j].posn]
            obsr = [obsr; robo[j].meas]
        end
        append!(mGP[i], loca, obsr)
        robo[i].loca = [robo[i].loca  loca]
        robo[i].obsr = [robo[i].obsr; obsr]
    end

    rch = RemoteChannel(()->Channel{Tuple}(M+1))
    pmap(1:M) do i
        mGPtem = mGP[i]
        GaussianProcesses.optimize!(mGPtem, domean = true, kern = true, noise = true, noisebounds = [-3., -2.], 
                                            kernbounds = [[-5.,-5.,-5.],[5.,5.,5.]]; method = Optim.NelderMead())
        if mGPtem.logNoise.value > -2. && mGPtem.logNoise.value < -3.
            mGPtem.logNoise.value = -2.5
        end
        put!(rch, (i, mGPtem))
    end

    β   = zeros(M)
    ϕℓ2 = zeros(Dim, M)
    σκ2 = zeros(M)
    σω2 = zeros(M)
    for j in 1:M
        i, mGPtem = take!(rch)
        mGP[i]    = mGPtem
        β[i]      = mGPtem.mean.β
        ϕℓ2[:,i]  = 1 ./mGPtem.kernel.iℓ2
        σκ2[i]    = mGPtem.kernel.σ2
        σω2[i]    = exp(2*mGPtem.logNoise.value)
    end

    NBe = [[NeiB[i];i] for i in 1:M]
    # Consensus
    for k in 1:50
        for i in 1:M
            robo[i].β     = 1/length(NBe[i])*sum(β[i]     for i in NBe[i])
            robo[i].ϕℓ2   = 1/length(NBe[i])*sum(ϕℓ2[:,i] for i in NBe[i])
            robo[i].σκ2   = 1/length(NBe[i])*sum(σκ2[i]   for i in NBe[i])
            robo[i].σω2   = 1/length(NBe[i])*sum(σω2[i]   for i in NBe[i])
            β[i]          = robo[i].β
            ϕℓ2[:,i]      = robo[i].ϕℓ2
            σκ2[i]        = robo[i].σκ2
            σω2[i]        = robo[i].σω2
        end
    end
    println([log.(ϕℓ2[:,1])/2; log(σκ2[1])/2; log(σω2[1])/2])

    for i in 1:M
        robo[i].iΦ2    =  diagm(1 ./robo[i].ϕℓ2)
        robo[i].iCθ    =  inv(dstbSEkernel(robo[i], robo[i].loca, robo[i].loca) + robo[i].σω2*I(length(robo[i].obsr)))
        robo[i].mGP    =  GPE(robo[i].loca, robo[i].obsr, 
                                MeanConst(robo[i].β), 
                                SEArd(log.(robo[i].ϕℓ2)/2, log(robo[i].σκ2)/2),
                                log(robo[i].σω2)/2)
    end
    clear!(CachingPool(workers()))
end


@everywhere function myGP_predict(mGP::GPBase, p::Matrix{Float64}, full_cov = false)
    return predict_y(mGP, p, full_cov = full_cov)
end


@everywhere function dstbLinearize_logdet(robo::robot, z::Matrix{Float64})
    numDa       = length(robo.obsr)
    K_ZZ        = dstbSEkernel(robo, z, z)
    K_NN        = robo.iCθ
    K_NZ        = dstbSEkernel(robo, robo.loca, z)
    K_ZN        = Matrix(K_NZ')    
    iCθ         = inv(myGP_predict(robo.mGP, z, true)[2])

    Dim, H      = size(z)
    ∇L          = zeros(Dim, H)

    for r in 1:Dim
        for h in 1:H
            Ω1 = zeros(H, H)
            Ω2 = zeros(H, H)

            for k in 1:H
                Ω1[h,k] = Ω1[k,h] = -K_ZZ[h,k]/robo.ϕℓ2[r]*(z[r,h] - z[r,k])
            end

            dK_ZN = zeros(H, numDa)

            for k in 1:numDa
                dK_ZN[h,k] = -K_ZN[h,k]/robo.ϕℓ2[r]*(z[r,h] - robo.loca[r,k])
            end

            Ω2 = 2*dK_ZN*K_NN*K_NZ
            
            ∇L[r,h] = -tr(iCθ*(Ω1 - Ω2))
        end
    end
    return ∇L
end


@everywhere function dstbSEkernel(robo::robot, x::Matrix{Float64}, y::Matrix{Float64})
    row   = size(x)[2]
    col   = size(y)[2]
    K     = zeros(row, col)

    for i in 1:row
        for j in 1:col
            K[i,j] = robo.σκ2*exp(-0.5*dot(x[:,i] - y[:,j], robo.iΦ2, x[:,i] - y[:,j]))
        end
    end

    return K
end


"Take measurement"
function measure!(posn::Vector{Float64}, mGP::GPBase)
    return predict_y(mGP, posn[:,:])[1][1]
end

function meshgrid(x, y)
    # Function to make a meshgrid
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    X = reshape(X, 1, length(X))
    Y = reshape(Y, 1, length(Y))
    return [X; Y]
end


@everywhere function distTrainingGrad(robo::robot)
    iCθ = robo.iCθ
    σκ2 = robo.σκ2
    iΦ2 = robo.iΦ2
    inDim = 2

    numDa = length(robo[i].obsr)
    ∂iϕ2  = zeros(numDa,numDa,inDim)
    ∂σκ2  = zeros(numDa,numDa)
    ∂σω2  = zeros(numDa,numDa)

    for i in 1:numDa
        for j in 1:numDa
            x = robo.loca[:,i]
            y = robo.loca[:,j]

            for ℓ in 1:inDim
                ∂iϕ2[i,j,ℓ] = -(x[ℓ]-y[ℓ])^2*σκ2*exp(-0.5*dot(x-y, iΦ2, x-y))
            end

            ∂σκ2[i,j] = exp(-0.5*dot(x-y, iΦ2, x-y))

            if i == j
                ∂σω2[i,j] = 1. 
            end
        end
    end

    # See Gaussian Process in ML (5.9)
    a  = iCθ*robo.obsr
    A  = a*a' - iCθ
    g1 = 1/2*tr(A*∂iϕ2[:,:,1])
    g2 = 1/2*tr(A*∂iϕ2[:,:,2])
    g3 = 1/2*tr(A*∂σκ2)
    g4 = 1/2*tr(A*∂σω2)

    return [g1, g2, g3, g4]
end
