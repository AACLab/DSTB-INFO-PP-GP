"Distributed version"
function dstbProxADMM!(robo::Vector{robot}, Pred::Array{Float64}, NB::Vector{Vector{Int64}}, pserSet::Vector{Vector{Int64}}; 
                                    MAX_ITER = 200, thres = 1e-2)
    Dim = length(robo[1].posn)
    M   = length(robo)
    H   = robo[1].H

    NBe = [[NB[i]; i] for i in 1:M]

    p   = SharedArray(Pred) # Initiate
    ξ   = SharedArray{Float64}(Dim, H, M, M)
    λ   = zeros(Dim, H, M, M)
    J   = SharedArray{Float64}(M)
    eps = zeros(MAX_ITER, M)
    ρ   = 1e-3
        
    for k in 1:MAX_ITER
        ρ = 1.2*ρ

        pmap(1:M) do i
            ξ[:,:,i,:], J[i] = dstbProjection!(robo[i], p, λ, NB[i], pserSet[i], Pred, ρ, M)
        end

        prox = 1e-2
        pmap(1:M) do i # Solve for p
            grad     = dstbLinearize_logdet(robo[i], p[:,:,i])
            p[:,:,i] = 1/(prox + ρ*length(NBe[i]))*(sum(ρ*ξ[:,:,j,i] - λ[:,:,i,j]  for j in NBe[i]) - grad + prox*ξ[:,:,i,i])
        end
        
        for i in 1:M # Update dual variable
            for j in NBe[i]
                λ[:,:,i,j] = λ[:,:,i,j] + ρ*(p[:,:,i] - ξ[:,:,j,i])
            end
        end
        
        for i in 1:M
            eps[k,i] = sum(norm(vec(p[:,:,i] - ξ[:,:,j,i])) for j in NBe[i])
        end

        ter = maximum(eps[k,:])
        if ter < thres
            println("Terminate at $k with error = $ter")
            break
        end
    end
    return Array(p), Array(J), eps
end


@everywhere function dstbProjection!(robo::robot, p::SharedArray{Float64, 3}, λ::Array{Float64}, NBi::Vector{Int64}, 
                            pserSeti::Vector{Int64}, Pred::Array{Float64}, ρ::Float64, M::Int64)
    Dim = length(robo.posn)

    # Variables
    H = robo.H
    i = robo.index
    r = robo.r

    @variable(robo.opti, ξ[1:Dim, 1:H, 1:M])

    # Objective function
    J = sum(dot(vec(ξ[:,:,j] - p[:,:,j] - 1/ρ*λ[:,:,j,i]), vec(ξ[:,:,j] - p[:,:,j] - 1/ρ*λ[:,:,j,i])) for j in [NBi; i])

    @objective(robo.opti, Min, J)

    for j in pserSeti # Connectivity preserving
        for h in 1:1
            @constraint(robo.opti, dot(ξ[:,h,i] - ξ[:,h,j], ξ[:,h,i] - ξ[:,h,j]) <= robo.R^2 - 2.0) # 2.0 for residual errors of ADMM
        end
    end

    # Collision avoidance
    for j in NBi
        for h in 1:1
            e = (Pred[:,h,i] - Pred[:,h,j])/norm(Pred[:,h,i] - Pred[:,h,j])
            @constraint(robo.opti, dot(ξ[:,h,i] - ξ[:,h,j], e) >= 2r)
        end
    end

    @constraints(robo.opti, begin robo.pBnd.x_min + r .<= ξ[1,:,i] .<= robo.pBnd.x_max - r
                                  robo.pBnd.y_min + r .<= ξ[2,:,i] .<= robo.pBnd.y_max - r
                        end)

    for h in 1:H
        if h == 1
            @constraint(robo.opti, (ξ[1,h,i]-robo.posn[1])^2 + (ξ[2,h,i]-robo.posn[2])^2 <= robo.pBnd.s_max^2)    
        else
            @constraint(robo.opti, (ξ[1,h,i]-ξ[1,h-1,i])^2 + (ξ[2,h,i]-ξ[2,h-1,i])^2 <= robo.pBnd.s_max^2)
        end
    end

    JuMP.optimize!(robo.opti) #Solve
    ξ = JuMP.value.(ξ)
    J = JuMP.value.(J)

    Base.empty!(robo.opti)
    return ξ, J
end



# @everywhere function distTraining(robo::Vector{robot}, NB::Vector{Vector{Float64}}; Nmax = 100)
#     θ = SharedArray{Float64}(4,M)
#     z = SharedArray(zeros(4,M))
#     M = length(robo)

#     for i in 1:M
#         θ[1:2,i] = 1 ./robo[i].ϕℓ2
#         θ[3,i]   = robo[i].σκ2
#         θ[4,i]   = robo[i].σω2
#     end

#     for k in 1:Nmax
#         pmap(1:M) do i
#             z[i]   = z[i] + rho*sum(θ[i] - θ[j] for j in NB[i])
#             θ[:,i] = distTrainProj(robo[i], θ, z[:,i], NB[i])
#         end

#         ter = maximum([sum(norm(θ[:,i] - θ[:,j]) for j in 1:M) for i in 1:M])
#         if ter < 1e-4
#             break
#         end
#     end
#     return θ
# end

# @everywhere function distTrainProj(robo::robot, θ::SharedArray{Float64}, z::SharedArray{Float64}, NB::Vector{Float64}; ρ = 1., γ = 1.)
#     opti = JuMP.Model(OSQP.Optimizer)
#     grad = distTrainingGrad(robot)
#     i = robo.index

#     @variable(opti, varθ[1:4])

#     J = dot(grad + z, varθ) + γ/2*dot(varθ - θ[:,i], varθ - θ[:,i]) 
#                             + ρ*sum(dot(varθ - 0.5*(θ[:,i] + θ[:,j]), varθ - 0.5*(θ[:,i] + θ[:,j])) for j in NB)

#     @constraints(opti, 1e-3 .<= varθ .<= [1. 1. 10. 10.])
#     JuMP.optimize!(opti)

#     return JuMP.value.(varθ)

# end