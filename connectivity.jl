function pserCon(robo::Vector{robot}, info::Vector{Float64})
    M = length(robo)
    nearB = find_nears(robo, M)
    pserR = Vector{Vector{Int64}}(undef, M)
    [pserR[i] = [] for i in 1:M]


    for i in 1:M
        for j in nearB[i]
            S = []; flag = true
            if checkConδ(robo[i], robo[j]) < 0
                pi = robo[i].posn
                pj = robo[j].posn
                # println("Note")
                for k in nearB[i] 
                    if k != j
                        if checkCon(robo[k], robo[j]) >= 0
                            if checkConδ(robo[i], robo[k]) < 0
                                S = [S; k]
                            else
                                pk = robo[k].posn
                                if dot(pk-pi, pk-pj) < 0
                                    flag = false #Ignore j
                                    break
                                end
                            end
                        end
                    end
                end


                if flag
                    if isempty(S)
                        pserR[i] = [pserR[i]; j] 
                    else
                        if info[j] <= findmin(info[S])[1]
                            pserR[i] = [pserR[i]; j]
                        end
                    end
                end
            end
        end
    end

    return pserR
end



function find_nears(robo::Vector{robot}, M::Int64)
    nearB     = Vector{Vector{Int64}}(undef, M)
    [nearB[i] = [] for i in 1:M]
    
    for i in 1:M
        for j in 1:M
            if j != i
                if checkCon(robo[i], robo[j]) > 0
                    nearB[i] = [nearB[i]; j]
                end
            end
        end 
    end

    return nearB
end



function checkCon(rb1::robot, rb2::robot)
    return min(rb1.R, rb2.R) - norm(rb1.posn - rb2.posn)
end

function checkConδ(rb1::robot, rb2::robot)
    return min(rb1.R, rb2.R) - norm(rb1.posn - rb2.posn) - rb1.pBnd.s_max - rb2.pBnd.s_max
end




function myPlot(robo::Vector{robot}, mGP::Vector{GPBase}, vecTruth::Vector{Float64}, testSize::Vector{Int64}, 
                                                            NB::Vector{Vector{Int64}}, color)
    M     = length(robo)
    x_min = robo[1].pBnd.x_min; x_max = robo[1].pBnd.x_max
    y_min = robo[1].pBnd.y_min; y_max = robo[1].pBnd.y_max

    hor_gr   = range(x_min, x_max, length = testSize[1])
    ver_gr   = range(y_min, y_max, length = testSize[2])
    test     = meshgrid(hor_gr, ver_gr)

    
    RMSE   = SharedArray(zeros(M))
    rch    = RemoteChannel(()->Channel{Vector{Float64}}(20))
    pmap(1:M) do i
        vecTemEst = myGP_predict(robo[i].mGP, test)[1]
        RMSE[i]   = sqrt((vecTruth-vecTemEst)'*(vecTruth-vecTemEst)/(length(vecTruth)))
        if i == 1 put!(rch, vecTemEst) end
    end

    vecTemEst = take!(rch)
    TemEst = reshape(vecTemEst, testSize[1], testSize[2])'
    gr(size=(700,600))
    Fig    = heatmap(hor_gr, ver_gr, TemEst, c = :turbo, xlims = (x_min,x_max), ylims = (y_min,y_max), aspect_ratio = 1, 
                                          tickfontsize = 16, clim = (2.5, 6.5))

    for i = 1:M
        plot!([robo[i].posn[1]], [robo[i].posn[2]], markershape = :circle, linewidth=3, markersize=7, markercolor=color[i], label="")
        for j = NB[i]
            Xp = [robo[i].posn[1], robo[j].posn[1]]
            Yp = [robo[i].posn[2], robo[j].posn[2]]
            plot!(Xp, Yp, label="")
        end
    end
    return Fig, Array(RMSE)
end


function Index!(NB::Vector{Vector{Int64}})
    M = length(NB)
    graph = zeros(M,M)
    for i in 1:M
        for j in NB[i]
            graph[i,j] = -1.
            graph[i,i] += 1.
        end
    end

    eig, orth = eigen(graph)
    return eig[2]
end


function init_position(pBnd::polyBound, R::Float64, r::Float64, M::Int64)
    x_min = pBnd.x_min
    x_max = pBnd.x_max
    y_min = pBnd.y_min
    y_max = pBnd.y_max
    s_max = pBnd.s_max
    while true
        s = [rand(Uniform(x_min+r, x_max-r), 1, M); rand(Uniform(y_min+r, y_max-r), 1, M)]
        if check_nears(s, R, r, M) return s end
    end
end


function check_nears(s::Matrix{Float64}, R::Float64, r::Float64, M::Int64)
    nearB     = Vector{Vector{Int64}}(undef, M)
    [nearB[i] = [] for i in 1:M]
    
    for i in 1:M
        for j in 1:M
            if j != i
                if (s[1,i] - s[1,j])^2 + (s[2,i] - s[2,j])^2 <= R^2
                    nearB[i] = [nearB[i]; j]
                end
                if (s[1,i] - s[1,j])^2 + (s[2,i] - s[2,j])^2 <= 4r^2
                    return false
                end
            end
        end 
    end

    for i in 1:M
        if length(nearB[i]) <= 1
            return false
        end
    end
    
    return Index!(nearB) > 1e-5
end

function random_move!(robo::Vector{robot}, mGP::Vector{GPBase}, GPtruth::GPBase)
    M     = length(robo)
    Dim   = length(robo[1].posn)
    s_max = robo[1].pBnd.s_max
    s     = zeros(Dim, M)
    while true
        ang   = rand(Uniform(-π, π), 1, M)
        for i in 1:M
            s[:,i] = robo[i].posn + s_max*[cos(ang[i]); sin(ang[i])]
        end
        if check_nears(s, robo[1].R, robo[1].r, M)
            NeiB = find_nears(robo, M)
            for i in 1:M # Exchange Previous Data
                loca = Matrix{Float64}(undef, Dim, 0)
                obsr = Vector{Float64}(undef, 0)
                for j in NeiB[i]
                    for r in 1:M
                        if robo[i].data[r,1][1]   == -1 && robo[j].data[r,1][1] != -1
                            robo[i].data[r,1][1:2] =   robo[j].data[r,1][1:2]
                            robo[i].data[r,1][3]   =   robo[j].data[r,1][3]
                            loca = [loca  robo[j].data[r,1][1:2]]
                            obsr = [obsr; robo[j].data[r,1][3]]
                        end
                    end
                end
                append!(mGP[i], loca, obsr)
                robo[i].loca = [robo[i].loca  loca]
                robo[i].obsr = [robo[i].obsr; obsr]
            end
        
            for i in 1:M # Exchange Current Data
                loca = Matrix{Float64}(undef, Dim, 0)
                obsr = Vector{Float64}(undef, 0)
                for j in [NeiB[i]; i]
                    robo[i].data[j,1][1:2] = robo[j].posn
                    robo[i].data[j,1][3]   = robo[j].meas
                    loca = [loca  robo[j].posn]
                    obsr = [obsr; robo[j].meas]
                end
                append!(mGP[i], loca, obsr)
                robo[i].loca = [robo[i].loca  loca]
                robo[i].obsr = [robo[i].obsr; obsr]
            end

            for i in 1:M # Move to next postions randomly
                robo[i].posn = s[:,i]
                robo[i].meas = measure!(s[:,i],GPtruth)
            end 
            return s 
        end
    end
end