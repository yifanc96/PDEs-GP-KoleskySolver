# For linear algebra
using StaticArrays: SVector 
using LinearAlgebra
using SparseArrays: SparseMatrixCSC, sparse

# Fast Cholesky
using KoLesky 

# for logs
using Printf

# multiple dispatch
import IterativeSolvers: cg!, cg
import LinearAlgebra: mul!, ldiv!
import Base: size

using BenchmarkTools
using Profile

## PDEs type
abstract type AbstractPDEs end
struct NonlinElliptic2d{Tα,Tm,TΩ} <: AbstractPDEs
    # eqn: -Δu + α*u^m = f in [Ω[1,1],Ω[2,1]]*[Ω[1,2],Ω[2,2]]
    α::Tα
    m::Tm
    Ω::TΩ
    bdy::Function
    rhs::Function
end

## sample points
function sample_points_rdm(eqn::NonlinElliptic2d, N_domain, N_boundary)
    Ω = eqn.Ω
    x1l = Ω[1,1]
    x1r = Ω[2,1]
    x2l = Ω[1,2]
    x2r = Ω[2,2]   

    X_domain = hcat(rand(Float64,(N_domain, 1))*(x1r-x1l).+x1l,rand(Float64,(N_domain, 1))*(x2r-x2l).+x2l)

    N_bd_each=convert(Int64, N_boundary/4)
    if N_boundary != 4 * N_bd_each
        println("[sample points] N_boundary not divided by 4, replaced by ", 4 * N_bd_each)
        N_boundary = 4 * N_bd_each
    end

    X_boundary = zeros((N_boundary, 2))
    # bottom face
    X_boundary[1:N_bd_each, :] = hcat((x1r-x1l)*rand(Float64,(N_bd_each,1)).+x1l, x2l*ones(N_bd_each))
    # right face
    X_boundary[N_bd_each+1:2*N_bd_each, :] = hcat(x1r*ones(N_bd_each),(x2r-x2l)*rand(Float64,(N_bd_each,1)).+x2l)
    # top face
    X_boundary[2*N_bd_each+1:3*N_bd_each, :] = hcat((x1r-x1l)*rand(Float64,(N_bd_each,1)).+x1l, x2r*ones(N_bd_each))

    # left face
    X_boundary[3*N_bd_each+1:N_boundary, :] = hcat(x1l*ones(N_bd_each), (x2r-x2l)*rand(Float64,(N_bd_each,1)).+x2l)
    return X_domain', X_boundary'
end
function sample_points_grid(eqn::NonlinElliptic2d, h_in, h_bd)
    Ω = eqn.Ω
    x1l = Ω[1,1]
    x1r = Ω[2,1]
    x2l = Ω[1,2]
    x2r = Ω[2,2]
    x = x1l + h_in:h_in:x1r-h_in
    y = x2l + h_in:h_in:x2r-h_in
    X_domain = reduce(hcat,[[x[i], y[j]] for i in 1:length(x) for j in 1:length(x)])

    l = length(x1l:h_bd:x1r-h_bd)
    X_boundary = vcat([x1l:h_bd:x1r-h_bd x2l*ones(l)], [x1r*ones(l) x2l:h_bd:x2r-h_bd], [x1r:-h_bd:x1l+h_bd x2r*ones(l)], [x1l*ones(l) x2r:-h_bd:x1l+h_bd])
    return X_domain, X_boundary'
end

## exact algorithms
# assemby Gram matrices
function get_Gram_matrices(eqn::NonlinElliptic2d, cov::KoLesky.AbstractCovarianceFunction, X_domain, X_boundary, sol_now)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    Δδ_coefs = -1.0
    δ_coefs_int = eqn.α*eqn.m*(sol_now.^(eqn.m-1)) 

    # get linearized PDEs correponding measurements
    meas_δ = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_Δδ = [KoLesky.ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, δ_coefs_int[i]) for i = 1:N_domain]
    meas_test_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)

    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ; measurements[2] = meas_Δδ
    cov(Theta_train, reduce(vcat,measurements))
    
    Theta_test = zeros(N_domain,N_domain+N_boundary)
    cov(view(Theta_test,1:N_domain,1:N_boundary), meas_test_int, meas_δ)
    cov(view(Theta_test,1:N_domain,N_boundary+1:N_domain+N_boundary), meas_test_int, meas_Δδ)
    return Theta_train, Theta_test

end
# iterative GPR
function iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps)
    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]
    v = sol_init

    for _ in 1:GNsteps
        Theta_train, Theta_test = get_Gram_matrices(eqn, cov, X_domain, X_boundary, v)
        rhs_now = vcat(bdy, rhs+eqn.α*(eqn.m-1)*v.^eqn.m)
    
        v = Theta_test*(((Theta_train+nugget*diagm(diag(Theta_train))))\rhs_now)
    end
    return v
end 

## algorithm using Kolesky and pcg
# struct that stores the factor of Theta_train
abstract type implicit_mtx end

# struct approx_Theta_train{Tv,Ti} <: implicit_mtx
#     P::Vector{Ti}
#     U::SparseMatrixCSC{Tv,Ti}
#     δ_coefs::Vector{Tv}
#     N_boundary::Ti
#     N_domain::Ti
# end

struct approx_Theta_train{Tv,Ti,Tmtx<:SparseMatrixCSC{Tv,Ti}} <: implicit_mtx
    P::Vector{Ti}
    U::Tmtx
    L::Tmtx
    δ_coefs::Vector{Tv}
    N_boundary::Ti
    N_domain::Ti
end

# struct precond_Theta_train{Tv,Ti} <: implicit_mtx
#     P::Vector{Ti}
#     U::SparseMatrixCSC{Tv,Ti}
# end
struct precond_Theta_train{Tv,Ti,Tmtx<:SparseMatrixCSC{Tv,Ti}} <: implicit_mtx
    P::Vector{Ti}
    U::Tmtx
    L::Tmtx
end

function size(A::approx_Theta_train, num)
    return size(A.U,num)
end

function mul!(x, Θtrain::approx_Theta_train, b)
    temp = vcat(b[1:Θtrain.N_boundary],Θtrain.δ_coefs.*b[Θtrain.N_boundary+1:end],b[Θtrain.N_boundary+1:end])
    temp = Θtrain.L\(Θtrain.U\temp[Θtrain.P])
    temp[Θtrain.P] = temp

    @views x[1:Θtrain.N_boundary] = temp[1:Θtrain.N_boundary]
    @views x[Θtrain.N_boundary+1:end] .= Θtrain.δ_coefs.*temp[Θtrain.N_boundary+1:Θtrain.N_boundary+Θtrain.N_domain] .+ temp[Θtrain.N_boundary+Θtrain.N_domain+1:end]
end

function ldiv!(x, precond::precond_Theta_train, b)
    x[precond.P] = precond.U*(precond.L*b[precond.P])
end

function iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps; ρ_big =4.0, ρ_small=6.0, k_neighbors = 4, lambda = 1.5, alpha = 1.0)
    
    @printf "[Algorithm]: iterative GPR + fast cholesky factorization + pcg\n"
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]
    
    sol_now = sol_init

    # form the fast Cholesky part that can be used to compute mtx-vct mul for Theta_test
    d = 2
    Δδ_coefs = -1.0
    δ_coefs = 0.0
    meas_δ = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_δ_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]
    meas_Δδ = [KoLesky.ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, δ_coefs) for i = 1:N_domain]
    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,3)
    measurements[1] = meas_δ; measurements[2] = meas_δ_int
    measurements[3] = meas_Δδ

    @printf("[Big Theta: implicit factorization] time")
    @time implicit_bigΘ = KoLesky.ImplicitKLFactorization_FollowDiracs(cov, measurements, ρ_big, k_neighbors; lambda = lambda, alpha = alpha)

    # @printf("[Big Theta: implicit factorization] length of row indices")
    # @show length.(KoLesky.row_indices.(implicit_bigΘ.supernodes.supernodes))
    # @printf("[Big Theta: implicit factorization] length of column indices")
    # @show length.(KoLesky.column_indices.(implicit_bigΘ.supernodes.supernodes))

    @printf("[Big Theta: explicit factorization] time")
    @time explicit_bigΘ = KoLesky.ExplicitKLFactorization(implicit_bigΘ; nugget = nugget)
    U_bigΘ = explicit_bigΘ.U
    L_bigΘ = sparse(U_bigΘ')
    P_bigΘ = explicit_bigΘ.P

    Θtrain = approx_Theta_train(P_bigΘ, U_bigΘ, L_bigΘ,zeros(N_domain),N_boundary,N_domain)

    for step in 1:GNsteps
        
        @printf "[Current GN step] %d\n" step
        # get Cholesky of Theta_train
        Δδ_coefs = -1.0
        δ_coefs_int = eqn.α*eqn.m*sol_now.^(eqn.m-1)

        meas_Δδ = [KoLesky.ΔδPointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), Δδ_coefs, δ_coefs_int[i]) for i = 1:N_domain]
        measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
        measurements[1] = meas_δ; measurements[2] = meas_Δδ

        @printf("[Theta Train: implicit factorization] time")
        @time implicit_factor = KoLesky.ImplicitKLFactorization(cov, measurements, ρ_small, k_neighbors; lambda = lambda, alpha = alpha)

        # @printf("[Theta Train:implicit factorization] length of row indices")
        # @show length.(KoLesky.row_indices.(implicit_factor.supernodes.supernodes))
        # @printf("[Theta Train:implicit factorization] length of column indices")
        # @show length.(KoLesky.column_indices.(implicit_factor.supernodes.supernodes))

        @printf("[Theta Train: explicit factorization] time")
        @time explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor; nugget = nugget)

        U = explicit_factor.U
        L = sparse(U')
        P = explicit_factor.P
        rhs_now = vcat(bdy, rhs.+eqn.α*(eqn.m-1)*sol_now.^eqn.m)

        # use the approximate solution as the initial point for the pCG iteration
        sol = U*(L*rhs_now[P])
        sol[P] = sol
        # pcg step for Theta_train\rhs

        Θtrain.δ_coefs .= δ_coefs_int
        # Θtrain = approx_Theta_train(P_bigΘ, U_bigΘ, δ_coefs_int,N_boundary,N_domain)
        # x = similar(rhs_now)
        # @time mul!(x, Θtrain::approx_Theta_train, rhs_now)

        precond = precond_Theta_train(P,U,L)
        @printf "[pcg started]\n"
        @time sol, ch = cg!(sol, Θtrain, rhs_now; Pl = precond, log=true)

        println("[pcg finished]",ch)
        
        # get approximated sol_now = Theta_test * sol
        temp = vcat(sol[1:N_boundary], δ_coefs_int .* sol[N_boundary+1:end], sol[N_boundary+1:end])
        temp = L_bigΘ\(U_bigΘ\temp[P_bigΘ]) 
        temp[P_bigΘ] = temp
        @views sol_now = temp[N_boundary+1:N_domain+N_boundary] 
    end
    return sol_now
end


############# main ################
const α = 1.0;
const m = 3;
const Ω = [[0,1] [0,1]]
# ground truth solution
const freq = 20
const s = 3
function fun_u(x)
    ans = 0
    for k = 1:freq
        ans += sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
        # H^t norm squared is sum 1/k^{2s-2t}
    end
    return ans
end
# right hand side
function fun_rhs(x)
    ans = 0
    for k = 1:freq
        ans += (2*k^2*pi^2)*sin(pi*k*x[1])*sin(pi*k*x[2])/k^s 
        # H^t norm squared is sum 1/k^{2s-2t}
    end
    return ans + α*fun_u(x)^m
end

# boundary value
function fun_bdy(x)
    return fun_u(x)
end

@printf "[solver started] NonlinElliptic2d\n"
eqn = NonlinElliptic2d(α,m,Ω,fun_bdy,fun_rhs)

h_in = 0.005; h_bd = h_in
X_domain, X_boundary = sample_points_grid(eqn, h_in, h_bd)
# X_domain, X_boundary = sample_points_rdm(eqn, 900, 124)
N_domain = size(X_domain,2)
N_boundary = size(X_boundary,2)
@printf "[sample points] N_domain is %d, N_boundary is %d\n" N_domain N_boundary

lengthscale = 0.2
cov = KoLesky.MaternCovariance5_2(lengthscale)
# cov = KoLesky.GaussianCovariance(lengthscale)
@printf "[kernel] kernel is MaternCovariance5_2, lengthscale %.2f\n" lengthscale

nugget = 1e-14
@printf "[nugget] nugget term %e\n" nugget

GNsteps_approximate = 3
@printf "[total GN steps] %d\n" GNsteps_approximate

ρ_big = 3.0
ρ_small = 3.0

# ρ_big = 5.0
# ρ_small = 5.0
k_neighbors = 2
@printf "[Fast Cholesky] ρ_big = %.2f, ρ_small = %.2f, k_neighbors = %d\n" ρ_big ρ_small k_neighbors

sol_init = randn(N_domain) # initial solution
truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]
# @profview 
@time sol = iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps_approximate; ρ_big = ρ_big, ρ_small = ρ_small, k_neighbors=k_neighbors);
pts_accuracy = sqrt(sum((truth-sol).^2)/N_domain)
println("[L2 accuracy: pCG method]",pts_accuracy)
pts_max_accuracy = maximum(abs.(truth-sol))
println("[Linf accuracy: pCG method]",pts_max_accuracy)


# GNsteps_exact = 3
# @time sol_exact = iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, nugget, GNsteps_exact)
# pts_accuracy_exact = sqrt(sum((truth-sol_exact).^2)/N_domain)
# println("[L2 accuracy: exact method]", pts_accuracy_exact)
# pts_max_accuracy_exact = maximum(abs.(truth-sol_exact))
# println("[Linf accuracy: exact method]", pts_max_accuracy_exact)

# using PProf
# pprof(;webport=58699)


