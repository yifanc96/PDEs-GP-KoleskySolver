# using Pkg
# Pkg.activate(@__DIR__)

# For linear algebra
using StaticArrays: SVector
using StaticArrays
using LinearAlgebra
using SparseArrays: SparseMatrixCSC, sparse
# Fast Cholesky
using KoLesky 

# multiple dispatch
import IterativeSolvers: cg!
import LinearAlgebra: mul!, ldiv!
import Base: size

# autoDiff
using ForwardDiff

# parser
using ArgParse

# logging
using Logging

# profile
# using Profile
# using BenchmarkTools

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--kernel"
            arg_type = String
            default = "Matern7half"
        "--sigma"
            help = "lengthscale"
            arg_type = Float64
            default = 0.3
        "--h"
            help = "grid size"
            arg_type = Float64
            default = 0.0125
        "--nugget"
            arg_type = Float64
            default = 1e-8
        "--GNsteps"
            arg_type = Int
            default = 3
        "--rho_big"
            arg_type = Float64
            default = 3.0
        "--rho_small"
            arg_type = Float64
            # default = -log(5*h_grid)
            default = 3.0
        "--k_neighbors"
            arg_type = Int
            default = 3
        "--compare_exact"
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end

## PDEs type
abstract type AbstractPDEs end
struct MongeAmpere2d{TΩ} <: AbstractPDEs
    # minimal surface equation in [Ω[1,1],Ω[2,1]]*[Ω[1,2],Ω[2,2]]
    Ω::TΩ
    bdy::Function
    rhs::Function
end

## sample points
function sample_points_rdm(eqn::MongeAmpere2d, N_domain, N_boundary)
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
function sample_points_grid(eqn::MongeAmpere2d, h_in, h_bd)
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
function get_Gram_matrices(eqn::MongeAmpere2d, cov::KoLesky.AbstractCovarianceFunction, X_domain, X_boundary, sol_xx_now, sol_xy_now,sol_yy_now)
    d = 2
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    # get linearized PDEs correponding measurements
    meas_δ_bd = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_∂∂ = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), sol_yy_now[i], -2*sol_xy_now[i], sol_xx_now[i]) for i = 1:N_domain]
    meas_δ_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)

    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ_bd; measurements[2] = meas_∂∂
    cov(Theta_train, reduce(vcat,measurements))
    
    measurement_test = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,4)
    measurement_test[1] = meas_δ_int; 
    measurement_test[2] = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 1.0, 0.0, 0.0) for i = 1:N_domain]
    measurement_test[3] = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, 1.0, 0.0) for i = 1:N_domain]
    measurement_test[4] = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, 0.0, 1.0) for i = 1:N_domain]


    Theta_test = zeros(4*N_domain,N_domain+N_boundary)
    cov(Theta_test, reduce(vcat,measurement_test), reduce(vcat,measurements))

    return Theta_train, Theta_test

end
# iterative GPR
function iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, sol_init_xx, sol_init_xy, sol_init_yy, nugget, GNsteps)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]
    v = sol_init
    v_xx = sol_init_xx
    v_xy = sol_init_xy
    v_yy = sol_init_yy

    for _ in 1:GNsteps
        Theta_train, Theta_test = get_Gram_matrices(eqn, cov, X_domain, X_boundary, v_xx, v_xy, v_yy)
        rhs_now = vcat(bdy, rhs+v_xx.*v_yy-v_xy.^2)
    
        v_all = Theta_test*(((Theta_train+nugget*diagm(diag(Theta_train))))\rhs_now)
        v = v_all[1:N_domain]
        
        v_xx = v_all[N_domain+1:2*N_domain]
        v_xy = v_all[2*N_domain+1:3*N_domain]
        v_yy = v_all[3*N_domain+1:4*N_domain]

        # in case that uxx or uyy becomes negative
        # eps = 1e-3
        # @inbounds for i in 1:N_domain
        #     if v_xx[i] < eps
        #         v_xx[i] = eps
        #     end
        #     if v_yy[i] < eps
        #         v_yy[i] = eps
        #     end
        #     if v_xy[i]^2 > v_xx[i]*v_yy[i]
        #         v_xy[i] = 0.5*sqrt(v_xx[i]*v_yy[i])*sign(v_xy[i])
        #     end
        # end
        
    end
    return v, v_xx, v_xy, v_yy
end 

## algorithm using Kolesky and pcg
# struct that stores the factor of Theta_train
abstract type implicit_mtx end
struct approx_Theta_train{Tv,Ti,Tmtx<:SparseMatrixCSC{Tv,Ti}} <: implicit_mtx
    P::Vector{Ti}
    U::Tmtx
    L::Tmtx
    ∂11_coefs::Vector{Tv}
    ∂12_coefs::Vector{Tv}
    ∂22_coefs::Vector{Tv}
    N_boundary::Ti
    N_domain::Ti
end
struct precond_Theta_train{Tv,Ti,Tmtx<:SparseMatrixCSC{Tv,Ti}} <: implicit_mtx
    P::Vector{Ti}
    U::Tmtx
    L::Tmtx
end

function size(A::approx_Theta_train, num)
    return size(A.U,num)
end

function mul!(x, Θtrain::approx_Theta_train, b)
    N_domain = Θtrain.N_domain
    N_boundary = Θtrain.N_boundary
    @views temp = vcat(b[1:N_boundary], zeros(N_domain), Θtrain.∂11_coefs .* b[N_boundary+1:end], Θtrain.∂22_coefs .*b[N_boundary+1:end], -2*Θtrain.∂12_coefs.*b[N_boundary+1:end])
    temp[Θtrain.P] = Θtrain.L\(Θtrain.U\temp[Θtrain.P])

    @views x[1:Θtrain.N_boundary] = temp[1:Θtrain.N_boundary]
    @views x[Θtrain.N_boundary+1:end] .= Θtrain.∂11_coefs.* temp[N_boundary+N_domain+1:N_boundary+2*N_domain] .+ Θtrain.∂22_coefs.*temp[N_boundary+2*N_domain+1:N_boundary+3*N_domain] .- 2*Θtrain.∂12_coefs.*temp[N_boundary+3*N_domain+1:N_boundary+4*N_domain]
end

function ldiv!(x, precond::precond_Theta_train, b)
    x[precond.P] = precond.U*(precond.L*b[precond.P])
end

function iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, sol_init, sol_init_xx, sol_init_xy, sol_init_yy, nugget, GNsteps; ρ_big =4.0, ρ_small=6.0, k_neighbors = 4, lambda = 1.5, alpha = 1.0)
    
    @info "[Algorithm]: iterative GPR + fast cholesky factorization + pcg"
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)

    # get the rhs and bdy data
    rhs = [eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
    bdy = [eqn.bdy(X_boundary[:,i]) for i in 1:N_boundary]
    
    v = sol_init
    v_xx = sol_init_xx
    v_xy = sol_init_xy
    v_yy = sol_init_yy

    # form the fast Cholesky part that can be used to compute mtx-vct mul for Theta_test
    d = 2
    meas_δ_bd = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_boundary[:,i])) for i = 1:N_boundary]
    meas_δ_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}(X_domain[:,i])) for i = 1:N_domain]
    meas_∂11_int = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 1.0, 0.0, 0.0) for i = 1:N_domain]
    meas_∂22_int = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, 0.0, 1.0) for i = 1:N_domain]
    meas_∂12_int = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), 0.0, 1.0, 0.0) for i = 1:N_domain]

    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,5)
    measurements[1] = meas_δ_bd
    measurements[2] = meas_δ_int
    measurements[3] = meas_∂11_int
    measurements[4] = meas_∂22_int
    measurements[5] = meas_∂12_int

    @info "[Big Theta: implicit factorization] time"
    @time implicit_bigΘ = KoLesky.ImplicitKLFactorization_DiracsFirstThenUnifScale(cov, measurements, ρ_big, k_neighbors; lambda = lambda, alpha = alpha)

    @info "[Big Theta: explicit factorization] time"
    @time explicit_bigΘ = KoLesky.ExplicitKLFactorization(implicit_bigΘ; nugget = nugget)
    U_bigΘ = explicit_bigΘ.U
    L_bigΘ = sparse(U_bigΘ')
    P_bigΘ = explicit_bigΘ.P

    Θtrain = approx_Theta_train(P_bigΘ, U_bigΘ, L_bigΘ,zeros(N_domain),zeros(N_domain),zeros(N_domain),N_boundary,N_domain)

    implicit_factor = nothing

    for step in 1:GNsteps
        
        @info "[Current GN step] $step"
        # get Cholesky of Theta_train
        
        meas_∂∂ = [KoLesky.∂∂PointMeasurement{Float64,d}(SVector{d,Float64}(X_domain[:,i]), v_yy[i], -2*v_xy[i], v_xx[i]) for i = 1:N_domain]

        measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
        measurements[1] = meas_δ_bd; measurements[2] = meas_∂∂

        @info "[Theta Train: implicit factorization] time"
        @time if implicit_factor === nothing
            implicit_factor = KoLesky.ImplicitKLFactorization(cov, measurements, ρ_small, k_neighbors; lambda = lambda, alpha = alpha)
        else
            implicit_factor.supernodes.measurements .= reduce(vcat, collect.(measurements))[implicit_factor.P]
        end
        @info "[Theta Train: explicit factorization] time"
        @time explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor; nugget = nugget)

        U = explicit_factor.U
        L = sparse(U')
        P = explicit_factor.P
        rhs_now = vcat(bdy, rhs+v_xx.*v_yy-v_xy.^2)

        # use the approximate solution as the initial point for the pCG iteration
        Θinv_rhs = U*(L*rhs_now[P])
        Θinv_rhs[P] = Θinv_rhs

        # pcg step for Theta_train\rhs
        Θtrain.∂11_coefs .= v_yy
        Θtrain.∂12_coefs .= v_xy
        Θtrain.∂22_coefs .= v_xx
        precond = precond_Theta_train(P,U,L)
        
        @info "[pcg started]"
        @time Θinv_rhs, ch = cg!(Θinv_rhs, Θtrain, rhs_now; Pl = precond, log=true)
        @info "[pcg finished], $ch"
        
        # get approximated sol_now = Theta_test * Θinv_rhs
        tmp = vcat(Θinv_rhs[1:N_boundary], zeros(N_domain), v_yy .* Θinv_rhs[N_boundary+1:end], v_xx .*Θinv_rhs[N_boundary+1:end], -2*v_xy.*Θinv_rhs[N_boundary+1:end])
        tmp[P_bigΘ] = L_bigΘ\(U_bigΘ\tmp[P_bigΘ]) 
        @views v = tmp[N_boundary+1:N_domain+N_boundary] 
        @views v_xx = tmp[N_domain+N_boundary+1:N_boundary+2*N_domain]
        @views v_yy = tmp[2*N_domain+N_boundary+1:N_boundary+3*N_domain]
        @views v_xy = tmp[3*N_domain+N_boundary+1:N_boundary+4*N_domain]
    end
    @info "[solver finished]"
    return v
end


function main(args)
    Ω = [[0,1] [0,1]]
    # ground truth solution
    function fun_u(x)
        return exp(0.5*sum((x.-0.5).^2))
    end

    function H_u(x)
        hessian = ForwardDiff.hessian(x -> fun_u(x),x)
        return @SVector [hessian[1],hessian[2],hessian[4]]
    end


    # right hand side
    function fun_rhs(x)
        H = H_u(x)
        return H[1]*H[3]-H[2]^2
    end

    # function fun_rhs(x)
    #     return (1+sum((x.-0.5).^2))*exp(sum((x.-0.5).^2))
    # end

    # boundary value
    function fun_bdy(x)
        return fun_u(x)
    end

    @info "[solver started] Monge-Ampere equation"
    @info "[equation] Monge-Ampere u_xxu_yy - (u_xy)² = f"
    eqn = MongeAmpere2d(Ω,fun_bdy,fun_rhs)
    
    h_in = args.h::Float64; h_bd = h_in
    X_domain, X_boundary = sample_points_grid(eqn, h_in, h_bd)
    # X_domain, X_boundary = sample_points_rdm(eqn, 900, 124)
    N_domain = size(X_domain,2)
    N_boundary = size(X_boundary,2)
    @info "[sample points] grid size $h_in"
    @info "[sample points] N_domain is $N_domain, N_boundary is $N_boundary"  

    lengthscale = args.sigma
    if args.kernel == "Matern5half"
        cov = KoLesky.MaternCovariance5_2(lengthscale)
    elseif args.kernel == "Matern7half"
        cov = KoLesky.MaternCovariance7_2(lengthscale)
    elseif args.kernel == "Matern9half"
        cov = KoLesky.MaternCovariance9_2(lengthscale)
    elseif args.kernel == "Matern11half"
        cov = KoLesky.MaternCovariance11_2(lengthscale)
    elseif args.kernel == "Gaussian"
        cov = KoLesky.GaussianCovariance(lengthscale)
    end
    @info "[kernel] choose $(args.kernel), lengthscale $lengthscale\n"  

    nugget = args.nugget::Float64
    @info "[nugget] $nugget" 

    GNsteps_approximate = args.GNsteps::Int
    @info "[total GN steps] $GNsteps_approximate" 

    ρ_big = args.rho_big::Float64
    ρ_small = args.rho_small::Float64
    k_neighbors = args.k_neighbors::Int
    @info "[Fast Cholesky] ρ_big = $ρ_big, ρ_small = $ρ_small, k_neighbors = $k_neighbors"

    # sol_init = [fun_u(X_domain[:,i]) for i in 1:N_domain] # initial solution
    # sol_init_derivatives = [H_u(X_domain[:,i]) for i in 1:N_domain]    
    # sol_init_xx = [sol_init_derivatives[i][1] for i in 1:N_domain]
    # sol_init_xy = [sol_init_derivatives[i][2] for i in 1:N_domain]
    # sol_init_yy = [sol_init_derivatives[i][3] for i in 1:N_domain]

    sol_init = [0.0 for i in 1:N_domain] # initial solution  
    sol_init_xx = [1.0 for i in 1:N_domain]
    sol_init_xy = [0.0 for i in 1:N_domain]
    sol_init_yy = [1.0 for i in 1:N_domain]

    truth = [fun_u(X_domain[:,i]) for i in 1:N_domain]

    fast_solve() = @time iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, sol_init, sol_init_xx, sol_init_xy, sol_init_yy, nugget, GNsteps_approximate; ρ_big = ρ_big, ρ_small = ρ_small, k_neighbors=k_neighbors);
    sol = fast_solve()

    pts_accuracy = sqrt(sum((truth-sol).^2)/N_domain)
    @info "[L2 accuracy: pCG method] $pts_accuracy"
    pts_max_accuracy = maximum(abs.(truth-sol))
    @info "[Linf accuracy: pCG method] $pts_max_accuracy"

    # @info  sqrt(sum((truth-sol_init).^2)/N_domain)
    # @info maximum(abs.(truth-sol_init))
    if args.compare_exact
        GNsteps_exact = 2
        @info "[comparison: exact method]"
        @time sol_exact, sol_xx, sol_xy, sol_yy = iterGPR_exact(eqn, cov, X_domain, X_boundary, sol_init, sol_init_xx, sol_init_xy, sol_init_yy, nugget, GNsteps_exact)
        pts_accuracy_exact = sqrt(sum((truth-sol_exact).^2)/N_domain)
        @info "[L2 accuracy: exact method] $pts_accuracy_exact"
        pts_max_accuracy_exact = maximum(abs.(truth-sol_exact))
        @info "[Linf accuracy: exact method] $pts_max_accuracy_exact"
        
        # @show sol_init_xx.*sol_yy .+ sol_init_yy.*sol_xx .- 2*sol_xy.*sol_init_xy .- 2*[eqn.rhs(X_domain[:,i]) for i in 1:N_domain]
        # @show sol_xx
        # @show sol_yy
    end

    
end

args = parse_commandline()
args = (; (Symbol(k) => v for (k,v) in args)...) # Named tuple from dict
main(args)
