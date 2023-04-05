# using Pkg
# Pkg.activate(@__DIR__)

# For linear algebra
using StaticArrays: SVector 
using LinearAlgebra
using SparseArrays: SparseMatrixCSC, sparse
# Fast Cholesky
using KoLesky 

# multiple dispatch
import IterativeSolvers: cg!
import LinearAlgebra: mul!, ldiv!
import Base: size

# parser
using ArgParse

# logging
using Logging

using ForwardDiff
using FastGaussQuadrature



# solving Burgers: u_t+ u u_x- nu u_xx=0
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--nu"
            help = "ν"
            arg_type = Float64
            # default = 0.01/π
            default = 0.001
        "--kernel"
            arg_type = String
            default = "Matern7half"
        "--sigma"
            help = "lengthscale"
            arg_type = Float64
            default = 0.02
        "--h"
            help = "spatial grid size"
            arg_type = Float64
            default = 0.001
        "--dt"
            help = "time step size"
            arg_type = Float64
            default = 0.02
        "--T"
            help = "final time"
            arg_type = Float64
            default = 1.0
        "--nugget"
            arg_type = Float64
            default = 1e-10
        "--GNsteps"
            arg_type = Int
            default = 2
        "--rho_big"
            arg_type = Float64
            default = 3.0
        "--rho_small"
            arg_type = Float64
            default = 3.0
        "--k_neighbors"
            arg_type = Int
            default = 1
        "--compare_exact"
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end

## PDEs type
abstract type AbstractPDEs end
struct Burgers1d{Tν} <: AbstractPDEs
    # eqn: u_t+ u u_x- νu_xx=0, (x,t) in [-1,1]*[0,1]
    ν::Tν
    bdy::Function
    rhs::Function
    init::Function
    init_∂x::Function
    init_∂∂x::Function
end

## sample points
function sample_points_grid(eqn::Burgers1d, h_in)
    X_domain = -1+h_in:h_in:1-h_in
    # X_domain = -1:h_in:1
    X_boundary = [-1,1]
    return collect(X_domain), X_boundary
end

## exact algorithms
# assemby Gram matrices
function get_Gram_matrices(eqn::Burgers1d, cov::KoLesky.AbstractCovarianceFunction, X_domain, X_boundary, dt, sol, sol_x)
    d = 1
    N_domain = length(X_domain)
    N_boundary = length(X_boundary)

    Δδ_coefs = -eqn.ν
    ∇δ_coefs = sol
    δ_coefs_int = 2/dt .+ sol_x # Crank-Nikoson discretization

    # get linearized PDEs correponding measurements
    meas_δ = [KoLesky.PointMeasurement{d}(SVector{d,Float64}([X_boundary[i]])) for i = 1:N_boundary]
    meas_Δ∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), Δδ_coefs, [∇δ_coefs[i]], δ_coefs_int[i]) for i = 1:N_domain]
    meas_δ_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}([X_domain[i]])) for i = 1:N_domain]

    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)

    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ; measurements[2] = meas_Δ∇δ
    cov(Theta_train, reduce(vcat,measurements))
    
    meas_∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), 0.0, [1.0], 0.0) for i = 1:N_domain]
    meas_Δδ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), 1.0, [0.0], 0.0) for i = 1:N_domain]
    measurements_test = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,3)
    measurements_test[1] = meas_δ_int
    measurements_test[2] = meas_∇δ
    measurements_test[3] = meas_Δδ

    Theta_test = zeros(3*N_domain,N_domain+N_boundary)
    cov(Theta_test, reduce(vcat,measurements_test), reduce(vcat, measurements))
    return Theta_train, Theta_test

end


# iterative GPR
function iterGPR_exact(eqn, cov, X_domain, X_boundary, nugget, dt, T, GNsteps)
    N_domain = length(X_domain)
    N_boundary = length(X_boundary)
    Nt = convert(Int,T/dt)
    # get the rhs and bdy data
    # @show X_domain
    sol_u = [eqn.init(X_domain[i]) for i in 1:N_domain]
    sol_ux = [eqn.init_∂x(X_domain[i]) for i in 1:N_domain]
    sol_uxx = [eqn.init_∂∂x(X_domain[i]) for i in 1:N_domain] 
    # @show sol_u
    bdy = [eqn.bdy(X_boundary[i]) for i in 1:N_boundary]
    rhs = [eqn.rhs(X_domain[i]) for i in 1:N_domain]

    rhs_CN = zeros(N_boundary+N_domain)
    rhs_CN[1:N_boundary] = bdy

    cur_rhs_CN = copy(rhs_CN)

    # measurements construction only once
    d = 1
    meas_δ_bd = [KoLesky.PointMeasurement{d}(SVector{d,Float64}([X_boundary[i]])) for i = 1:N_boundary]
    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
    measurements[1] = meas_δ_bd; 

    meas_∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), 0.0, [1.0], 0.0) for i = 1:N_domain]
    meas_Δδ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), 1.0, [0.0], 0.0) for i = 1:N_domain]
    meas_δ_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}([X_domain[i]])) for i = 1:N_domain]
    measurements_test = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,3)
    measurements_test[1] = meas_δ_int
    measurements_test[2] = meas_∇δ
    measurements_test[3] = meas_Δδ
    Theta_train = zeros(N_domain+N_boundary,N_domain+N_boundary)
    Theta_test = zeros(3*N_domain,N_domain+N_boundary)
    @inbounds for it in 1:Nt
        t = it*dt
        @info "[time] t = $t"
        rhs_CN[N_boundary+1:end] .= rhs + 2/dt*sol_u + eqn.ν*sol_uxx - sol_u.*sol_ux
        for _ in 1:GNsteps
            cur_rhs_CN[N_boundary+1:end] .= rhs_CN[N_boundary+1:end] + sol_u.*sol_ux

            d = 1
            Δδ_coefs = -eqn.ν        
            ∇δ_coefs = sol_u
            δ_coefs_int = 2/dt .+ sol_ux # Crank-Nikoson discretization    
            meas_Δ∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), Δδ_coefs, [∇δ_coefs[i]], δ_coefs_int[i]) for i = 1:N_domain]
            measurements[2] = meas_Δ∇δ

            cov(Theta_train, reduce(vcat,measurements))
            cov(Theta_test, reduce(vcat,measurements_test), reduce(vcat, measurements))
        
            v = Theta_test*(((Theta_train+nugget*diagm(diag(Theta_train))))\cur_rhs_CN)

            @views sol_u .= v[1:N_domain]
            @views sol_ux .= v[N_domain+1:2*N_domain]
            @views sol_uxx .= v[2*N_domain+1:end]
        end
    end
    return sol_u
end 

## algorithm using Kolesky and pcg
# struct that stores the factor of Theta_train
abstract type implicit_mtx end
struct approx_Theta_train{Tv,Ti,Tmtx<:SparseMatrixCSC{Tv,Ti}} <: implicit_mtx
    P::Vector{Ti}
    U::Tmtx
    L::Tmtx
    δ_coefs::Vector{Tv}
    ∇δ_coefs::Vector{Tv}
    Δδ_coef::Tv
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
    @views temp = vcat(b[1:Θtrain.N_boundary],Θtrain.δ_coefs.*b[Θtrain.N_boundary+1:end],Θtrain.∇δ_coefs.*b[Θtrain.N_boundary+1:end],Θtrain.Δδ_coef*b[Θtrain.N_boundary+1:end])
    temp[Θtrain.P] = Θtrain.L\(Θtrain.U\temp[Θtrain.P])

    @views x[1:Θtrain.N_boundary] = temp[1:Θtrain.N_boundary]
    @views x[Θtrain.N_boundary+1:end] .= Θtrain.δ_coefs.*(temp[Θtrain.N_boundary+1:Θtrain.N_boundary+Θtrain.N_domain]) .+ Θtrain.∇δ_coefs.* temp[Θtrain.N_boundary+Θtrain.N_domain+1:Θtrain.N_boundary+2*Θtrain.N_domain] + Θtrain.Δδ_coef * temp[Θtrain.N_boundary+2*Θtrain.N_domain+1:end]
end

function ldiv!(x, precond::precond_Theta_train, b)
    x[precond.P] = precond.U*(precond.L*b[precond.P])
end

function iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, dt, T, nugget, GNsteps; ρ_big =4.0, ρ_small=6.0, k_neighbors = 4, lambda = 1.5, alpha = 1.0)
    
    @info "[Algorithm]: iterative GPR + fast cholesky factorization + pcg"

    N_domain = length(X_domain)
    N_boundary = length(X_boundary)
    Nt = convert(Int,T/dt)
    # get the rhs and bdy data
    # @show X_domain
    sol_u = [eqn.init(X_domain[i]) for i in 1:N_domain]
    sol_ux = [eqn.init_∂x(X_domain[i]) for i in 1:N_domain]
    sol_uxx = [eqn.init_∂∂x(X_domain[i]) for i in 1:N_domain] 
    # @show sol_u
    bdy = [eqn.bdy(X_boundary[i]) for i in 1:N_boundary]
    rhs = [eqn.rhs(X_domain[i]) for i in 1:N_domain]

    rhs_CN = zeros(N_boundary+N_domain)
    rhs_CN[1:N_boundary] = bdy

    cur_rhs_CN = copy(rhs_CN)

    # measurements construction only once
    d = 1
    meas_δ_bd = [KoLesky.PointMeasurement{d}(SVector{d,Float64}([X_boundary[i]])) for i = 1:N_boundary]
    meas_∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), 0.0, [1.0], 0.0) for i = 1:N_domain]
    meas_Δδ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), 1.0, [0.0], 0.0) for i = 1:N_domain]
    meas_δ_int = [KoLesky.PointMeasurement{d}(SVector{d,Float64}([X_domain[i]])) for i = 1:N_domain]
    measurements = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,4)
    measurements[1] = meas_δ_bd; 
    measurements[2] = meas_δ_int
    measurements[3] = meas_∇δ
    measurements[4] = meas_Δδ
    
    @info "[Big Theta: implicit factorization] time"
    @time implicit_bigΘ = KoLesky.ImplicitKLFactorization_FollowDiracs(cov, measurements, ρ_big, k_neighbors; lambda = lambda, alpha = alpha)

    @info "[Big Theta: explicit factorization] time"
    @time explicit_bigΘ = KoLesky.ExplicitKLFactorization(implicit_bigΘ; nugget = nugget)
    U_bigΘ = explicit_bigΘ.U
    L_bigΘ = sparse(U_bigΘ')
    P_bigΘ = explicit_bigΘ.P

    Θtrain = approx_Theta_train(P_bigΘ, U_bigΘ, L_bigΘ,zeros(N_domain),zeros(N_domain), -eqn.ν, N_boundary,N_domain)

    implicit_factor = nothing
    
    measurement_train = Vector{Vector{<:KoLesky.AbstractPointMeasurement}}(undef,2)
    measurement_train[1] = meas_δ_bd; 

    @inbounds for it in 1:Nt
        t = it*dt
        @info "[time] t = $t"
        rhs_CN[N_boundary+1:end] .= rhs + 2/dt*sol_u + eqn.ν*sol_uxx - sol_u.*sol_ux
        for step in 1:GNsteps
            @info "[Current GN step] $step"
            cur_rhs_CN[N_boundary+1:end] .= rhs_CN[N_boundary+1:end] + sol_u.*sol_ux

            d = 1
            Δδ_coefs = -eqn.ν        
            ∇δ_coefs = sol_u
            δ_coefs_int = 2/dt .+ sol_ux # Crank-Nikoson discretization    
            meas_Δ∇δ = [KoLesky.Δ∇δPointMeasurement{Float64,d}(SVector{d,Float64}([X_domain[i]]), Δδ_coefs, [∇δ_coefs[i]], δ_coefs_int[i]) for i = 1:N_domain]
            measurement_train[2] = meas_Δ∇δ

            @info "[Theta Train: implicit factorization] time"
            @time if implicit_factor === nothing
                implicit_factor = KoLesky.ImplicitKLFactorization(cov,  measurement_train, ρ_small, k_neighbors; lambda = lambda, alpha = alpha)
            else
                implicit_factor.supernodes.measurements .= reduce(vcat, collect.(measurement_train))[implicit_factor.P]
            end
            @info "[Theta Train: explicit factorization] time"
            @time explicit_factor = KoLesky.ExplicitKLFactorization(implicit_factor; nugget = nugget)
        
            U = explicit_factor.U
            L = sparse(U')
            P = explicit_factor.P

            # use the approximate solution as the initial point for the pCG iteration
            Θinv_rhs = U*(L*cur_rhs_CN[P])
            Θinv_rhs[P] = Θinv_rhs

            # pcg step for Theta_train\rhs
            Θtrain.δ_coefs .= δ_coefs_int
            Θtrain.∇δ_coefs .= ∇δ_coefs
            precond = precond_Theta_train(P,U,L)
            
            @info "[pcg started]"
            @time Θinv_rhs, ch = cg!(Θinv_rhs, Θtrain, cur_rhs_CN; Pl = precond, log=true)
            @info "[pcg finished], $ch"
            
            # get approximated sol_now = Theta_test * Θinv_rhs
            tmp = vcat(Θinv_rhs[1:N_boundary], δ_coefs_int .* Θinv_rhs[N_boundary+1:end], ∇δ_coefs.*Θinv_rhs[N_boundary+1:end], Δδ_coefs *Θinv_rhs[N_boundary+1:end])
            tmp[P_bigΘ] = L_bigΘ\(U_bigΘ\tmp[P_bigΘ]) 
            
            @views sol_u = tmp[N_boundary+1:N_domain+N_boundary] 
            @views sol_ux .= tmp[N_domain+N_boundary+1:2*N_domain+N_boundary]
            @views sol_uxx .= tmp[2*N_domain+N_boundary+1:3*N_domain+N_boundary]
        end
    end
    @info "[solver finished]"
    return sol_u
end

function SolveBurgers_ColeHopf(x,t,ν)
    Gauss_pts, weights = gausshermite(100)
    temp = x.-sqrt(4*ν*t)*Gauss_pts
    val1 = weights .* sin.(π*temp) .* exp.(-cos.(π*temp)/(2*π*ν))
    val2 = weights .* exp.(-cos.(π*temp)/(2*π*ν))
    return -sum(val1)/sum(val2)
end


args = parse_commandline()
args = (; (Symbol(k) => v for (k,v) in args)...) # Named tuple from dict
# main(args)


# function main(args)
ν = args.nu
# ground truth solution
function fun_u(x)
    return -sin(π*x)
end

function grad_u(x)
    return ForwardDiff.derivative(fun_u, x)
end

function Delta_u(x)
    return ForwardDiff.derivative(grad_u, x)
end

# right hand side
function fun_rhs(x)
    return 0.0
end

# boundary value
function fun_bdy(x)
    return 0.0
end

@info "[solver started] Burgers 1d"
@info "[equation] u_t+ u u_x- $ν u_xx=0"
eqn = Burgers1d(ν,fun_bdy,fun_rhs,fun_u,grad_u,Delta_u)

h_in = args.h::Float64
X_domain, X_boundary = sample_points_grid(eqn, h_in)
N_domain = length(X_domain)
N_boundary = length(X_boundary)
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

dt = args.dt::Float64
T = args.T::Float64


GNsteps_approximate = args.GNsteps::Int
@info "[total GN steps] $GNsteps_approximate" 

ρ_big = args.rho_big::Float64
ρ_small = args.rho_small::Float64
k_neighbors = args.k_neighbors::Int
@info "[Fast Cholesky] ρ_big = $ρ_big, ρ_small = $ρ_small, k_neighbors = $k_neighbors"



fast_solve() = @time iterGPR_fast_pcg(eqn, cov, X_domain, X_boundary, dt, T, nugget, GNsteps_approximate; ρ_big = ρ_big, ρ_small = ρ_small, k_neighbors=k_neighbors);
sol = fast_solve()

truth =  [SolveBurgers_ColeHopf(X_domain[i],T,ν) for i in 1:N_domain]
pts_accuracy = sqrt(sum((truth-sol).^2)/N_domain)
@info "[L2 accuracy: pCG method] $pts_accuracy"
pts_max_accuracy = maximum(abs.(truth-sol))
@info "[Linf accuracy: pCG method] $pts_max_accuracy"

if args.compare_exact
    GNsteps_exact = 2
    @info "[comparison: exact method]"
    @time sol_exact = iterGPR_exact(eqn, cov, X_domain, X_boundary, nugget, dt, T, GNsteps_exact)
    pts_accuracy_exact = sqrt(sum((truth-sol_exact).^2)/N_domain)
    @info "[L2 accuracy: exact method] $pts_accuracy_exact"
    pts_max_accuracy_exact = maximum(abs.(truth-sol_exact))
    @info "[Linf accuracy: exact method] $pts_max_accuracy_exact"
    plot!(X_domain,sol_exact)
end


using PyCall
using PyPlot
function burgers_plot(X_domain,truth,sol)
    fsize = 15.0
    tsize = 15.0
    tdir = "in"
    major = 5.0
    minor = 3.0
    lwidth = 0.8
    lhandle = 2.0
    plt.style.use("default")
    rcParams = PyDict(matplotlib["rcParams"])
    rcParams["font.size"] = fsize
    rcParams["legend.fontsize"] = tsize
    rcParams["xtick.direction"] = tdir
    rcParams["ytick.direction"] = tdir
    rcParams["xtick.major.size"] = major
    rcParams["xtick.minor.size"] = minor
    rcParams["ytick.major.size"] = 5.0
    rcParams["ytick.minor.size"] = 3.0
    rcParams["axes.linewidth"] = lwidth
    rcParams["legend.handlelength"] = lhandle
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X_domain,truth, "-", linewidth=2.5, label="true sol")
    ax.plot(X_domain, sol, linestyle="dashed", linewidth=2.5, color="red", label="numeric sol")
    ax.legend()
    plt.xlabel(L"$x$")
    display(fig)

    fig.savefig("burgers_solution.pdf", bbox_inches="tight",dpi=100,pad_inches=0.15)
end
burgers_plot(X_domain,truth,sol)



    