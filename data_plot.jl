using JLD
using Printf

result = load("NonlinElliptic2d_data.jld")["result"]

# arr_kernel = ["Matern5half", "Matern7half", "Matern9half"]

arr_kernel = ["Matern5half"]
arr_h = [0.02,0.01,0.005,0.0025]
arr_ρ = [2.0, 3.0, 4.0]


data = Dict()

for kernel in arr_kernel
    data[kernel] = Dict()
    for ρ in arr_ρ
        data[kernel][ρ] = Dict()
        L2err = zeros(length(arr_h))
        Linferr = zeros(length(arr_h))
        time = zeros(length(arr_h))
        for idx in 1:length(arr_h)
            L2err[idx] = result[("kernel", kernel)][("h",arr_h[idx])][("rho", ρ)]["L2"]
            Linferr[idx] = result[("kernel", kernel)][("h",arr_h[idx])][("rho", ρ)]["Linf"]
            time[idx] = result[("kernel", kernel)][("h",arr_h[idx])][("rho", ρ)]["time"]
        end
        data[kernel][ρ]["L2"] = L2err
        data[kernel][ρ]["Linf"] = Linferr
        data[kernel][ρ]["time"] = time
    end
end

function linreg1d(x,y)
    ymean = sum(y)/length(y)
    xmean = sum(x)/length(x)
    β = sum((y.-ymean).*(x.-xmean))/sum((x.-xmean).*(x.-xmean))
    b = y .- β*x
    return β, b
end

using PyCall
using PyPlot
fsize = 15.0
tsize = 15.0
tdir = "in"
major = 5.0
minor = 3.0
lwidth = 0.8
lhandle = 2.0
plt.style.use("default")
rcParams = PyDict(matplotlib["rcParams"])
# rcParams["text.usetex"] = true
# I am not able to make tex font work. Using the anaconda pyplot environment leads to MKL errors, while the julia python environment cannot support tex font.
# need to ask Daniel how to have good plots
# rcParams["text.latex.preamble"] = [raw"\usepackage{amsmath}"]
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

# @show result

arr_N = (1.0./arr_h.+1).^2

plotL2 = plt.figure()
ax = plotL2.add_subplot(111)
for kernel in arr_kernel
    for ρ in arr_ρ
        X = arr_N
        Y = data[kernel][ρ]["L2"]
        slope, b = linreg1d(log.(X),log.(Y))
        kernelname = kernel[1:end-4] * "/2"
        ax.plot(X, Y, "-s", label = kernelname*L", $\rho=$"*"$ρ, slope $(@sprintf("%.2f", slope))")
    end
end
plt.xscale("log")
plt.yscale("log")
plt.xlabel(L"$N_{\mathrm{domain}}}$")
plt.ylabel(L"$L^{2}$ error")
plt.title(L"Solution Accuracy $L^{2}$")
plt.legend()
display(plotL2)

plotLinf = plt.figure()
ax = plotLinf.add_subplot(111)
for kernel in arr_kernel
    for ρ in arr_ρ
        X = arr_N
        Y = data[kernel][ρ]["Linf"]
        kernelname = kernel[1:end-4] * "/2"
        slope, b = linreg1d(log.(X),log.(Y))
        ax.plot(X, Y, "-s", label = kernelname*L", $\rho=$"*"$ρ, slope $(@sprintf("%.2f", slope))")
    end
end
plt.xscale("log")
plt.yscale("log")
plt.xlabel(L"$N_{\mathrm{domain}}}$")
plt.ylabel(L"$L^{\infty}$ error")
plt.title(L"Solution Accuracy $L^{\infty}$")
plt.legend()
display(plotLinf)

plottime = plt.figure()
ax = plottime.add_subplot(111)
for kernel in arr_kernel
    for ρ in arr_ρ
        X = arr_N
        Y = data[kernel][ρ]["time"]
        slope, b = linreg1d(log.(X),log.(Y))
        kernelname = kernel[1:end-4] * "/2"
        ax.plot(X, Y, "-s", label = kernelname*L", $\rho=$"*"$ρ, slope $(@sprintf("%.2f", slope))")
    end
end
plt.xscale("log")
plt.yscale("log")
plt.xlabel(L"$N_{\mathrm{domain}}}$")
plt.ylabel("Time")
plt.title("Computational Efficiency")
plt.legend()
display(plottime)

plottradeoff = plt.figure()
ax = plottradeoff.add_subplot(111)
for kernel in arr_kernel
    for ρ in arr_ρ
        X = data[kernel][ρ]["time"]
        Y = data[kernel][ρ]["L2"]
        slope, b = linreg1d(log.(X),log.(Y))
        kernelname = kernel[1:end-4] * "/2"
        ax.plot(X, Y, "-s", label = kernelname*L", $\rho=$"*"$ρ, slope $(@sprintf("%.2f", slope))")
    end
end
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel(L"$L^{2}$ error")
plt.title("Speed–Accuracy Tradeoff")
plt.legend()
display(plottradeoff)