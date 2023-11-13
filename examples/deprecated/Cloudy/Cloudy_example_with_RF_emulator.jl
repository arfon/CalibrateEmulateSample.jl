# Reference the in-tree version of CalibrateEmulateSample on Julias load path
include(joinpath(@__DIR__, "../..", "ci", "linkfig.jl"))
include(joinpath(@__DIR__, "DynamicalModel.jl")) # Import the module that runs Cloudy

# This example requires Cloudy to be installed (it's best to install the master
# branch), which can be done by:
#] add Cloudy#master
using Cloudy
using Cloudy.ParticleDistributions
using Cloudy.KernelTensors

# Import the module that runs Cloudy
#include("DynamicalModel.jl")
#using .DynamicalModel

# Import modules
using Distributions  # probability distributions and associated functions
using StatsBase
#using StableRNGs # needed?
using LinearAlgebra
using StatsPlots
using Plots
using Plots.PlotMeasures # is this needed?
using Random
using JLD2 # is this needed?

# Import Calibrate-Emulate-Sample modules
# For the calibration step we use the EnsembleKalmanProcesses package, and
# for the sampling we use Markov chain Monte Carlo methods. We'll 
# run this example twice, using first a Gaussian process emulator, and 
# then a Random Feature emulator.
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.DataContainers
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities

# This example requires Cloudy to be installed.
using Cloudy
const PDistributions = Cloudy.ParticleDistributions

function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2) # N_data x 1 array 
    return norm_factor
end


################################################################################
#                                                                              #
#                      Cloudy Calibrate-Emulate-Sample Example                 #
#                                                                              #
#                                                                              #
#     This example uses Cloudy, a microphysics model that simulates the        #
#     coalescence of cloud droplets into bigger drops, to demonstrate how      #
#     the full Calibrate-Emulate-Sample pipeline can be used for Bayesian      #
#     learning and uncertainty quantification of parameters, given some        #
#     observations.                                                            #
#                                                                              #
#     Specifically, this examples shows how to learn parameters of the         #
#     initial cloud droplet mass distribution, given observations of some      #
#     moments of that mass distribution at a later time, after some of the     #
#     droplets have collided and become bigger drops.                          #
#                                                                              #
#     In this example, Cloudy is used in a "perfect model" (aka "known         #
#     truth") setting, which means that the "observations" are generated by    #
#     Cloudy itself, by running it with the true parameter values. In more     #
#     realistic applications, the observations will come from some external    #
#     measurement system.                                                      #
#                                                                              #
#     The purpose is to show how to do parameter learning using                #
#     Calibrate-Emulate-Sample in a simple (and highly artificial) setting.    #
#                                                                              #
#     For more information on Cloudy, see                                      #
#              https://github.com/CliMA/Cloudy.jl.git                          #
#                                                                              #
################################################################################


rng_seed = 41
Random.seed!(rng_seed)
rng = Random.seed!(Random.GLOBAL_RNG, rng_seed)

homedir = pwd()
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"

data_save_file = joinpath(data_save_directory, "cloudy_calibrate_results.jld2")
ekiobj = load(data_save_file)["eki"]
priors = load(data_save_file)["priors"]
truth_sample_mean = load(data_save_file)["truth_sample_mean"]
truth_sample = load(data_save_file)["truth_sample"]
# True parameters:
# - ϕ: in constrained space
# - θ: in unconstrained space
ϕ_true = load(data_save_file)["truth_input_constrained"]
θ_true = transform_constrained_to_unconstrained(priors, ϕ_true)
Γy = ekiobj.obs_noise_cov

n_params = length(ϕ_true) # input dimension


###
###  Emulate: Random Features
###


# Setup random features
n_features = 400

min_iter = 1
max_iter = 5 # number of EKP iterations to use data from is at most this
optimizer_options = Dict(
    "verbose" => true,
    "scheduler" => DataMisfitController(terminate_at = 100.0),
    "cov_sample_multiplier" => 1.0,
    "n_iteration" => 20,
)

nugget = 1e-8 # What is this?
kernel_structure = SeparableKernel(LowRankFactor(3, nugget), OneDimFactor())
srfi = ScalarRandomFeatureInterface(
    n_features,
    n_par,
    kernel_structure = kernel_structure,
    optimizer_options = optimizer_options,
)

## Standardize the output data
#input_output_pairs = Utilities.get_training_points(ekiobj, N_iter)
norm_factor = get_standardizing_factors(get_outputs(input_output_pairs))
norm_factor = vcat(norm_factor...)

# Get training points from the EKP iteration number in the second input term  
N_iter = min(max_iter, length(get_u(ekiobj)) - 1) # number of paired iterations taken from EKP
min_iter = min(max_iter, max(1, min_iter))
input_output_pairs = Utilities.get_training_points(ekiobj, min_iter:(N_iter - 1))
input_output_pairs_test = Utilities.get_training_points(ekiobj, N_iter:(length(get_u(ekiobj)) - 1)) #  "next" iterations
# Save data
@save joinpath(data_save_directory, "cloudy_input_output_pairs.jld2") input_output_pairs

# Train emulator
standardize = true
retained_svd_frac = 1.0
normalized = true
# do we want to use SVD to decorrelate outputs
decorrelate = true


emulator = Emulator(
    srfi,
    input_output_pairs;
    obs_noise_cov = Γy,
    normalize_inputs = normalized,
    standardize_outputs = standardize,
    standardize_outputs_factors = norm_factor,
    retained_svd_frac = retained_svd_frac,
    decorrelate = decorrelate,
)

optimize_hyperparameters!(emulator)

# Check how well the Random Feature emulator predicts on the
# true parameters
y_mean, y_var = Emulators.predict(emulator, reshape(θ_true, :, 1); transform_to_real = true)

y_mean_test, y_var_test =
    Emulators.predict(emulator, get_inputs(input_output_pairs_test), transform_to_real = true)

println("Random Feature (RF) emulator prediction on true parameters: ")
println(vec(y_mean))
println("true data: ")
println(truth_sample) # what was used as truth
println("RF predicted standard deviation")
println(sqrt.(diag(y_var[1], 0)))
println("RF MSE (truth): ")
println(mean((truth_sample - vec(y_mean)) .^ 2))
println("RF MSE (next ensemble): ")
println(mean((get_outputs(input_output_pairs_test) - y_mean_test) .^ 2))

###
###  Sample: Markov Chain Monte Carlo
###

# initial values
u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
println("initial parameters: ", u0)

# First let's run a short chain to determine a good step size
yt_sample = truth_sample
mcmc = MCMCWrapper(RWMHSampling(), yt_sample, priors, emulator; init_params = u0)
new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)
chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 1_000)
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

post_mean = mean(posterior)
post_cov = cov(posterior)
println("posterior mean")
println(post_mean)
println("posterior covariance")
println(post_cov)

# Plot the posteriors together with the priors and the true parameter values
# (in the transformed/unconstrained space)

gr(size = (800, 600))

for idx in 1:n_params
    if idx == 1
        xs = collect(range(3.0, stop = 5.0, length = 1000))
    elseif idx == 2
        xs = collect(range(0.0, stop = 0.5, length = 1000))
    elseif idx == 3
        xs = collect(range(-3.0, stop = -2.0, length = 1000))
    else
        throw("not implemented")
    end

    label = "true " * par_names[idx]
    posterior_samples = dropdims(get_distribution(posterior)[par_names[idx]], dims = 1)
    histogram(
        posterior_samples,
        bins = 100,
        normed = true,
        fill = :slategray,
        thickness_scaling = 2.0,
        lab = "posterior",
        legend = :outertopright,
    )
    prior_dist = get_distribution(mcmc.prior)[par_names[idx]]
    plot!(xs, prior_dist, w = 2.6, color = :blue, lab = "prior")
    plot!([θ_true[idx]], seriestype = "vline", w = 2.6, lab = label)
    title!(par_names[idx])
    figpath = joinpath(figure_save_directory, "posterior_" * par_names[idx] * "_RF_emulator.png")
    StatsPlots.savefig(figpath)
    linkfig(figpath)
end
