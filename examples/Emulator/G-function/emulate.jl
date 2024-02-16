
using GlobalSensitivityAnalysis
const GSA = GlobalSensitivityAnalysis
using Distributions
using DataStructures
using Random
using LinearAlgebra

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers

using CairoMakie, ColorSchemes #for plots
seed = 2589456

output_directory = joinpath(@__DIR__, "output")
if !isdir(output_directory)
    mkdir(output_directory)
end


inner_func(x::AV,a::AV) where {AV <: AbstractVector} = prod((abs.(4 * x .- 2) + a) ./ (1 .+ a))

"G-Function taken from https://www.sfu.ca/~ssurjano/gfunc.html"
function GFunction(x::AM, a::AV) where {AM <: AbstractMatrix, AV <: AbstractVector}
    @assert size(x,1) == length(a)
    return mapslices(y -> inner_func(y,a), x; dims=1) #applys the map to columns
end
    
function GFunction(x::AM) where {AM <: AbstractMatrix}
    a = [(i-2.0)/2.0 for i = 1:size(x,1)]
    return GFunction(x,a)
end
    
function main()

    rng = MersenneTwister(seed)

    n_repeats = 20 # repeat exp with same data.
    n_dimensions = 10
    # To create the sampling
    n_data_gen = 1000

    data = SobolData(
        params = OrderedDict([Pair(Symbol("x",i),Uniform(0,1)) for i = 1:n_dimensions]),
        N = n_data_gen,
    )

    # To perform global analysis,
    # one must generate samples using Sobol sequence (i.e. creates more than N points)
    samples = GSA.sample(data)
    n_data = size(samples, 1) # [n_samples x n_dim]
    println("number of sobol points: ", n_data)
    # run model (example)
    y = GFunction(samples')' # G is applied to columns
    # perform Sobol Analysis
    result = analyze(data, y)

    # plot the first 3 dimensions
    f1 = Figure(resolution = (1.618 * 900, 300), markersize = 4)
    axx = Axis(f1[1, 1], xlabel = "x1", ylabel = "f")
    axy = Axis(f1[1, 2], xlabel = "x2", ylabel = "f")
    axz = Axis(f1[1, 3], xlabel = "x3", ylabel = "f")

    scatter!(axx, samples[:, 1], y[:], color = :orange)
    scatter!(axy, samples[:, 2], y[:], color = :orange)
    scatter!(axz, samples[:, 3], y[:], color = :orange)

    save(joinpath(output_directory, "GFunction_slices_truth.png"), f1, px_per_unit = 3)
    save(joinpath(output_directory, "GFunction_slices_truth.pdf"), f1, px_per_unit = 3)

    n_train_pts = 1000
    ind = shuffle!(rng, Vector(1:n_data))[1:n_train_pts]
    # now subsample the samples data
    n_tp = length(ind)
    input = zeros(n_dimensions, n_tp)
    output = zeros(1, n_tp)
    Γ = 1e-3
    noise = rand(rng, Normal(0, Γ), n_tp)
    for i in 1:n_tp
        input[:, i] = samples[ind[i], :]
        output[i] = y[ind[i]] + noise[i]
    end
    iopairs = PairedDataContainer(input, output)

    cases = ["Prior", "GP", "RF-scalar"]
    case = cases[3]
    decorrelate = true
    nugget = Float64(1e-12)

    overrides =
        Dict("verbose" => true,
             "scheduler" => DataMisfitController(terminate_at = 1e4),
             "n_features_opt" => 100,
             "n_iteration" => 20,
             "cov_sample_multiplier" => 5.0,
             "n_ensemble" => 200)
    if case == "Prior"
        # don't do anything
        overrides["n_iteration"] = 0
        overrides["cov_sample_multiplier"] = 0.1
    end

    y_preds = []
    result_preds = []

    for rep_idx in 1:n_repeats

        # Build ML tools
        if case == "GP"
            gppackage = Emulators.SKLJL()
            pred_type = Emulators.YType()
            mlt = GaussianProcess(gppackage; prediction_type = pred_type, noise_learn = false)

        elseif case ∈ ["RF-scalar", "Prior"]

            kernel_structure = SeparableKernel(LowRankFactor(9, nugget), OneDimFactor())
            n_features = n_dimensions*200
            mlt = ScalarRandomFeatureInterface(
                n_features,
                n_dimensions,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = deepcopy(overrides),
            )
        end

        # Emulate
        emulator = Emulator(mlt, iopairs; obs_noise_cov = Γ * I, decorrelate = decorrelate)
        optimize_hyperparameters!(emulator)

        # predict on all Sobol points with emulator (example)    
        y_pred, y_var = predict(emulator, samples', transform_to_real = true)

        # obtain emulated Sobol indices
        result_pred = analyze(data, y_pred')
        push!(y_preds, y_pred)
        push!(result_preds, result_pred)

    end

    # analytic sobol indices taken from
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8989694/pdf/main.pdf
    a = [(i-2.0)/2.0 for i = 1:n_dimensions]  # a_i < a_j => a_i more sensitive
    prod_tmp = prod(1 .+ 1 ./ (3 .* (1 .+ a).^2))-1
    V = [(1/(3*(1+ai)^2)) / prod_tmp for ai in a]
    prod_tmp2 = [prod(1 .+ 1 ./ (3 .* (1 .+ a[1:end .!== j]).^2)) for j = 1:n_dimensions]
    TV = [(1/(3*(1+ai)^2))*prod_tmp2[i] / prod_tmp for (i,ai) in enumerate(a)]

    println(" ")
    println("True Sobol Indices")
    println("******************")
    println("    firstorder: ", V)
    println("    totalorder: ", TV)
    println(" ")
    println("Sampled truth Sobol Indices (# points $n_data)")
    println("***************************")
    println("    firstorder: ", result[:firstorder])
    println("    totalorder: ", result[:totalorder])
    println(" ")

    println("Sampled Emulated Sobol Indices (# obs $n_train_pts, noise var $Γ)")
    println("***************************************************************")
    if n_repeats == 1
        println("    firstorder: ", result_preds[1][:firstorder])
        println("    totalorder: ", result_preds[1][:totalorder])
    else
        firstorder_mean = mean([rp[:firstorder] for rp in result_preds])
        firstorder_std = std([rp[:firstorder] for rp in result_preds])
        totalorder_mean = mean([rp[:totalorder] for rp in result_preds])
        totalorder_std = std([rp[:totalorder] for rp in result_preds])

        println("(mean) firstorder: ", firstorder_mean)
        println("(std)  firstorder: ", firstorder_std)
        println("(mean) totalorder: ", totalorder_mean)
        println("(std)  totalorder: ", totalorder_std)

        #
        f3, ax3, plt3 = errorbars(1:n_dimensions,firstorder_mean, 2*firstorder_std; whiskerwidth=10, color=:red)
        scatter!(ax3, result[:firstorder], color= :black, markersize=8)
        errorbars!(ax3, 1:n_dimensions,firstorder_mean, 3*firstorder_std; whiskerwidth=10, color=:red)
        errorbars!(ax3, 1:n_dimensions,firstorder_mean, firstorder_std; whiskerwidth=10, color=:red)

        save(joinpath(output_directory, "GFunction_sens_$(case).png"), f3, px_per_unit = 3)
        save(joinpath(output_directory, "GFunction_sens_$(case).pdf"), f3, px_per_unit = 3)
    end

    

    # plots - first 3 dimensions

    f2 = Figure(resolution = (1.618 * 900, 300), markersize = 4)
    axx_em = Axis(f2[1, 1], xlabel = "x1", ylabel = "f")
    axy_em = Axis(f2[1, 2], xlabel = "x2", ylabel = "f")
    axz_em = Axis(f2[1, 3], xlabel = "x3", ylabel = "f")
    scatter!(axx_em, samples[:, 1], y_preds[1][:], color = :blue)
    scatter!(axy_em, samples[:, 2], y_preds[1][:], color = :blue)
    scatter!(axz_em, samples[:, 3], y_preds[1][:], color = :blue)
    scatter!(axx_em, samples[ind, 1], y[ind] + noise, color = :red, markersize = 8)
    scatter!(axy_em, samples[ind, 2], y[ind] + noise, color = :red, markersize = 8)
    scatter!(axz_em, samples[ind, 3], y[ind] + noise, color = :red, markersize = 8)

    save(joinpath(output_directory, "GFunction_slices_$(case).png"), f2, px_per_unit = 3)
    save(joinpath(output_directory, "GFunction_slices_$(case).pdf"), f2, px_per_unit = 3)


end


main()
