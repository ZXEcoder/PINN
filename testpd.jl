########################################
# 1. Libraries and Setup
########################################
using NeuralPDE
import ModelingToolkit: Interval
using Optimization, OptimizationOptimisers
using Lux, LuxCUDA, ComponentArrays
using Printf, Random
using Plots

const gpud = gpu_device()  # Allocate GPU device (if available)

########################################
# 2. Define PDE Problem (1D Wave Equation)
########################################
@parameters x t
@variables u(..)

# Differential operators
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt  = Differential(t)

# Domain bounds
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0  # You can increase t_max for more wave dynamics

# Wave speed
c = 1.0

# PDE: u_tt = c² * u_xx
eq = Dtt(u(x, t)) - c^2 * Dxx(u(x, t)) ~ 0

# Boundary and initial conditions
bcs = [
    u(0.0, t) ~ 0.0,                  # u(0, t) = 0
    u(1.0, t) ~ 0.0,                  # u(1, t) = 0
    u(x, 0.0) ~ x * (1.0 - x),        # u(x, 0) = x(1 - x)
    Dt(u(x, 0.0)) ~ 0.0               # u_t(x, 0) = 0
]

# Domain definitions
domains = [
    t ∈ Interval(t_min, t_max),
    x ∈ Interval(x_min, x_max)
]

@named pde_system = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])


########################################
# 3. Define Neural Network Architecture
########################################
glorot_init = Lux.glorot_uniform
inner = 100  # Number of neurons per hidden layer

# Define MLP architecture (initially can use dropout until loss is ~0.02, then remove it)
chain = Chain(
    Dense(2, inner, σ; init_weight=glorot_init),   # Input layer
    # Dropout(0.1),                                # Dropout for early training only (optional)
    Dense(inner, inner, σ; init_weight=glorot_init), # Hidden layer 1
    # Dropout(0.1),
    Dense(inner, inner, σ; init_weight=glorot_init), # Hidden layer 2
    # Dropout(0.1),
    Dense(inner, inner, σ; init_weight=glorot_init), # Hidden layer 3
    # Dropout(0.1),
    Dense(inner, 1; init_weight=glorot_init)         # Output layer
)

# Initialize network parameters
ps = Lux.setup(Random.default_rng(), chain)[1]
ps = ps |> ComponentArray |> gpud .|> Float64


########################################
# 4. Discretization and Problem Setup
########################################
strategy = GridTraining(0.1)  # Grid spacing; decrease to 0.01 for finer training
discretization = PhysicsInformedNN(chain, strategy, init_params = ps)
prob = discretize(pde_system, discretization)

# Callback to monitor training
callback = function (state, l)
    println("Epoch $(state.iter): Current loss is $l")
    return false
end


########################################
# 5. Training Phases
########################################

# Phase 1: Coarse training with higher learning rate
res = Optimization.solve(prob, Adam(0.01); callback = callback, maxiters = 50000)

# Phase 2: Fine-tuning with lower learning rate (retrain without dropout if used above)
prob = remake(prob, u0 = res.u)
res = Optimization.solve(prob, Adam(0.001); callback = callback, maxiters = 50000)


########################################
# 6. Prediction and Visualization
########################################

# Network approximation function (phi)
phi = discretization.phi
ts, xs = [infimum(d.domain):0.01:supremum(d.domain) for d in domains]

# Analytical solution using eigenfunction expansion
function true_solution(x, t; nmax=20)
    u_true = 0.0
    for n in 1:2:nmax  # Only odd terms contribute
        a_n = 8 / ((n * π)^3)
        u_true += a_n * cos(n * π * t) * sin(n * π * x)
    end
    return u_true
end

# Function to animate predicted vs. true solution
function plot_solution(res)
    anim = @animate for (i, t) in enumerate(ts)
        @info "Animating frame $i..."

        # Evaluate predicted and true solutions
        u_pred = [first(Array(phi([x, t], res.u))) for x in xs]
        u_true = [true_solution(x, t) for x in xs]

        # Plotting
        title_str = "Solutions at t = " * @sprintf("%.2f", t)
        p = plot(xs, u_pred,
                 label = "Predicted",
                 lw = 2,
                 ylim = (0, 0.3),
                 xlabel = "x",
                 ylabel = "u",
                 title = title_str)
        plot!(p, xs, u_true,
              label = "True",
              lw = 2,
              ls = :dash)
        display(p)
    end
    gif(anim, "wave2d.gif", fps = 20)
end

# Generate the animated GIF
print(plot_solution(res))
