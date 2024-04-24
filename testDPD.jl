# Self Landing Rocket Final Project Main.
# Authors: Carson Kohlbrenner, Thomas Dunnington, Owen Craig

cd(@__DIR__) # Change to the directory of the script

# Importing the necessary modules
using CommonRLInterface 
using Plots # For plotting the environment
using Images # For rendering the rocket
using POMDPTools: FunctionPolicy

# Importing the environment
include("environment.jl")
include("helperFuncs.jl")
include("DQN.jl")
include("PD_Heuristic.jl")
include("DQN_continous.jl")

# Environment parameters
x_min = -500.0 # m
x_max = 500.0 # m
y_max = 5000.0 # m
dt = 0.1 # s

# Rocket parameters
thrust = 30.0 * 10^3 #N
torque = 100.0 * 10^3 #Nm
m = 3000.0 #kg
h = 50.0 # m height of rocket
I = (1.0/12.0)*m*(h^2) #kg*m^2 using simple rod model

# Create a 2D rocket environment
env = RocketEnv2D([x_min, x_max, 0.0, y_max], dt, thrust, torque, m, I)
print_env(env)

# Calculate the gains for the PD controller heuristic
print("Calculating Gains...\n")
calculate_gains(env)

# Test the render function
# total_plots, state_plots = render(env)
# display(state_plots)
# display(total_plots)

# Train a DQN model
#Q = DQN_Solve(env)


# Test NNet action approximator
inputData, outputData = episode!(env, heuristic_policy, 10000)
net = DPD_Continuous(env, heuristic_policy)

# Define basic policy
policy = state -> begin
    return [0.0, 0.0]
end

# Policy based on the neural network
function netPolicy(s)
    return convert(Vector{Float64}, net(s))
end


# Evaluate policies
max_steps = 10000
numEps = 100
nothingReward = mean([simulate!(env, policy, max_steps) for _ in 1:numEps])
heuristicReward = mean([simulate!(env, heuristic_policy, max_steps) for _ in 1:numEps])
DPDReward = mean([simulate!(env, policy, max_steps) for _ in 1:numEps])
print("Nothing Average Reward: ", nothingReward, '\n')
print("Heuristic Average Reward: ", heuristicReward, '\n')
print("DPD Total Reward: ", DPDReward, '\n')


# Plot results
#render(env, policy, "Do Nothing")
#render(env, netPolicy, "DPD Function Approximation")



