# Self Landing Rocket Final Project Main.
# Authors: Carson Kohlbrenner, Thomas Dunnington, Owen Craig

cd(@__DIR__) # Change to the directory of the script

# Importing the necessary modules
using CommonRLInterface 
using Plots # For plotting the environment
using Images # For rendering the rocket
using POMDPTools: FunctionPolicy

# Importing the environment
include("environment_new.jl")
include("helperFuncs.jl")
include("DQN.jl")
include("PD_Heuristic.jl")
include("Behavior_Cloning.jl")

# Environment parameters
x_min = -500.0 # m
x_max = 500.0 # m
y_max = 1000.0 # m
dt = 0.1 # s
g = 9.81 # m/s^2

# Rocket parameters
m = 3000.0 #kg
h = 50.0 # m height of rocket
I = (1.0/12.0)*m*(h^2) #kg*m^2 using simple rod model
ϕ_max = pi/8.0 # rad

# Create a 2D rocket environment
env = RocketEnv2D([x_min, x_max, 0.0, y_max], dt, g, ϕ_max, m, I, h)
print_env(env)


# Test the render function
# total_plots, state_plots = render(env)
# display(state_plots)
# display(total_plots)

# Train a DQN model
#Q = DQN_Solve(env)


# Test NNet action approximator
inputData, outputData = episode!(env, heuristic_policy, 10000)
net = CloneExpert(env, heuristic_policy)

# Define basic policy
policy = state -> begin
    return 0.0
end

# Policy based on the neural network
function netPolicy(s)
    thrust = net(s)
    return convert(Float64, thrust[1])
end

# Evaluate policies
max_steps = 10000
numEps = 100
nothingReward = mean([simulate!(env, policy, max_steps) for _ in 1:numEps])
heuristicReward = mean([simulate!(env, heuristic_policy, max_steps) for _ in 1:numEps])
DPDReward = mean([simulate!(env, netPolicy, max_steps) for _ in 1:numEps])
print("Nothing Average Reward: ", nothingReward, '\n')
print("Heuristic Average Reward: ", heuristicReward, '\n')
print("DAgger Total Reward: ", DPDReward, '\n')


# Plot results
#render(env, policy, "Do Nothing")
#render(env, netPolicy, "DPD Function Approximation")



