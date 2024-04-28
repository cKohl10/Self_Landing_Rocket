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

# Environment parameters
x_min = -500.0 # m
x_max = 500.0 # m
y_max = 5000.0 # m
dt = 0.1 # s

# Rocket parameters
thrust = 200.0 * 10^3 #N
torque = 500.0 * 10^3 #Nm
m = 3000.0 #kg
h = 50.0 # m height of rocket
I = (1.0/12.0)*m*(h^2) #kg*m^2 using simple rod model

# Create a 2D rocket environment
env = RocketEnv2D([x_min, x_max, 0.0, y_max], dt, thrust, torque, m, I)
print_env(env)

# Calculate the gains for the PD controller heuristic
#print("Calculating Gains...\n")
#calculate_gains(env)

# Test the discrete translation of the heuristic policy
total_plots, state_plots = render(env)
display(total_plots)

# Test the continuous translation of the heuristic policy
total_plots, state_plots = render(env, heuristic_policy, "Heuristic PD Controller")
display(total_plots)

# Test the discrete translation of the PD controller
total_plots, state_plots = render(env, s->actions(env)[discrete_policy_distance_metric(s)], "PD to Discrete Controller")
display(state_plots)
display(total_plots)

# Train a DQN model
# Q = DQN_Solve(env)
# Q = DQN_Solve_Metric(env)

# Define basic policy
policy = state -> begin
    return [0.0, 0.0]
end


# Test simulate function
# max_steps = 1000
# total_reward = simulate!(env, policy, max_steps)
# print("Total Reward: ", total_reward)




