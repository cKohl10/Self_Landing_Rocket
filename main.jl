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

# Environment parameters
x_max = 1000.0 # m
y_max = 5000.0 # m
dt = 0.1 # s

# Rocket parameters
thrust = 100.0 * 10^3 #kN
torque = 5.0 * 10^3 #kNm
m = 3000.0 #kg
h = 50.0 # m height of rocket
I = (1.0/12.0)*m*(h^2) #kg*m^2 using simple rod model

# Create a 2D rocket environment
env = RocketEnv2D([0.0, x_max, 0.0, y_max], dt, thrust, torque, m, I)
print_env(env)

# Test the render function
total_plots, state_plots = render(env)
display(state_plots)
display(total_plots)

# Define basic policy
policy = state -> begin
    return [0.0, 0.0]
end

# Test simulate function
max_steps = 1000
total_reward = simulate!(env, policy, max_steps)
print("Total Reward: ", total_reward)








