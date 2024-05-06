# Self Landing Rocket Final Project Main.
# Authors: Carson Kohlbrenner, Thomas Dunnington, Owen Craig
# Overleaf: https://www.overleaf.com/8254649933hsqnfwkjcfdj#764db9

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
#include("Owen_gpu_test.jl")
include("PD_Heuristic.jl")

# Environment parameters
x_min = -100.0 # m
x_max = 100.0 # m
y_max = 2000.0 # m
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

# Calculate the gains for the PD controller heuristic
#print("Calculating Gains...\n")
#calculate_gains(env)

# Test the discrete translation of the heuristic policy
# total_plots, state_plots = render(env)
# display(total_plots)

# # Test the continuous translation of the heuristic policy
# println("Heuristic PD Controller Avg Reward: ", eval(env, heuristic_policy, 10000))
# total_plots, state_plots = render(env, heuristic_policy, "Heuristic PD Controller", 10)
# display(state_plots)
# display(total_plots)

# # # Test the discrete translation of the PD controller
# total_plots, state_plots = render(env, s->actions(env)[discrete_policy_distance_metric(s)], "PD to Discrete Controller")
# display(state_plots)
# display(total_plots)

# # Save the output of a single PD to Discrete Controller
# print("Saving PD to Discrete Controller Output in CSV...\n")
# total_plots, state_plots = render_and_save(env, s->actions(env)[discrete_policy_distance_metric(s)], "PD2Disc")
# display(state_plots)
# display(total_plots)

# Train a DQN model
# Q = DQN_Solve(env)
Q = DQN_Solve_Metric(env)

# Define basic policy
# policy = state -> begin
#     return [0.0, 0.0]
# end


# Test simulate function
# max_steps = 1000
# total_reward = simulate!(env, policy, max_steps)
# print("Total Reward: ", total_reward)


#Display Best Q network generated
# file_path = "E:/owenc/Documents/Self_Landing_Rocket/models/Q_discrete_metric_5535_best_reward"
# file_path = "/Users/owencraig/Documents/GitHub/Self_Landing_Rocket/models/Q_discrete_metric_5535_best_reward"
# Q_best = BSON.load(file_path)[:Q]
# Qpolicy = s->actions(env)[argmax(Q_best(s))]
# s,p = render(env, Qpolicy, "Q_best", 10)
# display(p) # Display the state plots
# display(s) # Display the inertial path plot
# # savefig(s,"DQN_Traj.png")
# # savefig(p,"DQN_States.png")
# # Define policy from the Q values, take the maximum Q value for a given state
# reward = [];
# for i in 1:10000
#     reset!(env)
#     r = simulate!(env, Qpolicy, 10000)
#     push!(reward, r)
#     if i % 100 == 0
#         println("Episode: ", i)
#     end
# end
# avg_reward = mean(reward)
# sem_DQN = std(reward)/sqrt(length(reward))
# println("Average Reward: ", avg_reward)
# println("SEM: ", sem_DQN)