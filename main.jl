# Self Landing Rocket Final Project Main.
# Authors: Carson Kohlbrenner, Thomas Dunnington, Owen Craig

cd(@__DIR__) # Change to the directory of the script

# Importing the necessary modules
using CommonRLInterface 
using Plots # For plotting the environment
using Images # For rendering the rocket

# Importing the environment
include("environment.jl")

# Environment parameters
x_max = 100.0 # m
y_max = 100.0 # m
dt = 0.1 # s

# Rocket parameters
thrust = 10.0 #kN
torque = 1.0 #kNm
m = 1000.0 #kg
I = 10.0 #kg*m^2

# Create a 2D rocket environment
env = RocketEnv2D([0.0, x_max, 0.0, y_max], dt, thrust, torque, m, I)
print_env(env)
render(env)

