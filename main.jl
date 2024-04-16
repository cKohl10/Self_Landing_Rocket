# Self Landing Rocket Final Project Main.
# Authors: Carson Kohlbrenner, Thomas Dunnington, Owen Craig

# Importing the necessary modules
import CommonRLInterface

# Importing the environment
include("environment.jl")

# Create a 2D rocket environment
x_max = 100.0
y_max = 100.0
thrust = 10.0
torque = 1.0
dt = 0.1

env = RocketEnv2D([0, x_max, 0, y_max], thrust, torque, dt);
print_env(env)


