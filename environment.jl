# This script is where the environment for the problem will be defined

import CommonRLInterface

# Environment to work with CommonRLInterface
mutable struct RocketEnv2D

    #### Parameters ####
    # Bounds of the environment (meters) [x_min, x_max, y_min, y_max]
    bounds::Vector{Float64}

    # Thrust and torque of the rocket (scalar in N and N*m)
    thrust::Float64
    torque::Float64

    # Time step for the simulation (seconds)
    dt::Float64
    ####################

    #### State Space ####
    # [x, y, x_dot, y_dot, theta, theta_dot]
    state::Vector{Float64}

    #### Action Space ####
    # [Thrust On, Torque CCW]
    # [Thrust On, Torque CW]
    # [Thrust On, No Torque]
    # [Thrust Off, Torque CCW]
    # [Thrust Off, Torque CW]
    # [Thrust Off, No Torque]
    action_space::Vector{Vector{Float64}}

    #### Rewards ####
    # Reward for landing upright, low velocity, and on the target
    reward::Function

end

# Define the reward function for the environment
function reward(env::RocketEnv2D)
    # Unpack the state
    x, y, x_dot, y_dot, theta, theta_dot = env.state

    # Landing defined as hitting the ground
    if y <= 0.0
        # Add the reward for landing on the target

        # Add the reward for landing upright

        # Add the reward for low velocity

        return reward

    end

    # Return a negative reward for going out of bounds

    # A step in the environment is a small negative reward
    reward = -0.1

    return reward
end

# Constructor for the environment
function RocketEnv2D(bounds::Vector{Float64}, thrust::Float64, torque::Float64, dt::Float64)
    
    # Initialize the state to the top of the environment
    state = [rand(bounds[1]:bounds[2]), bounds[4], 0.0, 0.0, 0.0, 0.0]

    # Define the action space
    action_space = [[thrust, torque], [thrust, -torque], [thrust, 0.0], [0.0, torque], [0.0, -torque], [0.0, 0.0]]

    return RocketEnv2D(bounds, thrust, torque, dt, state, action_space)
end

################ COMMON RL INTERFACE FUNCTIONS ################

# Function to initialize the environment and reset after landing
function CommonRLInterface.reset!(env::RocketEnv2D)
    # Reset the environment to a random x position and the top of the y bounds
    env.state = [rand(env.bounds[1]:env.bounds[2]), env.bounds[4], 0.0, 0.0, 0.0, 0.0 ]
end

# Returns the actions in the environment
function CommonRLInterface.actions(env::RocketEnv2D)
    return env.action_space
end

# Function to observe the state of the environment
function CommonRLInterface.observe(env::RocketEnv2D)
    return env.state
end

# Function to step the environment
# This is a numerical integrator for the rocket dynamics
# Make sure to specify a good timestep for the simulation
function CommonRLInterface.act!(env::RocketEnv2D, action::Vector{Float64})

    #### Update the State ####
    # Unpack the state and action
    x, y, x_dot, y_dot, theta, theta_dot = env.state
    thrust, torque = action

    # Update the state
    x += x_dot * env.dt
    y += y_dot * env.dt
    x_dot += (thrust * cos(theta) / 1.0) * env.dt
    y_dot += (thrust * sin(theta) / 1.0) * env.dt
    theta_dot += torque * env.dt

    # Update the state
    env.state = [x, y, x_dot, y_dot, theta, theta_dot]
    ##########################

    #### Check for Boundaries ####
    # If the rocket goes out of bounds, reset the environment and return a negative reward

    #### Return the reward ####
end

################################################################

################# Custom Functions #############################

# Print the variables in the environment
function print_env(env::RocketEnv2D)
    print(" --- 2D Rocket Environment ---\n")
    println("Bounds: ", env.bounds, " m")
    println("Thrust: ", env.thrust, " N")
    println("Torque: ", env.torque, " Nm")
    println("dt: ", env.dt, " s \n")
    println("State: ", env.state)
    println("[x, y, x_dot, y_dot, theta, theta_dot]\n")

    println("Action Space: ", env.action_space)
    println("[Thrust, Torque]\n")
end