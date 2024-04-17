# This script is where the environment for the problem will be defined

import CommonRLInterface

# Environment to work with CommonRLInterface
mutable struct RocketEnv2D

    #### Environment Parameters ####
    bounds::Vector{Float64} # Bounds of the environment (meters) [x_min, x_max, y_min, y_max]
    dt::Float64 # Time step for the simulation (seconds)
    target::Float64 # X Position of the target (meters)
    ################################

    #### Rocket Parameters ####
    thrust::Float64 # Thrust of the rocket (kN)
    torque::Float64 # Torque of the rocket (kNm)
    m::Float64 # Mass of the rocket (kg)
    I::Float64 # Moment of Inertia of the rocket (kg*m^2)
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

end

# Define the reward function for the environment
function reward(env::RocketEnv2D)
    # Unpack the state
    x, y, x_dot, y_dot, theta, theta_dot = env.state

    # Landing defined as hitting the ground
    if y <= 0.0
        base_reward = 100.0
        crash_vel = 2.0 #m/s

        # Add the reward for landing on the target (Normalized by the bounds of the environment)
        reward = base_reward * (1 - (abs(x - env.target) / (env.bounds[2] - env.bounds[1]))^2)

        # Add the reward for landing upright
        if abs(theta) < 0.1
            reward += base_reward
        end

        # Add the reward for low velocity
        if abs(x_dot) < crash_vel && abs(y_dot) < crash_vel
            reward += base_reward
        end

        return reward

    end

    # Return a negative reward for going out of bounds
    if x < env.bounds[1] || x > env.bounds[2]
        return -100.0
    end

    # A step in the environment is a small negative reward
    reward = -0.1

    return reward
end

# Constructor for the environment
function RocketEnv2D(bounds::Vector{Float64}, dt::Float64, thrust::Float64, torque::Float64, m::Float64, I::Float64)
    
    # Initialize the state to the top of the environment
    state = [rand(bounds[1]:bounds[2]), bounds[4], 0.0, 0.0, 0.0, 0.0]

    # Define the action space
    action_space = [[thrust, torque], [thrust, -torque], [thrust, 0.0], [0.0, torque], [0.0, -torque], [0.0, 0.0]]

    # Make the target in the middle of the environment
    target = (bounds[2] - bounds[1]) / 2.0

    return RocketEnv2D(bounds, dt, target, thrust, torque, m, I, state, action_space)
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

# Function to check if the environment is in a terminal state
function CommonRLInterface.terminated(env::RocketEnv2D)
    # Unpack the state
    x, y, x_dot, y_dot, theta, theta_dot = env.state

    # Check if the rocket has landed
    if y <= 0.0
        return true
    end

    # Check if the rocket has gone out of bounds
    if x < env.bounds[1] || x > env.bounds[2]
        return true
    end

    return false
end

# Function to step the environment
# This is a numerical integrator for the rocket dynamics
# Make sure to specify a good timestep for the simulation
function CommonRLInterface.act!(env::RocketEnv2D, action::Vector{Float64})

    #### Update the State ####
    # Unpack the state and action
    x, y, x_dot, y_dot, theta, theta_dot = env.state
    thrust, torque = action
    m = env.m

    # Update the state
    x += x_dot * env.dt
    y += y_dot * env.dt
    x_dot += (thrust * cos(theta) / m) * env.dt
    y_dot += (thrust * sin(theta) / m) * env.dt - 9.8 * env.dt
    theta += theta_dot * env.dt
    theta_dot += (torque / env.I) * env.dt

    # Update the state
    env.state = [x, y, x_dot, y_dot, theta, theta_dot]
    ##########################

    #### Check for Boundaries ####
    # If the rocket goes out of bounds, reset the environment and return a negative reward

    #### Return the reward ####
end

function CommonRLInterface.render(env::RocketEnv2D)
    # Unpack the state
    x, y, x_dot, y_dot, theta, theta_dot = env.state

    # Get rocket Images
    rocket = load("imgs/rocket.png")

    # Plot the rocket
    scatter([env.target], [0.0], label="Target", color="red")
    plot!([env.bounds[1], env.bounds[2]], [0.0, 0.0], label="Ground", color="green", lw=2)
    scatter!([x],[y], label="Rocket", color="blue", ms=10, marker=:circle)
    xlims!(env.bounds[1], env.bounds[2])
    ylims!(0.0, env.bounds[4])
end

################################################################

################# Simulation Functions #########################
# function simulate!(env::RocketEnv2D, policy::FunctionPolicy, max_steps::Int)
#     # Initialize the total reward
#     total_reward = 0.0

#     # Reset the environment
#     CommonRLInterface.reset!(env)
#     s = CommonRLInterface.observe(env)

#     # Loop through the simulation
#     for i in 1:max_steps
#         # Get the action from the policy
#         a = policy(s)

#         # Step in the environment
#         CommonRLInterface.act!(env, a)

#         # Get the reward
#         reward = reward(env)

#         # Add the reward to the total
#         total_reward += reward

#         # Check if the environment is terminated
#         if CommonRLInterface.terminated(env)
#             break
#         end
#     end

#     return total_reward
# end

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