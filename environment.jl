# This script is where the environment for the problem will be defined

import CommonRLInterface

# Environment to work with CommonRLInterface
mutable struct RocketEnv2D

    #### Environment Parameters ####
    bounds::Vector{Float64} # Bounds of the environment (meters) [x_min, x_max, y_min, y_max]
    dt::Float64 # Time step for the simulation (seconds)
    target::Float64 # X Position of the target (meters)
    γ::Float64 # Discount factor
    ################################

    #### Rocket Parameters ####
    thrust::Float64 # Thrust of the rocket (kN)
    torque::Float64 # Torque of the rocket (kNm)
    m::Float64 # Mass of the rocket (kg)
    I::Float64 # Moment of Inertia of the rocket (kg*m^2)
    ####################

    #### State Space ####
    # [x, y, x_dot, y_dot, theta, theta_dot, time]
    # theta is defined as 0 when the rocket is upright
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
    x_target = env.target
    # Landing defined as hitting the ground
    if y <= 0.0
        # Reward for landing on the target
        reward = 100 * exp(-((x - x_target)^2) / (2 * 100^2))
        crash_vel = 5.0 #m/s

        # Add the reward for landing upright
        if abs(theta) < 0.1 && abs(theta_dot) < 0.1
            reward += 25
        end
        # Add the reward for low velocity
        if abs(x_dot) < crash_vel && abs(y_dot) < crash_vel
            reward += 25
        end
        return reward
    end
    # Return a negative reward for going out of bounds
    if x < env.bounds[1] || x > env.bounds[2] || y > env.bounds[4]*1.5
        return -200
    end

    # A step in the environment is a small negative reward
    reward = -1*env.dt

    return reward
end

# Constructor for the environment
function RocketEnv2D(bounds::Vector{Float64}, dt::Float64, thrust::Float64, torque::Float64, m::Float64, I::Float64)

    # Define the action space
    action_space = [[thrust, torque], [thrust, -torque], [thrust, 0.0], [0.0, torque], [0.0, -torque], [0.0, 0.0]]

    # Make the target in the middle of the environment
    target = ((bounds[2] - bounds[1]) / 2.0) + bounds[1]

    # Discount factor
    γ = 1.0

    R = RocketEnv2D(bounds, dt, target, γ, thrust, torque, m, I, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], action_space)

    CommonRLInterface.reset!(R) # Reset the environment's state

    return R
end


################ COMMON RL INTERFACE FUNCTIONS ################

# Function to initialize the environment and reset after landing
function CommonRLInterface.reset!(env::RocketEnv2D)
    bounds = env.bounds
    # Reset the environment to a random x position and the top of the y bounds, also random orientation
     ### Hyperparameters ###
     max_angle = pi/4.0 # Maximum angle of the rocket spawn
     max_x_dot = 5.0 # Maximum x velocity of the rocket spawn
     max_y_dot = 20.0 # Maximum y velocity of the rocket spawn
     
     # Initialize the state to the top of the environment
     width = (bounds[2] - bounds[1]) # Middle of the environment
     width_scale = 0.05 # Scale the width of spawn points
     #env.state = [rand_float(bounds[1], bounds[2]), bounds[4], rand_float(-max_x_dot, max_x_dot), rand_float(-max_y_dot, -max_y_dot*0.5), rand_float(-max_angle, max_angle), 0.0, 0.0]
     #env.state = [bounds[1]+width*width_scale, bounds[4], max_x_dot, -2*max_y_dot, max_angle, 0.0, 0.0]
     env.state = [rand_float(bounds[1]+width*width_scale, bounds[1]+width*2*width_scale), bounds[4] - rand_float(0.0, 20.0), max_x_dot + rand_float(-0.1*max_x_dot, 0.1*max_x_dot), -2*max_y_dot + rand_float(-0.1*max_y_dot, 0.1*max_y_dot), max_angle + rand_float(-0.05*max_angle, 0.05*max_angle), 0.0, 0.0]
end

function CommonRLInterface.reset!(env::RocketEnv2D, x, y)
    bounds = env.bounds
    # Reset the environment to a given x and y position
    env.state = [x, y, 0.0, 0.0, 0.0, 0.0, 0.0]
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
    if y <= 0.0 || y >= env.bounds[4]*1.5
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
    x, y, x_dot, y_dot, theta, theta_dot, t = env.state
    thrust, torque = action
    m = env.m

    # Update the state
    x += x_dot * env.dt
    y += y_dot * env.dt
    x_dot += (thrust * cos(theta + pi/2) / m) * env.dt
    y_dot += (thrust * sin(theta + pi/2) / m) * env.dt - 9.81 * env.dt
    theta += theta_dot * env.dt
    theta_dot += (torque / env.I)*env.dt
    t += env.dt

    # Update the state
    env.state = [x, y, x_dot, y_dot, theta, theta_dot, t]
    ##########################

    #### Check for Boundaries ####
    # If the rocket goes out of bounds, reset the environment and return a negative reward

    #### Return the reward ####
    return reward(env)
end

function CommonRLInterface.render(env::RocketEnv2D)

    println("Rendering the environment")

    # Get rocket Images
    #rocket = load("imgs/rocket.png")

    # Plot the rocket
    s = scatter([env.target], [0.0], label="Target", color="red")
    plot!(s, [env.bounds[1], env.bounds[2]], [0.0, 0.0], label="Ground", color="green", lw=2)
    #scatter!([x],[y], label="Rocket", color="blue", ms=10, marker=:circle)
    xlims!(s, env.bounds[1], env.bounds[2])
    ylims!(s, -200.0, env.bounds[4]*1.2)

    # Plot the target bounds
    plot!(s, [env.target - 0.1 * (env.bounds[2] - env.bounds[1]), env.target + 0.1 * (env.bounds[2] - env.bounds[1])], [0.0, 0.0], label=nothing, color="red", lw=2)

    # Simulate n trajectories and plot the results
    n = 10
    # Define the number of arrows to plot per trajectory
    num_arrows = 7
    arrow_scale = 1000.0

    function make_arrow!(arrow_scale, state)
        theta = state[5]
        # Calculate the magnitude of the velocity vector
        velocity_magnitude = sqrt(state[3]^2 + state[4]^2)
        # Normalize the velocity vector for 50 m/s
        velocity_magnitude_x = (velocity_magnitude * (env.bounds[2] - env.bounds[1])) / arrow_scale
        velocity_magnitude_y = (velocity_magnitude * (env.bounds[4] - env.bounds[3])) / arrow_scale

        u = velocity_magnitude_x * cos(theta + pi/2)
        v = velocity_magnitude_y * sin(theta + pi/2)
        quiver!(s, [state[1]], [state[2]], quiver=([u], [v]), color="black")
    end

    # Create a new plot for the states over time
    p = plot(layout=(5,1), size=(800, 600))

    # Define the range of points to Test
    x_width = env.bounds[2] - env.bounds[1]
    y_width = env.bounds[4] - env.bounds[3]
    A = CommonRLInterface.actions(env)

    for i in 1:n
        for j in 1:n
            # Simulate the trajectory
            state, total_reward, actions = simulate_trajectory!(env, s->A[discrete_policy_distance_metric(s)], 10000, (i/n)*x_width + env.bounds[1], (j/n)*y_width + env.bounds[3])

            # Break if state is empty
            if isempty(state)
                println("State is empty")
                break
            end

            x_traj = [s[1] for s in state]
            y_traj = [s[2] for s in state]

            # Plot the trajectory
            plot!(s, x_traj, y_traj, label=nothing, color=reward_to_color(total_reward), lw=2)

            # Calculate the interval at which to plot the arrows
            interval = Int(ceil(length(x_traj) / num_arrows))

            # Plot the arrows at regular intervals along the trajectory
            for i in 1:interval:length(x_traj)
                make_arrow!(arrow_scale, state[i])
            end

            # Make an arrow at the end of the trajectory
            make_arrow!(arrow_scale, state[end])

            # Plot the state over time
            p = state_plot(p, state, actions, reward_to_color(total_reward))
        end
    end

    return s, p
end

function CommonRLInterface.render(env::RocketEnv2D, policy::Function, title::String)

    println("Rendering the environment")

    # Get rocket Images
    #rocket = load("imgs/rocket.png")

    # Plot the rocket
    s = scatter([env.target], [0.0], label="Target", color="red")
    plot!(s, [env.bounds[1], env.bounds[2]], [0.0, 0.0], label="Ground", color="green", lw=2)
    #scatter!([x],[y], label="Rocket", color="blue", ms=10, marker=:circle)
    xlims!(s, env.bounds[1], env.bounds[2])
    ylims!(s, -200.0, env.bounds[4]*1.2)

    # title
    title!(s, title)

    # Plot the target bounds
    plot!(s, [env.target - 0.1 * (env.bounds[2] - env.bounds[1]), env.target + 0.1 * (env.bounds[2] - env.bounds[1])], [0.0, 0.0], label=nothing, color="red", lw=2)

    # Simulate n trajectories and plot the results
    n = 1
    # Define the number of arrows to plot per trajectory
    num_arrows = 7
    arrow_scale = 2000.0

    function make_arrow!(arrow_scale, state)
        theta = state[5]
        # Calculate the magnitude of the velocity vector
        velocity_magnitude = sqrt(state[3]^2 + state[4]^2)
        # Normalize the velocity vector for 50 m/s
        velocity_magnitude_x = (velocity_magnitude * (env.bounds[2] - env.bounds[1])) / arrow_scale
        velocity_magnitude_y = (velocity_magnitude * (env.bounds[4] - env.bounds[3])) / arrow_scale

        u = velocity_magnitude_x * cos(theta + pi/2)
        v = velocity_magnitude_y * sin(theta + pi/2)
        quiver!(s, [state[1]], [state[2]], quiver=([u], [v]), color="black")
    end

    # Create a new plot for the states over time
    p = plot(layout=(5,1), size=(800, 600))

    for i in 1:n
        # Simulate the trajectory
        state, total_reward, actions = simulate_trajectory!(env, policy, 10000)

        x_traj = [s[1] for s in state]
        y_traj = [s[2] for s in state]

        # Plot the trajectory
        plot!(s, x_traj, y_traj, label=nothing, color=reward_to_color(total_reward), lw=2)

        # Calculate the interval at which to plot the arrows
        interval = Int(ceil(length(x_traj) / num_arrows))

        # Plot the arrows at regular intervals along the trajectory
        for i in 1:interval:length(x_traj)
            make_arrow!(arrow_scale, state[i])
        end

        # Make an arrow at the end of the trajectory
        make_arrow!(arrow_scale, state[end])

        # Plot the state over time
        p = state_plot(p, state, actions, reward_to_color(total_reward))
    end

    return s, p
end

################################################################

################# Simulation Functions #########################
function simulate_trajectory!(env::RocketEnv2D, policy::Function, max_steps::Int)
    # Initialize the total reward
    total_reward = 0.0

    # Get the discount factor
    γ = env.γ

    # Reset the environment
    CommonRLInterface.reset!(env)
    s = CommonRLInterface.observe(env)

    #println("Initial State in Trajectory: $(round.(s; digits=2))")

    # Append s to a list of all states over the entire simulation
    states = [s]
    actions = []

    # Loop through the simulation
    for i in 1:max_steps
        # Get the action from the policy
        s = CommonRLInterface.observe(env)
        a = policy(s)
        push!(actions, a)

        # Step in the environment
        CommonRLInterface.act!(env, a)

        # Append the state to the list of states
        push!(states, CommonRLInterface.observe(env))

        # Get the reward
        r = reward(env)

        # Add the reward to the total
        total_reward += γ^i * r

        # Check if the environment is terminated
        if CommonRLInterface.terminated(env)
            break
        end
    end 

    #println("Total Reward: ", total_reward)
    #print("Final State in Trajectory: ", round.(env.state; digits=2), "\n \n")

    return states, total_reward, actions
end

# Simulate starting from a specified x and y position
function simulate_trajectory!(env::RocketEnv2D, policy::Function, max_steps::Int, x::Float64, y::Float64)
    # Initialize the total reward
    total_reward = 0.0

    # Get the discount factor
    γ = env.γ

    # Reset the environment
    CommonRLInterface.reset!(env, x, y)
    s = CommonRLInterface.observe(env)

    #println("Initial State in Trajectory: $(round.(s; digits=2))")

    # Append s to a list of all states over the entire simulation
    states = [s]
    actions = []

    # Loop through the simulation
    for i in 1:max_steps
        # Get the action from the policy
        s = CommonRLInterface.observe(env)
        a = policy(s)
        push!(actions, a)

        # Step in the environment
        CommonRLInterface.act!(env, a)

        # Append the state to the list of states
        push!(states, CommonRLInterface.observe(env))

        # Get the reward
        r = reward(env)

        # Add the reward to the total
        total_reward += γ^i * r

        # Check if the environment is terminated
        if CommonRLInterface.terminated(env)
            break
        end
    end 

    #println("Total Reward: ", total_reward)
    #print("Final State in Trajectory: ", round.(env.state; digits=2), "\n \n")

    return states, total_reward, actions
end

function simulate!(env::RocketEnv2D, policy::Function, max_steps::Int)
    # Initialize the total reward
    total_reward = 0.0

    # Get the discount factor
    γ = env.γ

    # Reset the environment
    CommonRLInterface.reset!(env)
    s = CommonRLInterface.observe(env)

    # Loop through the simulation
    for i in 1:max_steps
        # Get the action from the policy
        a = policy(s)

        # Step in the environment
        CommonRLInterface.act!(env, a)

        # Get the reward
        r = reward(env)

        # Add the reward to the total
        total_reward += γ^i * r

        # Check if the environment is terminated
        if CommonRLInterface.terminated(env)
            break
        end
    end 
    return total_reward
end


# Simulate and return a history for an episode
function episode!(env::RocketEnv2D, policy::Function, max_steps::Int)
    inputData = []
    outputData = []
    # Loop through n steps in the environment and add to the buffer
    reset!(env)
    for i = 1:max_steps
        done = terminated(env)
        if done  # Break if a terminal state is reached
            break
        end
        s = observe(env)
        thrust, torque = policy(s)
        r = act!(env, [thrust, torque])
        push!(inputData, s)                     # State Data
        push!(outputData, [thrust, torque])     # Output thrust and torque
    end
    return inputData, outputData
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