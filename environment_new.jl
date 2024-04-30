# This script is where the environment for the problem will be defined

import CommonRLInterface
using DataFrames
using CSV

# Environment to work with CommonRLInterface
mutable struct RocketEnv2D

    #### Environment Parameters ####
    bounds::Vector{Float64} # Bounds of the environment (meters) [x_min, x_max, y_min, y_max]
    dt::Float64 # Time step for the simulation (seconds)
    target::Float64 # X Position of the target (meters)
    γ::Float64 # Discount factor
    g::Float64 # Acceleration due to gravity (m/s^2)
    ################################

    #### Rocket Parameters ####
    ϕ_max::Float64 # Maximum angle the rocket can vector its thrust (radians)
    m::Float64 # Mass of the rocket (kg)
    I::Float64 # Moment of Inertia of the rocket (kg*m^2)
    h::Float64 # Height of the rocket (meters)
    gains::Vector{Float64} # Gains for the PD controller [k1_rot, k2_rot, k3_rot, k4_rot, k1_thrust, k2_thrust]
    ####################

    #### State Space ####
    # [x, y, x_dot, y_dot, theta, theta_dot, time]
    # theta is defined as 0 when the rocket is upright
    state::Vector{Float64}

    #### Action Space ####
    # [100% Thrust (2*mg), 50% Thrust (mg), 10% Thrust (0.1*mg), 0% Thrust]
    action_space::Vector{Float64}
    landed::Bool # Boolean to check if the rocket has landed

end

# Constructor for the environment
function RocketEnv2D(bounds::Vector{Float64}, dt::Float64, g::Float64, ϕ_max::Float64, m::Float64, I::Float64, h::Float64)

    ###### Environment Parameters ######
    action_space = [2*m*g, m*g, 0.2*m*g, 0.0]

    # Make the target in the middle of the environment
    target = ((bounds[2] - bounds[1]) / 2.0) + bounds[1]

    # Discount factor
    γ = 0.999

    ####### Rotational Gains #######
    τ_1 = 2 # Settling time of mode 1 in seconds
    τ_2 = 0.2 # Settling time of mode 2 in seconds
    λ_1 = -1/τ_1
    λ_2 = -1/τ_2
    k1_rot = (λ_1 * λ_2)*I
    k2_rot = -(λ_1 + λ_2)*I
    k3_rot = -2 * 10^5
    k4_rot = 0.02

    ####### Thrust Gains #######
    τ_1th = 20 # Settling time of mode 1 in seconds
    τ_2th = 1 # Settling time of mode 2 in seconds
    λ_1th = -1/τ_1th
    λ_2th = -1/τ_2th
    k1_thrust = (λ_1th * λ_2th)*m
    k2_thrust = -(λ_1th + λ_2th)*m

    gains = [k1_rot, k2_rot, k3_rot, k4_rot, k1_thrust, k2_thrust]


    R = RocketEnv2D(bounds, dt, target, γ, g, ϕ_max, m, I, h, gains, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], action_space, false)

    CommonRLInterface.reset!(R) # Reset the environment's state

    return R
end


################ COMMON RL INTERFACE FUNCTIONS ################

# Define the reward function for the environment
function reward!(env::RocketEnv2D)
    # Unpack the state
    x, y, x_dot, y_dot, theta, theta_dot = env.state
    x_target = env.target

    #### Hyperparameters ####
    crash_vel = 20.0 #m/s
    crash_theta = 10 * (pi/180) #radians
    missed_target = 100.0 #meters

    # Landing defined as hitting the ground
    if y <= 0.0
        # Reward for landing on the target
        reward = 15 * exp(-((x - x_target)^2) / (env.bounds[2] - env.bounds[1]))
        if abs(x - x_target) < missed_target
            reward += 50
        end

        # Add the reward for landing upright
        reward += 50 * exp(-pi*abs(theta))
        if abs(theta) < crash_theta && abs(theta_dot) < crash_theta
            reward += 25
        end
        # Add the reward for low velocity
        reward = 40 * exp(-(sqrt(x_dot^2 + y_dot^2) / (crash_vel)))
        if abs(x_dot) < crash_vel && abs(y_dot) < crash_vel
            reward += 25
        end

        env.landed = true
        reward = reward * 2000
        return reward
    end
    # Return a negative reward for going out of bounds
    if x < env.bounds[1] || x > env.bounds[2] || y > env.bounds[4]*1.5
        env.landed = true
        return -200
    end

    # A step in the environment is a small negative reward
    reward = -1*env.dt

    return reward
end

# function reward!(env::RocketEnv2D)
#     # Unpack the state
#     x, y, x_dot, y_dot, theta, theta_dot = env.state
#     x_target = env.target
#     # Landing defined as hitting the ground
#     if y <= 0.0
#         # Reward for landing on the target
#         reward = 100 * exp(-((x - x_target)^2) / (2 * 100^2))
#         crash_vel = 5.0 #m/s

#         # Add the reward for landing upright
#         if abs(theta) < 0.1 && abs(theta_dot) < 0.1
#             reward += 25
#         end
#         # Add the reward for low velocity
#         if abs(x_dot) < crash_vel && abs(y_dot) < crash_vel
#             reward += 25
#         end
#         return reward
#     end
#     # Return a negative reward for going out of bounds
#     if x < env.bounds[1] || x > env.bounds[2] || y > env.bounds[4]*1.5
#         return -200
#     end

#     # A step in the environment is a small negative reward
#     reward = -1*env.dt

#     return reward
# end

# Function to initialize the environment and reset after landing
function CommonRLInterface.reset!(env::RocketEnv2D)
    bounds = env.bounds
    # Reset the environment to a random x position and the top of the y bounds, also random orientation
     ### Hyperparameters ###
     max_angle = pi/4.0 # Maximum angle of the rocket spawn
     max_x_dot = 5.0 # Maximum x velocity of the rocket spawn
     max_y_dot = 200.0 # Maximum y velocity of the rocket spawn

     env.landed = false
     
     # Initialize the state to the top of the environment
     width = (bounds[2] - bounds[1]) # Middle of the environment
     width_scale = 0.05 # Scale the width of spawn points
     
     # Randomize the spawn point
     env.state = [rand_float(bounds[1], bounds[2]), bounds[4], rand_float(-max_x_dot, max_x_dot), rand_float(-max_y_dot, -max_y_dot*0.5), rand_float(-max_angle, max_angle), 0.0, 0.0]
     
     #env.state = [bounds[1]+width*width_scale, bounds[4], max_x_dot, -2*max_y_dot, max_angle, 0.0, 0.0]
     
     # Small randomization of the spawn point
     #env.state = [rand_float(bounds[1]+width*width_scale, bounds[1]+width*2*width_scale), bounds[4] - rand_float(0.0, 20.0), max_x_dot + rand_float(-0.1*max_x_dot, 0.1*max_x_dot), -2*max_y_dot + rand_float(-0.1*max_y_dot, 0.1*max_y_dot), max_angle + rand_float(-0.05*max_angle, 0.05*max_angle), 0.0, 0.0]
end

# Incase we want to reset the environment to a specific x and y position
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
    x, y = env.state

    # Check if the rocket has landed
    if env.landed
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
function CommonRLInterface.act!(env::RocketEnv2D, action::Float64)

    #### Update the State ####
    # Unpack the state and action
    x, y, x_dot, y_dot, theta, theta_dot, t = env.state
    k1_rot, k2_rot, k3_rot, k4_rot = env.gains
    m = env.m # Mass of the rocket
    h = env.h # Height of the rocket
    thrust = action


    ####### Controlling the torque of the rocket #######
    if thrust > 0.0
        # Use a PD controller to determine the best vectoring angle ϕ
        torque_r = -k1_rot*theta - k2_rot*theta_dot - k3_rot*x_dot - k4_rot*k3_rot*x

        # We are assuming the thrust moment arm is exactly half of the rocket height
        ϕ_max = env.ϕ_max

        # Check if the torque is within the limits of the rocket
        if abs(torque_r) > thrust*(h/2)*sin(ϕ_max)
            # If the torque is too high, set the vectoring angle to the maximum value
            ϕ = sign(torque_r)*ϕ_max
        else 
            # Desired angle of the thrust vector
            ϕ = asin((2*torque_r)/(h*thrust)) 
        end

        torque = thrust * sin(ϕ) * (h/2)
    else
        # If the thrust is zero, the torque and vectoring angle are zero
        torque = 0.0
    end

    ####### Update the State #######
    # Update the state
    x += x_dot * env.dt
    y += y_dot * env.dt
    x_dot += (thrust * cos(theta + pi/2) / m) * env.dt
    y_dot += (thrust * sin(theta + pi/2) / m) * env.dt - env.g * env.dt
    theta += theta_dot * env.dt
    theta_dot += (torque / env.I)*env.dt
    t += env.dt

    # Update the state
    env.state = [x, y, x_dot, y_dot, theta, theta_dot, t]
    ##########################

    #### Check for Boundaries ####
    # If the rocket goes out of bounds, reset the environment and return a negative reward

    #### Return the reward ####
    return reward!(env)
end

function CommonRLInterface.render(env::RocketEnv2D, policy::Function, title::String, n::Int=10, max_steps::Int=10000, save::Bool=false)

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
    # Define the number of arrows to plot per trajectory
    num_arrows = 7
    arrow_scale = 2000.0 # Scale the arrows for the velocity

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
    p = plot(layout=(6,1), size=(800, 800))

    for i in 1:n
        # Simulate the trajectory
        if save
            state, total_reward, actions = simulate_trajectory_and_save!(env, policy, max_steps, title * "_$(i)")
        else
            state, total_reward, actions = simulate_trajectory!(env, policy, max_steps)
        end
        all_actions = [] # Back out the vectoring angles and torque
        for i in 1:length(actions)
            push!(all_actions, torque_controls(env, state[i], actions[i]))
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
        p = state_plot(p, state, all_actions, reward_to_color(total_reward))
    end

    return s, p
end

render_and_save(env::RocketEnv2D, policy::Function, title::String, n::Int=1, max_steps::Int=10000) = CommonRLInterface.render(env, policy, title, n, max_steps, true)
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
        r = reward!(env)

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

# Saves the output of the simulation as a CSV file
function simulate_trajectory_and_save!(env::RocketEnv2D, policy::Function, max_steps::Int, filename::String, fps::Float64=10.0)
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
    data = []

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

        # Save the state and action to a DataFrame
        # In the form [x, y, x_dot, y_dot, theta, theta_dot, time, action]
        x, y, x_dot, y_dot, theta, theta_dot, t = s
        thrust, ϕ, torque = torque_controls(env, s, a)
        thrust_ratio = thrust / (2*env.g*env.m)
        push!(data, (frame=round(Int, i*(fps*env.dt)),time=t, x=x, y=0.0, z=y, phi=0.0, theta=theta, psi=0.0, thrust_ratio=thrust_ratio, thrust=thrust, vectoring_angle=ϕ, torque=torque))

        # Get the reward
        r = reward!(env)

        # Add the reward to the total
        total_reward += γ^i * r

        # Check if the environment is terminated
        if CommonRLInterface.terminated(env)
            break
        end
    end 

    #println("Total Reward: ", total_reward)
    #print("Final State in Trajectory: ", round.(env.state; digits=2), "\n \n")

    # Save the states and actions to a CSV file
    filename = "simulations/" * filename * "_2D.csv"
    CSV.write(filename, DataFrame(data), writeheader=true)

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
        r = reward!(env)

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
        r = reward!(env)

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
        a = policy(s)
        r = act!(env, a)
        push!(inputData, s)                     # State Data
        push!(outputData, a)     # Output thrust and torque
    end
    return inputData, outputData
end

################################################################

################# Custom Functions #############################

# Print the variables in the environment
function print_env(env::RocketEnv2D)
    print(" --- 2D Rocket Environment ---\n")
    println("Bounds: ", env.bounds, " m")
    println("dt: ", env.dt, " s")
    println("gravity: ", env.g, " m/s^2\n")
    println("State: ", env.state)
    println("[x, y, x_dot, y_dot, theta, theta_dot]\n")

    println("Action Space: ", env.action_space)
    println("[2*mg, 1*mg, 0.1*mg]\n")

    println("Rocket Parameters:")
    println("Mass: ", env.m, " kg")
    println("Moment of Inertia: ", env.I, " kg*m^2")
    println("Height: ", env.h, " m")
    println("Gains: ", env.gains)
end

# This render will spawn the rocket in a grid of positions to test the PD controller gains
# function CommonRLInterface.render(env::RocketEnv2D)

#     println("Rendering the environment")

#     # Get rocket Images
#     #rocket = load("imgs/rocket.png")

#     # Plot the rocket
#     s = scatter([env.target], [0.0], label="Target", color="red")
#     plot!(s, [env.bounds[1], env.bounds[2]], [0.0, 0.0], label="Ground", color="green", lw=2)
#     #scatter!([x],[y], label="Rocket", color="blue", ms=10, marker=:circle)
#     xlims!(s, env.bounds[1], env.bounds[2])
#     ylims!(s, -200.0, env.bounds[4]*1.2)

#     # Plot the target bounds
#     plot!(s, [env.target - 0.1 * (env.bounds[2] - env.bounds[1]), env.target + 0.1 * (env.bounds[2] - env.bounds[1])], [0.0, 0.0], label=nothing, color="red", lw=2)

#     # Simulate n trajectories and plot the results
#     n = 10
#     # Define the number of arrows to plot per trajectory
#     num_arrows = 7
#     arrow_scale = 1000.0

#     function make_arrow!(arrow_scale, state)
#         theta = state[5]
#         # Calculate the magnitude of the velocity vector
#         velocity_magnitude = sqrt(state[3]^2 + state[4]^2)
#         # Normalize the velocity vector for 50 m/s
#         velocity_magnitude_x = (velocity_magnitude * (env.bounds[2] - env.bounds[1])) / arrow_scale
#         velocity_magnitude_y = (velocity_magnitude * (env.bounds[4] - env.bounds[3])) / arrow_scale

#         u = velocity_magnitude_x * cos(theta + pi/2)
#         v = velocity_magnitude_y * sin(theta + pi/2)
#         quiver!(s, [state[1]], [state[2]], quiver=([u], [v]), color="black")
#     end

#     # Create a new plot for the states over time
#     p = plot(layout=(4,1), size=(800, 600))

#     # Define the range of points to Test
#     x_width = env.bounds[2] - env.bounds[1]
#     y_width = env.bounds[4] - env.bounds[3]
#     A = CommonRLInterface.actions(env)

#     for i in 1:n
#         for j in 1:n
#             # Simulate the trajectory
#             state, total_reward, actions = simulate_trajectory!(env, s->A[discrete_policy_distance_metric(s)], 10000, (i/n)*x_width + env.bounds[1], (j/n)*y_width + env.bounds[3])

#             # Break if state is empty
#             if isempty(state)
#                 println("State is empty")
#                 break
#             end

#             x_traj = [s[1] for s in state]
#             y_traj = [s[2] for s in state]

#             # Plot the trajectory
#             plot!(s, x_traj, y_traj, label=nothing, color=reward_to_color(total_reward), lw=2)

#             # Calculate the interval at which to plot the arrows
#             interval = Int(ceil(length(x_traj) / num_arrows))

#             # Plot the arrows at regular intervals along the trajectory
#             for i in 1:interval:length(x_traj)
#                 make_arrow!(arrow_scale, state[i])
#             end

#             # Make an arrow at the end of the trajectory
#             make_arrow!(arrow_scale, state[end])

#             # Plot the state over time
#             p = state_plot(p, state, actions, reward_to_color(total_reward))
#         end
#     end

#     return s, p
# end