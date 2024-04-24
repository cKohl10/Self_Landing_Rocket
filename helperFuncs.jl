using ColorSchemes
using Statistics

function reward_to_color(reward)
    # Normalize the reward to a value between 0 and 1
    # its range is -500 to 300, so we add 500 and divide by 800
    normalized_reward = (reward + 300) / 600

    # Create a color map that interpolates between red and green
    color_map = ColorSchemes.RdYlGn

    # Get the color corresponding to the normalized reward
    color = color_map[normalized_reward]

    return color
end

function state_plot(p, states, actions, color)
    # This will create a plot of the ydot, theta, and theta_dot states as a layout plot with 3 subplots
    # The x-axis will be the time step and the y-axis will be the value of the state
    # states = [x, y, x_dot, y_dot, theta, theta_dot]

    # Create a plot for the ydot state
    plot!(p[1], [s[4] for s in states], color=color, label=nothing, xlabel="Time Step", ylabel="Y Velocity")

    # Create a plot for the theta state
    plot!(p[2], [s[5] for s in states], color=color, label=nothing, xlabel="Time Step", ylabel="Theta")

    # Create a plot for the theta_dot state
    plot!(p[3], [s[6] for s in states], color=color, label=nothing, xlabel="Time Step", ylabel="Angular Velocity")

    # Create a plot for the thrust
    plot!(p[4], [a[1] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="Thrust")

    # Create a plot for the torque
    plot!(p[5], [a[2] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="Torque")

    return p
    
end

function rand_float(min::Float64, max::Float64)
    return min + rand() * (max - min)
end

# Useful for plotting the learning curves
function data_plot(data, label)
    epochs = 1:length(data)
    n = ceil(Int, length(epochs)/500)
    if n <= 0
        n = 1
    end
    x = n:n:length(epochs)
    y = [mean(data[i-n+1:i]) for i in x]
    p = plot(x, y)
    xlabel!("Training Epochs")
    ylabel!(label)
    display(p)
end


# Evaluate function, calculates the average reward using the simulate function for a given number of episodes
function eval(policy, num_eps)
    totReward = 0
    for _ in 1:num_eps
        totReward += simulate!(env, policy, n)
    end
    # Return the average reward per episode
    return totReward / num_eps
end


    # Define Huristic policy by using controler and taking the closest value
    function discrete_policy(s)
        discrete_action = []
        actions_index = 0
        a = heuristic_policy(s)
        thrust_cont = a[1]
        thrust_threshold = 1.2
        torque_threshold = 1.2
        torque_cont = a[2]
        if env.thrust <= thrust_cont * thrust_threshold
            thrust = env.thrust
        else
            thrust = 0.0
        end
        if abs(env.torque) <= abs(torque_cont) * torque_threshold
            if torque_cont > 0
                torque = env.torque
            else
                torque = -env.torque
            end
        else
            torque = 0.0
        end
        discrete_action = [thrust,torque]
        for i in 1:length(actions(env))
            if discrete_action == actions(env)[i]
                actions_index = i
            end
        end
        return actions_index
    end