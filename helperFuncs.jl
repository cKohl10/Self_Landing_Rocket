using ColorSchemes
using Statistics

function reward_to_color(reward)
    # Normalize the reward to a value between 0 and 1
    # its range is -500 to 300, so we add 500 and divide by 800
    normalized_reward = (reward + 500) / 800

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