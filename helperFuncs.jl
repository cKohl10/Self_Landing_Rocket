using ColorSchemes
using Statistics
using ProgressMeter

function reward_to_color(reward)

    max_reward = 3000
    min_reward = -200

    # Normalize the reward to a value between 0 and 1
    # its range is -500 to 300, so we add 500 and divide by 800
    normalized_reward = (reward - min_reward) / (max_reward - min_reward)

    # Create a color map that interpolates between red and green
    color_map = ColorSchemes.RdYlGn
    #color_map = ColorSchemes.viridis

    # Get the color corresponding to the normalized reward
    color = color_map[normalized_reward]

    return color
end

function state_plot(p, states, actions, color)
    # This will create a plot of the ydot, theta, and theta_dot states as a layout plot with 3 subplots
    # The x-axis will be the time step and the y-axis will be the value of the state
    # states = [x, y, x_dot, y_dot, theta, theta_dot]

    # Create a plot for the y state
    plot!(p[1], [s[2] for s in states], color=color, label=nothing, xlabel="Time Step", ylabel="Y Position")

    # Create a plot for the ydot state
    plot!(p[2], [s[4] for s in states], color=color, label=nothing, xlabel="Time Step", ylabel="Y Velocity")

    # Create a plot for the theta state
    plot!(p[3], [s[5] for s in states], color=color, label=nothing, xlabel="Time Step", ylabel="Theta")


    a_size = length(actions[1])

    if a_size == 3
        # Create a plot for the thrust
        plot!(p[4], [a[1] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="Thrust")

        # Create a plot for the vectoring angle ϕ
        plot!(p[5], [a[2] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="ϕ")

        # Create a plot for the torque
        plot!(p[6], [a[3] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="Torque")

    elseif a_size == 2
        # Create a plot for the thrust
        plot!(p[4], [a[1] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="Thrust")

        # # Create a plot for the torque
        plot!(p[5], [a[2] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="Torque")

    elseif a_size == 1
        # Create a plot for the thrust
        plot!(p[4], [a[1] for a in actions], color=color, label=nothing, xlabel="Time Step", ylabel="Thrust")
    end

    return p
    
end

function rand_float(min::Float64, max::Float64)
    return min + rand() * (max - min)
end

# Useful for plotting the learning curves
function data_plot(data, label, title=nothing, scalar=1.0)
    epochs = (1:length(data))*scalar
    n = ceil(Int, length(epochs)/100)
    if n <= 0
        n = 1
    end
    x = n:n:length(epochs)
    y = [mean(data[i-n+1:i]) for i in x]
    p = plot(x, y)
    xlabel!("Training Epochs")
    ylabel!(label)
    if title != nothing
        title!(title)
    end
    display(p)
end


# Evaluate function, calculates the average reward using the simulate function for a given number of episodes
function eval(env::RocketEnv2D, policy::Function, num_eps)
    totReward = 0.0
    for _ in 1:num_eps
        totReward += simulate!(env, policy::Function, 100000)
    end
    # Return the average reward per episode
    return totReward / num_eps
end

# This function can be used to back out the controls at each state
function torque_controls(env::RocketEnv2D, state, action)
    #### Update the State ####
    # Unpack the state and action
    x, y, x_dot, y_dot, theta, theta_dot, t = state
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
            torque = thrust * sin(ϕ) * (h/2)
        else 
            # Desired angle of the thrust vector
            ϕ = asin((2*torque_r)/(h*thrust)) 

            # Limit the angle of the thrust vector
            torque = thrust * sin(ϕ) * (h/2)
        end
    else
        # If the thrust is zero, the torque and vectoring angle are zero
        torque = 0.0
        ϕ = 0.0
    end

    return [thrust, ϕ, torque]

end

function supervised_learning!(model, X, Y, epochs, batch_size, loss_fn, opt=ADAM(0.001), using_GPU=false, eval_steps=5)
    # X = input data [nxm], Y = target data [nx1]
    # Train the model using supervised learning
    if using_GPU
        #println("Loading data onto GPU...")
        # Shuffle the data
        train_loader = Flux.DataLoader((X, Y), batchsize=batch_size, shuffle=true, partial=true) |> gpu
    else
        println("Loading data onto CPU...")
        # Shuffle the data
        train_loader = Flux.DataLoader((X, Y), batchsize=batch_size, shuffle=true, partial=true)    
    end

    # Create a progress bar
    p = Progress(Int(epochs/eval_steps), 1)
    losses = []
    loss_val = 0.0

    for epoch in 1:epochs

        # Take a subset of the data based on the batch size and train the model
        # for (x, y) in train_loader
        #     data = [(x[i], y[i]) for i in 1:length(x)]
        #     Flux.train!(loss_fn, model, data, opt)
        #     #println("Epoch: ", epoch, " Loss: ", loss_fn(model, x, y))
        # end

        data, state = iterate(train_loader)
        data = vec(collect(zip(data[1], data[2])))
        Flux.train!(loss_fn, model, data, opt)

        # Print the loss every 10 epochs
        if epoch % eval_steps == 0
            # Get a validation set of data to compare to the training data
            x, y = first(train_loader)
            loss_val = 0.0
            for i in 1:length(x)
                loss_val += loss_fn(model, x[i], y[i])
            end
            loss_val = round(loss_val/length(x), digits=1)
            
            # Update the progress bar
            next!(p)
            println("\nEpoch: ", epoch, " Loss: ", loss_val)

            # Append the loss to the losses array
            push!(losses, loss_val)

        end

        if epoch % (eval_steps*10) == 0
            # Save the model every 100 epochs
            println("Saving model...")
            save_model(model, "models/supervised_model_temp.bson")

            # Plot the learning curve
            data_plot(losses, "Loss", "Training Curve for Supervised Behavior Cloning", scalar=eval_steps)
        end
    end

    return losses
end