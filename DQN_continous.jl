using Flux

# DQN Function
function DQN_Solve_Continuous(env)

    print("Training DQN Model...\n")
    reset!(env)

    # Deep Q Network to approximate the Q values
    Q = Chain(Dense(length(observe(env)), 128, relu),
            Dense(128, 2))
    Q_target = deepcopy(Q)

    # HYPERPARAMETERS
    bufferSize = 5000
    epsilon = 0.1
    n = 10000
    epochs = 1000
    num_eps = 100   # For evaluate function

    # Epsilon Greedy Policy
    function policy(s, epsilon=0.1)
        if rand() < epsilon
            return [rand_float(-env.thrust, env.thrust), rand_float(-env.torque, env.torque)]
        else
            return Q(s)
        end
    end

    # Gain experience function, appends the buffer
    function experience(buffer, n)
        # Loop through n steps in the environment and add to the buffer
        for i = 1:n

            done = terminated(env)
            if done  # Break if a terminal state is reached
                break
            end
            s = observe(env)
            thrust, torque = policy(s)
            r = act!(env, [thrust, torque])
            sp = observe(env)
            experience_tuple = (s, r, sp, done)
            push!(buffer, experience_tuple)                 # Add to the experience
        end
        return buffer
    end

    # Evaluate function, calculates the average reward using the simulate function for a given number of episodes
    function eval(Q_eval, num_eps)
        # Define policy from the Q values, take the maximum Q value for a given state
        Qpolicy = s->actions(env)[argmax(Q_eval(s))]

        totReward = 0
        for _ in 1:num_eps
            totReward += simulate!(env, Qpolicy, n)
        end
        # Return the average reward per episode
        return totReward / num_eps
    end


    # Instantiate the buffer, 1000 steps for each episode
    buffer = []
    rewards_history = []
    buffer = experience(buffer, n)      # Initial buffer episode

    # Execute DQN Learning
    for epoch in 1:epochs
        # Rest the environment
        reset!(env)

        # Set Optimizer
        opt = Flux.setup(ADAM(0.0005), Q)

        # Copy Q network and define the loss function
        Q_target = deepcopy(Q)
        function loss(Q, s, r, sp, done)
            # Discount factor
            g = 0.99
            # Reached terminal state
            if done
                return (r - Q(s))^2
            end
            # DQN Loss Function
            return (r + g*Q_target(sp) - Q(s))^2
        end

        # Get random data from the buffer
        data = rand(buffer, 2000)

        # Train based on random data in the buffer
        Flux.Optimise.train!(loss, Q, data, opt)

        # Evaluate the epoch
        avgReward = eval(Q, num_eps)

        # Append the average reward to the rewards history
        push!(rewards_history, avgReward)

        # Shift the buffer if exceeding buffer size
        if length(buffer) > bufferSize
            buffer = buffer[end-bufferSize:end]
        end

        # Output data
        print("Epoch: ", epoch, "\t Buffer Size: ", length(buffer), "\t Average Reward: ", avgReward, "\n")

        # Display simulated trajectories
        if epoch % 10 == 0
            # Simulate the environment with a few trajectories
            title_name = "Epoch: " * string(epoch)
            display(render(env, s->Q(s), title_name))

            # Plot the learning curve
            display(data_plot(rewards_history, "Average Reward"))
        end
    end

    return Q
end






## Function approximator based on the PD heuristic
# Create a neural network to approximate the torque and thrust given a state
function DPD_Continuous(env)

    print("Training DPD Model...\n")
    reset!(env)

    # Deep action network to approximate thrust and torque
    Q = Chain(Dense(length(observe(env)), 128, relu),
            Dense(128, 2))

    
    # heuristic_policy(s)

    # Simulate with the controller to get data
    

end