using Flux
using LinearAlgebra

# DQN Function
function DQN_Solve(env)

    print("Training DQN Model...\n")
    reset!(env)

    # Deep Q Network to approximate the Q values
    Q = Chain(Dense(length(observe(env)), 128, relu),
            Dense(128, length(actions(env))))
    Q_target = deepcopy(Q)

    # HYPERPARAMETERS
    bufferSize = 50000
    epsilon = 0.1
    n = 10000
    batch_size = 2000
    epochs = 100
    num_eps = 100   # For evaluate function

    # Define Huristic policy by using controler and taking the closest value
    function discrete_policy(s)
        discrete_actions = []
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
        discrete_actions = [thrust,torque]
        for i in 1:length(actions(env))
            if discrete_actions == actions(env)[i]
                actions_index = i
            end
        end
        return actions_index
    end
    # Epsilon Greedy Policy
    function policy(s, epsilon=0.1)
        if rand() < epsilon
            return discrete_policy(s)
        else
            return argmax(Q(s))
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
            a_ind = policy(s)
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            experience_tuple = (s, a_ind, r, sp, done)
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

        # Gain experience
        buffer = experience(buffer, n)

        # Copy Q network and define the loss function
        Q_target = deepcopy(Q)
        function loss(Q, s, a_ind, r, sp, done)
            # Discount factor
            g = 0.99
            # Reached terminal state
            if done
                return (r - Q(s)[a_ind])^2
            end
            # DQN Loss Function
            return (r + g*maximum(Q_target(sp)) - Q(s)[a_ind])^2
        end

        # Get random data from the buffer
        data = rand(buffer, batch_size)

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
            display(render(env, s->actions(env)[argmax(Q(s))], title_name))

            # Plot the learning curve
            display(data_plot(rewards_history, "Average Reward"))
        end
    end

    return Q
end