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



## Function approximator based on the expert PD controller
# Create a neural network to approximate the torque and thrust given a state
# Uses DAgger to improve the approximation
function CloneExpert(env, heuristic)

    print("Training DAgger Model...\n")
    reset!(env)

    # Deep action network to approximate thrust and torque
    net = Chain(Dense(length(observe(env)), 128, relu),
            Dense(128, 1))
    bestNet = deepcopy(net)

    # HYPERPARAMETERS
    maxSteps = 10000        # Max steps in the environment for episode calls
    numEps = 1000           # Episodes for initial data set
    daggerEPs = 10          # Addition simulations to augment data set
    best_reward = -Inf

    # Define loss function and optimizer
    loss(model, x, y) = Flux.mse(model(x), y);
    #opt = Flux.ADAM(0.01)
    opt = Flux.setup(Adam(0.01), net)
    
    # Gather large initial data set for behavior cloning
    inputData = []
    outputData = []
    for _ in 1:numEps
        inEp, outEp = episode!(env, heuristic, maxSteps)
        append!(inputData, inEp)
        append!(outputData, outEp)
    end

    # Train with expert data set
    print("Cloning Behavior...\n")
    data = [(inputData[i], outputData[i]) for i in 1:length(inputData)]
    Flux.train!(loss, net, data, opt)
    print("Cloned Behavior, executing DAgger...\n")

    # DAgger to improve the neural net
    for i in 1:daggerEPs
        # Reset the environment
        reset!(env)
        for i = 1:maxSteps
            # Expert data
            inputData = []
            outputData = []

            # Check if terminated
            done = terminated(env)
            if done  # Break if a terminal state is reached
                break
            end

            # Create policy from the network
            function netPolicy(s)
                return convert(Float64, net(s))
            end

            # Act in the environment and collect data
            s = observe(env)
            thrustNet = netPolicy(s)
            thrust = heuristic(s)           # Expert data
            r = act!(env, thrustNet)        # Act with neural network
            push!(inputData, s)             # State Data
            push!(outputData, thrust)       # Output thrust and torque from the heuristic to train
        end

        # Train with new expert data
        data = [(inputData[i], outputData[i]) for i in 1:length(inputData)]
        Flux.train!(loss, net, data, opt)

        # Create policy
        function netPolicy(s)
            thrust = net(s)
            return convert(Float64, thrust[1])
        end

        # Calculate the reward
        DAggerReward = mean([simulate!(env, netPolicy, maxSteps) for _ in 1:100])

        if DAggerReward > best_reward
            best_reward = DAggerReward
            bestNet = deepcopy(net)
        end

        print("DAgger Epoch: ", i, " Average Reward: ", DAggerReward, "\n")
    end
    return bestNet
end



