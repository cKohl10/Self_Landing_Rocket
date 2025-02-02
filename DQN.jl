using Flux
using LinearAlgebra
using BSON
using Printf


# DQN Function
function DQN_Solve_Metric(env)

    print("Training DQN Model...\n")
    reset!(env)

    # Deep Q Network to approximate the Q values
    Q = Chain(
        Dense(length(observe(env)), 128, relu),
        Dropout(0.1),
        Dense(128, 128, relu),
        Dropout(0.1),
        Dense(128, length(actions(env)))
    )

    load = false
    if load
        name = "models/Q_discrete_metric_best_reward.bson"
        Q = BSON.load(name)[:Q]
        print("Model loaded from: ", name, "\n")
    end

    Q_target = deepcopy(Q)
    Q_best = deepcopy(Q)
    #Q_best_local = deepcopy(Q)

    # HYPERPARAMETERS
    bufferSize = 150000
    batch = 2000
    ϵ_max = 0.6
    ϵ_min = 0.05
    exploration_epochs = 5000
    n = 1000 # Number of steps in an episode
    epochs = 10000
    num_eps = 100   # For evaluate function
    max_steps = 2000 # Maximum number of steps in an eval episode
    set_Q_targ = 1 # Set the target Q network every set_Q_targ epochs
    decay_rate = 0.005

    function continuous_policy(s)
        thrust_cont, torque_cont = heuristic_policy(s)
        return [thrust_cont, torque_cont]
    end

    
    # Epsilon Greedy Policy
    function policy(s, epsilon=0.1, δ=0.7)
        if rand() < epsilon

            if rand() < δ
                return discrete_policy_distance_metric(s)
            else
                return rand(1:length(actions(env)))
            end
        else
            return argmax(Q(s))
        end
    end

    # Gain experience function, appends the buffer
    function experience(buffer, n, ϵ)
        #step = 0
        # Loop through n steps in the environment and add to the buffer
        for i = 1:n

            done = terminated(env)
            if done  # Break if a terminal state is reached
                break
            end
            s = observe(env)
            a_ind = policy(s, ϵ)
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            experience_tuple = (s, a_ind, r, sp, done)
            push!(buffer, experience_tuple)                 # Add to the experience
            
            # debug
            #step = i
        end
        #println("Steps taken in episode: ", step)
        return buffer
    end

    # Evaluate function, calculates the average reward using the simulate function for a given number of episodes
    function eval(Q_eval, num_eps)
        # Define policy from the Q values, take the maximum Q value for a given state
        Qpolicy = s->actions(env)[argmax(Q_eval(s))]

        totReward = 0
        for _ in 1:num_eps
            reset!(env)
            r = simulate!(env, Qpolicy, max_steps)
            totReward += r
        end
        # Return the average reward per episode
        return totReward / num_eps
    end


    # Instantiate the buffer, 1000 steps for each episode
    buffer = []
    rewards_history = []
    best_reward = -1000.0
    buffer = experience(buffer, n, 1)      # Initial buffer episode

    # Execute DQN Learning
    for epoch in 1:epochs
        if epoch == exploration_epochs
            print("Exploration phase completed...\n")
            Q = deepcopy(Q_best)
            Q_target = deepcopy(Q_best)
        end
        # Rest the environment
        reset!(env)

        # Set Optimizer
        lr = 0.0005 * exp(-decay_rate * epoch)  # Decrease learning rate gradually
        opt = Flux.setup(ADAM(lr), Q)

        # Gain experience
        # function ϵ(ϵ_min, ϵ_max, exploration_epochs, epoch) 
        #     if epoch <= exploration_epochs
        #         return ϵ_min + (ϵ_max - ϵ_min) * ((epochs - epoch)/exploration_epochs)
        #     else
        #         return ϵ_min
        #     end
        # end
        # buffer = experience(buffer, n, ϵ(ϵ_min, ϵ_max, exploration_epochs, epoch))

        function epsilon(ϵ_min, ϵ_max, decay_rate, epoch)
            return ϵ_min + (ϵ_max - ϵ_min) * exp(-decay_rate * epoch)
        end
        buffer = experience(buffer, n, epsilon(ϵ_min, ϵ_max, decay_rate, epoch))
        

        # Copy Q network and define the loss function
        if epoch % set_Q_targ == 0
            Q_target = deepcopy(Q)
        end
        # function loss(Q, s, a_ind, r, sp, done)
        #     # Discount factor
        #     g = env.γ
        #     # Reached terminal state
        #     if done
        #         return (r - Q(s)[a_ind])^2
        #     end
        #     # DQN Loss Function
        #     return (r + g*maximum(Q_target(sp)) - Q(s)[a_ind])^2
        # end
        function loss(Q, s, a_ind, r, sp, done)
            g = env.γ
            target_Q_value = done ? r : r + g * maximum(Q_target(sp))
            return (target_Q_value - Q(s)[a_ind])^2
        end
        
        # Get random data from the buffer
        data = rand(buffer, batch)

        # Train based on random data in the buffer
        Flux.Optimise.train!(loss, Q, data, opt)

        # Evaluate the epoch
        avgReward = eval(Q, num_eps)

        # Append the average reward to the rewards history
        push!(rewards_history, avgReward)

        # Save the reward if it is the best reward
        if avgReward > best_reward
            best_reward = avgReward
            Q_best = deepcopy(Q)
        end

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
            s,p = render(env, s->actions(env)[argmax(Q(s))], title_name, 20)
            display(p) # State space plot
            display(s) # Display the inertial path plot

            # Plot the learning curve
            display(data_plot(rewards_history, "Average Reward"))

            # Update the local best model
            #Q_best_local = deepcopy(Q)
        end
    end

    # Save the model as previous trained model
    print("Evaluating the best model...\n")
    best_reward = eval(Q_best, 10000)
    print("Saved model with best reward of: ", best_reward, "\n")
    save_model(Q_best, string("models/Q_discrete_metric_" * @sprintf("%0.1f",best_reward) * "_best_reward"))

    return Q_best
end

function save_model(Q, filename)
    # Save the model to a file
    BSON.@save filename Q
end