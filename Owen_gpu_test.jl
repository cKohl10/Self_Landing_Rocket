using Flux
using LinearAlgebra
using BSON
using Printf
using DataStructures
using Random
using StatsBase
# DQN Function
function DQN_Solve_Metric(env)
    print("Training DQN Model...\n")
    CommonRLInterface.reset!(env)
    # Priority Queue for experience replay
    buffer = PriorityQueue{Tuple{Any, Any, Any, Any, Bool}, Float64}()  # Key is experience, value is priority
    # Deep Q Network to approximate the Q values
    Q = Chain(Dense(length(observe(env)), 64, relu),
              Dense(64, 64, relu),
              Dense(64, length(actions(env))))

    load = false
    if load
        name = "models/Q_discrete_metric_best_reward.bson"
        Q = BSON.load(name)[:Q]
        print("Model loaded from: ", name, "\n")
    end

    Q_target = deepcopy(Q)
    Q_best = deepcopy(Q)

    # HYPERPARAMETERS
    bufferSize = 100000
    batch = 2000
    ϵ_max = 0.6
    ϵ_min = 0.05
    exploration_epochs = 5000
    n = 1000  # Number of steps in an episode
    epochs = 10000
    num_eps = 100  # For evaluate function
    max_steps = 2000  # Maximum number of steps in an eval episode
    set_Q_targ = 5  # Set the target Q network every set_Q_targ epochs

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
    
    # Function to add experience with priority
    function add_experience!(buffer, experience, priority)
        enqueue!(buffer, experience, priority)
    end

    # Modified experience function to calculate TD Error
    function experience(buffer, n, ϵ)
        for i = 1:n
            done = terminated(env)
            if done
                break
            end
            s = observe(env)
            a_ind = policy(s, ϵ)
            r = act!(env, actions(env)[a_ind])
            sp = observe(env)
            experience_tuple = (s, a_ind, r, sp, done)

            # Calculate initial priority (TD Error approximation) or max priority in the buffer
            initial_priority = buffer |> isempty ? 1.0 : maximum(values(buffer))
            add_experience!(buffer, experience_tuple, initial_priority)
        end
        return buffer
    end

    # Function to sample based on priorities
    function sample_experiences(buffer, batch_size)
        # Extract all items and their priorities from the buffer
        items = collect(keys(buffer))
        priorities = Float64[buffer[item] for item in items]
        total_priority = sum(priorities)
        probabilities = priorities / total_priority
    
        # Ensure we do not sample more than available
        actual_batch_size = min(batch_size, length(items))
        if actual_batch_size < batch_size
            @warn "Reduced batch size due to insufficient samples in buffer"
        end
    
        # Sample based on these probabilities
        indices = sample(1:length(items), Weights(probabilities), actual_batch_size, replace=false)
        sampled_experiences = [items[i] for i in indices]
        return sampled_experiences
    end

    # Evaluate function, calculates the average reward using the simulate function for a given number of episodes
    function eval(Q_eval, num_eps)
        Qpolicy = s->actions(env)[argmax(Q_eval(s))]

        totReward = 0
        for _ in 1:num_eps
            CommonRLInterface.reset!(env)
            r = simulate!(env, Qpolicy, max_steps)
            totReward += r
        end
        return totReward / num_eps
    end

    rewards_history = []
    best_reward = -1000.0
    buffer = experience(buffer, n, 1)

    for epoch in 1:epochs
        if epoch == exploration_epochs
            print("Exploration phase completed...\n")
            Q = deepcopy(Q_best)
            Q_target = deepcopy(Q_best)
        end
        CommonRLInterface.reset!(env)

        opt = Flux.setup(ADAM(0.0005), Q)

        function ϵ(ϵ_min, ϵ_max, exploration_epochs, epoch)
            if epoch <= exploration_epochs
                return ϵ_min + (ϵ_max - ϵ_min) * ((epochs - epoch) / exploration_epochs)
            else
                return ϵ_min
            end
        end

        buffer = experience(buffer, n, ϵ(ϵ_min, ϵ_max, exploration_epochs, epoch))

        if epoch % set_Q_targ == 0
            Q_target = deepcopy(Q)
        end

        function loss(Q, s, a_ind, r, sp, done)
            g = env.γ
            if done
                return (r - Q(s)[a_ind])^2
            end
            return (r + g * maximum(Q_target(sp)) - Q(s)[a_ind])^2
        end

        data = sample_experiences(buffer, batch)

        Flux.Optimise.train!(loss, Q, data, opt)

        avgReward = eval(Q, num_eps)
        push!(rewards_history, avgReward)

        if avgReward > best_reward
            Q_best = deepcopy(Q)
        end

        # Shift the buffer if the model is not learning well
        if avgReward <= best_reward * 0.9 && length(buffer) > bufferSize * 0.75
            buffer = experience(buffer, n, 1)  # Replace buffer with new experiences
        end

        if length(buffer) > bufferSize
            buffer = buffer[end - bufferSize:end]
        end

        print("Epoch: ", epoch, "\t Buffer Size: ", length(buffer), "\t Average Reward: ", avgReward, "\n")

        if epoch % 10 == 0
            title_name = "Epoch: " * string(epoch)
            s, p = render(env, s->actions(env)[argmax(Q(s))], title_name, 20)
            display(p)
            display(s)
            display(data_plot(rewards_history, "Average Reward"))
        end
    end

    print("Evaluating the best model...\n")
    best_reward = eval(Q_best, 10000)
    print("Saved model with best reward of: ", best_reward, "\n")
    save_model(Q_best, string("models/Q_discrete_metric_" * @sprintf("%0.1f", best_reward) * "_best_reward"))

    return Q_best
end

function save_model(Q, filename)
    BSON.@save filename Q
end
