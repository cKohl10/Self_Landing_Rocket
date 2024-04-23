# Importing the environment
include("environment.jl")


# DQN Function
function DQN_Solve(env)
    # Deep Q Network to approximate the Q values
    Q = Chain(Dense(length(actions(env)), 128, relu),
            Dense(128, length(actions(env))))
    Q_target = deepcopy(Q)

    # HYPERPARAMETERS
    bufferSize = 50000
    epsilon = 0.1
    n = 10000
    epochs = 1000

    # Epsilon Greedy Policy
    function policy(s, epsilon=0.1)
        if rand() < epsilon
            return rand(actions(env))
        else
            return actions(env)[argmax(Q(s))]
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
            a = policy(s)
            r = act!(env, a)
            sp = observe(env)
            experience_tuple = (s, a, r, sp, done)
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


end