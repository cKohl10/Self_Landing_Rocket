# Importing the environment
include("environment.jl")


# DQN Function
function DQN_Solve(env)
    # Deep Q Network to approximate the Q values
    Q = Chain(Dense(2, 128, relu),
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
            
        else
            #return argmax(aInd->Q(s)[aInd], 1:length(actions(env)))
        end
    end

    # Gain experience function, appends the buffer
    function experience(buffer, n, epsilon)
        # See if terminal
        done = terminated(env)
        i = 1
        # Recently added buffer
        recentBuffer = []
        # Loop through n steps in the environment and add to the buffer
        while i <= n && !done

        end
        return buffer
    end

    # Evaluate function
    function eval(Q_eval, num_eps)
        totReward = 0
        for _ in 1:num_eps
            totReward += simulate!(env, policy, n)
        end
        # Return the average reward per episode
        return totReward / num_eps
    end

end