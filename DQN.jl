using Flux
using LinearAlgebra

# DQN Function
function DQN_Solve(env)

    print("Training DQN Model...\n")
    reset!(env)

    # Deep Q Network to approximate the Q values
    Q = Chain(Dense(length(actions(env)), 128, relu),
            Dense(128, length(actions(env))))
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
            return rand(1:length(actions(env)))
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
            display(render(env, s->actions(env)[argmax(Q(s))], title_name))

            # Plot the learning curve
            display(data_plot(rewards_history, "Average Reward"))
        end
    end

    return Q
end

function heuristic_policy(s)

    # Hyperparameters
    ζ_rot = 1.2
    ω_n_rot = 0.1 # Natural frequency Hz
    λ_1 = -1/2
    λ_2 = -1/0.2
    k1_rot = (λ_1 * λ_2)*env.I
    k2_rot = -(λ_1 + λ_2)*env.I
    k3_rot = 0.0
    k4_rot = 0.0

    A = [0 1 0 0; 0 0 -9.81 0; 0 0 0 1; -(k3_rot*k4_rot)/env.I -k3_rot/env.I -k1_rot/env.I -k2_rot/env.I]
    @show eigvals(A)

    k1_thrust = 0.1
    k2_thrust = 0.1

    # Unpack the state
    x, y, x_dot, y_dot, theta, theta_dot = s

    # If the rocket is not above the landing pad, move to the left or right
    torque = -k1_rot*theta - k2_rot*theta_dot - k3_rot*x_dot - k4_rot*k3_rot*x
    thrust = env.m*9.81

    return [thrust, torque]

end