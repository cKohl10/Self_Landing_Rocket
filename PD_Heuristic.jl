using ProgressMeter

function root_locus_plot(A::Function, k, range, title_name)
    # Create a plot of the root locus of the system
    # Inputs
        # A: The A matrix of the system
        # k: The gains of the system
        # title_name: The title of the plot
    # Outputs
        # p: The plot of the root locus of the system

    #zeros = eigvals(D - C*inv(A)*B)
    k_length = length(k)
    k_new = k
    append!(k_new, 0)

    min_real = Inf
    max_real = -Inf

    # Create a plot of the root locus
    p = plot(title=title_name, xlabel="Real", ylabel="Imaginary")

    # Initiate a progress bar
    pbar = Progress(length(range), 1)

    for i in 1:length(range)

        k_new[k_length+1] = range[i]
        A_new = A(k_new)
        # Get the poles and zeros of the system
        poles = eigvals(A_new)
        r_poles = real(poles)
        i_poles = imag(poles)

        # Plot the poles and zeros
        # Give the value as the label if it has no imaginary component and the real is negative
        for j in 1:length(poles)
            if i_poles[j] == 0 && r_poles[j] < 0
                scatter!([r_poles[j]], [i_poles[j]], label=nothing, color=:green)
                println("K", k_length+1, ": ", range[i], " Real: ", r_poles[j], " Imag: ", i_poles[j])
            else
                scatter!([r_poles[j]], [i_poles[j]], label=nothing, color=:red)
            end
        end
        #scatter!(r_poles, i_poles, label=nothing, color=:red)

        # Keep track of the lowest and highest real values
        min_real = min(min_real, minimum(r_poles))
        max_real = max(max_real, maximum(r_poles))

        # Update the progress bar
        next!(pbar)
    end

    # Scale the axis to fit the poles and zeros
    xlims!(min_real, max_real)
   
    #scatter!(real(zeros), imag(zeros), label="Zeros", color=:blue)

    return p
end

function calculate_gains(env)
    # Hyperparameters
    λ_1 = -1/2
    λ_2 = -1/0.2
    @show k1_rot = (λ_1 * λ_2)*env.I
    @show k2_rot = -(λ_1 + λ_2)*env.I
    k3_rot = -2.0 * 10^5
    k4_rot = 0.0

    function A_2(k)
        A_mat = [0 1; -k[1]/env.I -k[2]/env.I]
        return A_mat
    end

    k_2 = [k1_rot]
    range_2 = LinRange(1*10^6, 5*10^6, 100)
    #display(root_locus_plot(A_2, k_2, range_2, "Root Locus for K2"))

    #A(k) = [0 1 0 0; 0 0 -9.81 0; 0 0 0 1; -(k(3)*k(4))/env.I -k(3)/env.I -k(1)/env.I -k(2)/env.I]
    function A_3(k)
        A_mat = [0 -9.81 0; 0 0 1; -k[3]/env.I -k[1]/env.I -k[2]/env.I]
        return A_mat
    end

    k_3 = [k1_rot, k2_rot]
    range_3 = LinRange(0, 5*10^6, 100)
    #display(root_locus_plot(A_3, k_3, range_3, "Root Locus for K3"))

    function A_4(k)
        A_mat = [0 1 0 0; 0 0 -9.81 0; 0 0 0 1; -(k[3]*k[4])/env.I -k[3]/env.I -k[1]/env.I -k[2]/env.I]
        return A_mat
    end

    k_4 = [k1_rot, k2_rot, k3_rot]
    range_4 = LinRange(0, 5*10^0, 1000)
    #display(root_locus_plot(A_4, k_4, range_4, "Root Locus for K4"))


    ######## Thrust Gains ########
    λ_1th = -1/2
    λ_2th = -1/0.2
    @show k1_thrust = (λ_1th * λ_2th)*env.m
    @show k2_thrust = -(λ_1th + λ_2th)*env.m

end

function heuristic_policy(s)

    # Hyperparameters
    ζ_rot = 1.2
    ω_n_rot = 0.1 # Natural frequency Hz
    λ_1 = -1/2
    λ_2 = -1/0.2
    k1_rot = (λ_1 * λ_2)*env.I
    k2_rot = -(λ_1 + λ_2)*env.I
    k3_rot = -2 * 10^5
    k4_rot = 0.02

    A = [0 1 0 0; 0 0 -9.81 0; 0 0 0 1; -(k4_rot*k3_rot)/env.I -k3_rot/env.I -k1_rot/env.I -k2_rot/env.I]

    λ_1th = -1/30
    λ_2th = -1/5
    k1_thrust = (λ_1th * λ_2th)*env.m
    k2_thrust = -(λ_1th + λ_2th)*env.m

    # Unpack the state
    x, y, x_dot, y_dot, theta, theta_dot, t = s

    if t < 80 #seconds
        y_ref = env.bounds[4]
    else
        y_ref = 0
    end

    # If the rocket is not above the landing pad, move to the left or right
    torque = -k1_rot*theta - k2_rot*theta_dot - k3_rot*x_dot - k4_rot*k3_rot*x
    thrust = -k1_thrust*(y - y_ref) - k2_thrust*y_dot

    return [thrust, torque]

end