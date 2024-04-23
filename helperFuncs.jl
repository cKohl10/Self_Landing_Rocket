using ColorSchemes

function reward_to_color(reward)
    # Normalize the reward to a value between 0 and 1
    # its range is -500 to 300, so we add 500 and divide by 800
    normalized_reward = (reward + 500) / 800

    # Create a color map that interpolates between red and green
    color_map = ColorSchemes.RdYlGn

    # Get the color corresponding to the normalized reward
    color = color_map[normalized_reward]

    return color
end