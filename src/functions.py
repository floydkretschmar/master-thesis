####################### BENEFIT FUNCTIONS #######################
def demographic_parity(**benefit_parameters):
    policy = benefit_parameters["policy"]
    x_s = benefit_parameters["x_s"]
    s = benefit_parameters["s"]

    decisions = policy(x_s, s).reshape(-1, 1)
    return decisions

def equal_opportunity(**benefit_parameters):
    y_s = benefit_parameters["y_s"]
    return y_s * demographic_parity(**benefit_parameters)

####################### UTILITY FUNCTIONS #######################

def cost_utility(cost_factor, **utility_parameters):
    policy = utility_parameters["policy"]
    x = utility_parameters["x"]
    s = utility_parameters["s"]
    y = utility_parameters["y"]

    decisions = policy(x, s).reshape(-1, 1)
    return decisions * (y - cost_factor)