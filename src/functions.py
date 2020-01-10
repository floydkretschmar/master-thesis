####################### BENEFIT FUNCTIONS #######################
def demographic_parity(**benefit_parameters):
    decisions = benefit_parameters["decisions"].reshape(-1, 1)
    return decisions

def equal_opportunity(**benefit_parameters):
    y_s = benefit_parameters["y"]
    decisions = benefit_parameters["decisions"].reshape(-1, 1)
    return y_s * decisions

####################### UTILITY FUNCTIONS #######################

def cost_utility(cost_factor, **utility_parameters):
    decisions = utility_parameters["decisions"].reshape(-1, 1)
    y = utility_parameters["y"]
    
    return decisions * (y - cost_factor)