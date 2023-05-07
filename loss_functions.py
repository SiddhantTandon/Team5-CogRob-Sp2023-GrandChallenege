import numpy as np
from multi_agent_sim_data import DataFrame


def action_equality(a1, a2):
    for i in range(len(a1)):
        if a1[i] != a2[i]:
            return False


def reward_equality(r1, r2):
    for i in range(len(r1)):
        if r1[i] != r2[i]:
            return False

def state_delta_mag(s1, s2, mapping):
    return np.linalg.norm(mapping@s2-mapping@s1)

def temporal_loss(batch, mapping):
    time_steps = len(batch)
    total_loss = 0
    for i in range(0, time_steps-1):
        total_loss += state_delta_mag(batch[i+1].image,batch[i].image, mapping)**2
    return total_loss/(time_steps-1)

def proportional_loss(batch,mapping):
    time_steps = len(batch)
    total_loss = 0
    pairings = 0
    for i in range(0, time_steps - 1):
        for j in range(i + 1, time_steps - 1):
            if batch[i].action == batch[j].action:
                pairings += 1
                delta1 = state_delta_mag(batch[i+1].image, batch[i].image, mapping)
                delta2 = state_delta_mag(batch[j+1].image, batch[j].image, mapping)
                total_loss += (delta2- delta1)**2
    if pairings == 0:
        return 0
    return total_loss/pairings

def causal_loss(batch,mapping):
    time_steps = len(batch)
    total_loss = 0
    pairings = 0
    for i in range(0, time_steps - 1):

        for j in range(i + 1, time_steps - 1):
            if batch[i].action == batch[j].action and batch[i].reward == batch[j].reward:
                pairings += 1
                delta = state_delta_mag(batch[i].image, batch[j].image, mapping)
                total_loss += np.exp(-delta)

    if pairings == 0:
        return 0
    return total_loss/pairings

def repeatability_loss(batch,mapping):
    time_steps = len(batch)
    total_loss = 0
    pairings = 0
    for i in range(0, time_steps - 1):
        for j in range(i + 1, time_steps - 1):
            if batch[i].action == batch[j].action:
                pairings += 1
                causal_part = np.exp(-state_delta_mag(batch[i].image, batch[j].image, mapping))
                delta1 = mapping@batch[i+1].image - mapping@batch[i].image
                delta2 = mapping@batch[j+1].image - mapping@batch[j].image
                total_loss += causal_part*(np.linalg.norm(delta2 - delta1)**2)
    if pairings == 0:
        return 0
    return total_loss/pairings

def multi_agent_loss(batch,mapping):
    time_steps = len(batch)
    total_loss = 0
    counter = 0
    for i in range(0, time_steps - 1):
        if batch[i].action[0] != batch[i].action[1]:
            counter += 1
            delta1 = (mapping @ batch[i + 1].image)[0:2] - (mapping @ batch[i].image)[0:2]
            delta2 = (mapping @ batch[i + 1].image)[2:4] - (mapping @ batch[i].image)[2:4]
            total_loss += np.exp(-np.linalg.norm(delta2 - delta1))

    if counter == 0:
        return 0
    return total_loss / counter



def temporal_cohesion_sol(batch, mapping):
    """
    computes the gradient of the temporal cohesion loss on the given batch of state representation
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: the gradient of the temporal cohesion loss
    """

    time_steps = len(batch)
    total_loss_grad = np.zeros(mapping.shape)
    for i in range(0,time_steps-1):
        loss_grad = temporal_loss_gradient(batch[i], batch[i+1], mapping)
        total_loss_grad += loss_grad.reshape(mapping.shape)

    return total_loss_grad/(time_steps-1)

def temporal_loss_gradient(init_state, next_state, mapping):
    dims = mapping.shape
    output = np.zeros(dims)
    for i in range(0, dims[0]):
        outside_comp = mapping[i,:]@init_state.image - mapping[i,:]@next_state.image
        for j in range(0, dims[1]):
            output[i,j] = 2*outside_comp*(init_state.image[j] - next_state.image[j])
    return output





##TODO: write up the data set data structure as well as

def proportionality_prior_sol(batch, mapping):
    """
    computes the gradient proportionality prior from the batch
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: Proportionality loss defined as average over:
    If the same action is taken at t1 and t2: (||(s_{t2+1} - s_{t2}|| - ||(s_{t1+1} - s_{t1}||)^2

    We want the gradient of this
    """

    time_steps = len(batch)
    total_loss_grad = np.zeros(mapping.shape)
    pairings = 0
    for i in range(0, time_steps - 1):
        for j in range(i+1,time_steps - 1):
            if batch[i].action == batch[j].action:
                pairings += 1
                loss_grad = proportional_loss_gradient(batch[i].image, batch[i + 1].image, batch[j].image, batch[j+1].image, mapping)
                total_loss_grad += loss_grad.reshape(mapping.shape)

    return total_loss_grad / pairings


def proportional_loss_gradient(s1, s2, s3, s4, mapping):
    dims = mapping.shape
    output = np.zeros(dims)
    delta1 = mapping@s2 - mapping@s1
    delta2 = mapping@s4 - mapping@s3
    outest = (np.linalg.norm(delta2) - np.linalg.norm(delta1))*2

    for i in range(0, dims[0]):

        outer_denom_1 = delta1[i]*np.linalg.norm(delta1)
        outer_denom_2 = delta2[i]*np.linalg.norm(delta2)
        for j in range(0, dims[1]):
            frac_1 = (delta1[i]**2)*(s1[j]- s2[j])/outer_denom_1
            frac_2 = (delta2[i]**2)*(s3[j] - s4[j])/outer_denom_2
            output[i,j] = -(frac_2 - frac_1)*outest

    return output


def causality_prior_sol(batch, mapping):
    """
    computes the gradient causality prior from the batch
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: Causality loss defined as average over:
    If the same action is taken at t1 and t2, but different rewards are received:
    e^(-||(s_{t2} - s_{t1}||)

    We want the gradient of this
    """

    time_steps = len(batch)
    total_loss_grad = np.zeros(mapping.shape)
    pairings = 0
    for i in range(0, time_steps - 1):

        for j in range(i + 1, time_steps - 1):
            if batch[i].action == batch[j].action and batch[i].reward == batch[j].reward:
                loss_grad = causal_loss_gradient(batch[i].image, batch[j].image,
                                                mapping)
                pairings += 1
                total_loss_grad += loss_grad.reshape(mapping.shape)

    print("pairings: {}".format(pairings))
    if pairings == 0:
        return np.zeros(mapping.shape)
    return total_loss_grad / pairings

def causal_loss_gradient(s1,s2,mapping):
    dims = mapping.shape
    output = np.zeros(dims)

    delta = mapping@s1 - mapping@s2
    delta_norm = np.linalg.norm(delta)
    for i in range(0,dims[0]):
        denom = 2*delta[i]*delta_norm

        for j in range(0,dims[1]):
            numer = delta[i]*np.exp(-delta_norm) * (s1[j] - s2[j])*delta[i]*2
            if denom == 0:
                output[i,j] = 0
            else:
                output[i,j] = -numer/denom
    return output

def repeatability_prior_sol(batch, mapping):
    time_steps = len(batch)
    total_loss_grad = np.zeros(mapping.shape)
    pairings = 0
    for i in range(0, time_steps - 1):
        for j in range(i + 1, time_steps - 1):
            if batch[i].action == batch[j].action:
                pairings += 1
                loss_grad = repeatability_loss_gradient(batch[i].image, batch[i + 1].image, batch[j].image,
                                                       batch[j + 1].image, mapping)
                total_loss_grad += loss_grad.reshape(mapping.shape)

    return total_loss_grad / pairings


def repeatability_loss_gradient(s1, s2, s3, s4, mapping):
    """
    :param batch:
    :return:
    """
    dims = mapping.shape
    output = np.zeros(dims)

    delta1 = (mapping@s2 - mapping@s1)
    delta2 = (mapping@s4 - mapping@s3)
    base_priorish_loss = np.linalg.norm(delta2 - delta1)**2
    base_causal_loss = np.exp(-np.linalg.norm(mapping@s3 - mapping@s1))
    causal_losses = causal_loss_gradient(s1, s3, mapping)

    for i in range(0,dims[0]):
        outer = delta2[i] - delta1[i]
        for j in range(0,dims[1]):
            inner = s4[j] - s3[j] - s2[j] + s1[j]
            output[i,j] = 2*inner*outer*base_causal_loss + causal_losses[i,j]*base_priorish_loss
    return output

def multi_prior_sol(batch, mapping):

    time_steps = len(batch)
    total_loss_grad = np.zeros(mapping.shape)
    pairings = 0
    counter = 0
    for i in range(0,time_steps-1):
        if batch[i].action[0] != batch[i].action[1]:
            counter += 1
            total_loss_grad += multi_loss_gradient(batch[i].image, batch[i+1].image, mapping)

    if counter == 0:
        return np.zeros(mapping.shape)
    return total_loss_grad/counter


def multi_loss_gradient(s1, s2, mapping):

    dims = mapping.shape
    output = np.zeros(dims)

    delta = mapping@s2 - mapping@s1

    deltadelta = delta[0:2] - delta[2:4]
    squared = np.multiply(deltadelta, deltadelta)
    expon = -squared[0] - squared[1]
    for i in range(0,dims[0]):
        outer = deltadelta[i % 2]
        for j in range(0, dims[1]):
            individual = s1[j] - s2[j]
            output[i,j] = 2*np.exp(expon)*outer*individual
    return output



if __name__ == "__main__":
    image1 = np.array([0, 1, 1, 1]).T  # (0,0)
    image2 = np.array([1, 0, 1, 1]).T  # (1,0)
    image3 = np.array([1, 1, 0, 1]).T  # (0,0)
    image4 = np.array([1, 1, 1, 0.1]).T  # (1,0)
    frame1 = DataFrame(np.array([1, 2]), 4, 1, image1)
    frame2 = DataFrame(np.array([2, 3]), 4, 2, image2)
    frame3 = DataFrame(np.array([2, 1]), 4, 1, image3)
    frame4 = DataFrame(np.array([2, 4]), 4, 2, image4)
    multiframe1 = DataFrame(np.array([1, 2]), (4,4), 1, image1)
    multiframe2 = DataFrame(np.array([1, 2]), (4,4), 1, image2)
    mapping = np.arange(16).reshape((4, 4))
    mapping[0,2] = 0
    mapping[0,1] = 0
    print(mapping)
    print(multi_prior_sol([multiframe1, multiframe2], mapping))

