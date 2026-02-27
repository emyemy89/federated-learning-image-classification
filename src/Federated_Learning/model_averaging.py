import torch


def aggregate(number_of_samples, client_state_dictionary, global_model):
    global_state_dictionary = {} # here averaged weights will be stored
    total_samples = sum(number_of_samples)
    for key in client_state_dictionary[0].keys(): # here client_state_dictionary[0].keys() is ok because all clients have the same keys:
                                                  # ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fully_connected.weight', 'fully_connected.bias']
        global_state_dictionary[key] = torch.zeros_like(client_state_dictionary[0][key]) # initialise with first client   
    for client_id, state_dictionary in enumerate(client_state_dictionary): # iterate over clients
        weight = number_of_samples[client_id] / total_samples # we normalise because different clients can have different data size
        for key in state_dictionary.keys(): # iterate over keys of each client
            global_state_dictionary[key] += state_dictionary[key] * weight # eg. global_layer = sum(client_layer * weight)
    # set global model parameters for the next step
    global_model.load_state_dict(global_state_dictionary)
    return global_model