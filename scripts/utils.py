def to_split_tensor_data(labels, data, label_index, train_ratio, random_seed):
    import torch
    from sklearn.model_selection import train_test_split

    y = []
    X = []
    for line in data:
        y.append(labels.index(line[label_index]))
        X.append([float(s) for s in (line[:label_index]+line[label_index+1:])]) 

    y = torch.tensor(y)
    X = torch.tensor(X)

    return train_test_split(X, y, train_size=train_ratio, random_state= (random_seed if random_seed else None))

def turn_json_to_torch(model_raw):
    import torch

    model = {}
    for key in model_raw.keys():
        model[key] = torch.tensor(model_raw[key])

    return model