def to_split_tensor_data(labels, data, label_index):
    import torch
    from sklearn.model_selection import train_test_split

    y = []
    X = []
    for line in data:
        y.append(labels.index(line[label_index]))
        X.append([float(s) for s in (line[:label_index]+line[label_index+1:])]) 

    y = torch.tensor(y)
    X = torch.tensor(X)

    return train_test_split(X, y, test_size=0.2)