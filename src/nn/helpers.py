import csv

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

def determine_type(value):
    try:
        float(value)
        return 'Float'
    except ValueError:
        return 'Nominal'
    
def csv_parser(label_index, file):
    data = []
    labels = []
    data_types = []
    data_by_labels = []
    file_contents = file.read().decode('utf-8')
    reader = csv.reader(file_contents.splitlines(), delimiter=',')

    for index, line in enumerate(reader):

        if index == 0:
            continue
        else:
            if (line[label_index] not in labels):
                labels.append(line[label_index])
            data.append(line)

    for value in data[0]:
        data_types.append(determine_type(value))

    data_types[label_index] = 'Nominal'

    for index, type in enumerate(data_types):
        if (type == 'Nominal'):
            unique_values = list(set(row[index] for row in data))
            unique_values.sort()
            data_by_labels.append(unique_values)
            if index == label_index:
                labels = [str(unique_values.index(label)) for label in labels]
            for row in data:
                row[index] = str(unique_values.index(row[index]))
        else: 
            data_by_labels.append('Number')

    return labels, data, data_by_labels