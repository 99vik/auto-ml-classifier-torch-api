import csv

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
            data_by_labels.append(unique_values)
            if index == label_index:
                labels = [str(unique_values.index(label)) for label in labels]
            for row in data:
                row[index] = str(unique_values.index(row[index]))
        else: 
            data_by_labels.append('Number')

    return labels, data, data_by_labels
