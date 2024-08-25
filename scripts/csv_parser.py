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

    for index, type in enumerate(data_types):
        if (type == 'Nominal'):
            unique_values = list(set(row[index] for row in data))
            if index == label_index:
                labels = [str(unique_values.index(label)) for label in labels]
            for row in data:
                row[index] = str(unique_values.index(row[index]))

    return labels, data
