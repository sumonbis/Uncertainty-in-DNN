import csv

def read_result(file):
    h = []
    acc = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            h.append(int(row[0]))
            acc.append(float(row[1]))

    return h,acc