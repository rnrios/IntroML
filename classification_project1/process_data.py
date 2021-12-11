import csv


def load_access():

    X = []
    Y = []

    datafile = open('access.csv', 'r')
    file_reader = csv.reader(datafile)
    next(file_reader)
     
    for page1, page2, page3, bought in file_reader:
        
        data = [int(page1), int(page2), int(page3)]
        X.append(data)
        Y.append(int(bought))

    return X, Y


def load_search():

    X = []
    Y = []

    datafile = open('buscas.csv', 'r')
    file_reader = csv.reader(datafile)
    next(file_reader)

    for home, search, logged, bought in file_reader:
        dados = [int(home), int(search), int(logged)]
        X.append(dados)
        Y.append(int(bought))

    return X, Y




