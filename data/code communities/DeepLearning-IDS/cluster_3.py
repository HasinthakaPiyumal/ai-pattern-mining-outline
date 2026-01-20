# Cluster 3

def main(inputFile):
    results = []
    inputFile = os.path.join(folderPath, inputFile)
    outputFile = '{}.ordered'.format(inputFile)
    with open(inputFile, 'r') as fin:
        data = fin.readlines()
    for line in data:
        values = line.split()
        acc = values[1]
        std_dev = values[3]
        acc = acc.replace(':', '')
        std_dev = std_dev.replace(':', '')
        results.append([float(acc), float(std_dev)])
    results.sort(key=operator.itemgetter(1))
    results.sort(key=operator.itemgetter(0), reverse=True)
    with open(outputFile, 'w') as fout:
        for acc, std in results:
            fout.write('accuracy: {:.2f}% std_dev: {:.2f}\n'.format(acc, std))

