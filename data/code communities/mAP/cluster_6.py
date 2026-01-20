# Cluster 6

def find_class(class_name):
    file_list = glob.glob('*.txt')
    file_list.sort()
    file_found = False
    for txt_file in file_list:
        with open(txt_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            class_name = line.split()[0]
            if class_name == searching_class_name:
                print(' ' + txt_file)
                file_found = True
                break
    if not file_found:
        print(' No file found with that class')

