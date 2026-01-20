# Cluster 54

def get_pdf_files(directory):
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.abspath(os.path.join(root, file)))
    return pdf_files

