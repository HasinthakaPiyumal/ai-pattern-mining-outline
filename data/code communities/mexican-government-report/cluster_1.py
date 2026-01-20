# Cluster 1

def extract_text():
    """Read the PDF contents and extract the text that we need."""
    reader = PyPDF2.PdfFileReader('informe.pdf')
    full_text = ''
    pdf_page_number = 3
    for i in range(14, 327):
        if pdf_page_number <= 9:
            page_text = reader.getPage(i).extractText().strip()[1:]
        elif pdf_page_number >= 10 and pdf_page_number <= 99:
            page_text = reader.getPage(i).extractText().strip()[2:]
        else:
            page_text = reader.getPage(i).extractText().strip()[3:]
        full_text += page_text.replace('\n', '')
        pdf_page_number += 1
    for item, replacement in CHARACTERS.items():
        full_text = full_text.replace(item, replacement)
    full_text = re.sub(' +', ' ', full_text)
    with open('transcript_clean.txt', 'w', encoding='utf-8') as temp_file:
        temp_file.write(full_text)

