# Cluster 3

def assign_category(folio):
    try:
        folio_num = int(''.join(filter(str.isdigit, folio)))
        for category, r in folio_categories.items():
            if folio_num in r:
                return category
    except:
        pass
    return 'Uncategorized'

