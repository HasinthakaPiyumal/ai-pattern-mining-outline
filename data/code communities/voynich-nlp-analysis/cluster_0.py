# Cluster 0

def infer_role(total, unique_words, starts, ends):
    if total > 3000 and unique_words < 500:
        return 'Function'
    elif unique_words > 1000:
        return 'Root'
    elif starts > 500 and ends > 500:
        return 'Modifier'
    elif starts > ends:
        return 'Subject Marker?'
    elif ends > starts:
        return 'Object Marker?'
    else:
        return 'Unknown'

