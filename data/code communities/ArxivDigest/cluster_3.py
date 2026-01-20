# Cluster 3

def process_subject_fields(subjects):
    all_subjects = subjects.split(';')
    all_subjects = [s.split(' (')[0] for s in all_subjects]
    return all_subjects

def generate_body(topic, categories, interest, threshold):
    if topic == 'Physics':
        raise RuntimeError('You must choose a physics subtopic.')
    elif topic in physics_topics:
        abbr = physics_topics[topic]
    elif topic in topics:
        abbr = topics[topic]
    else:
        raise RuntimeError(f'Invalid topic {topic}')
    if categories:
        for category in categories:
            if category not in category_map[topic]:
                raise RuntimeError(f'{category} is not a category of {topic}')
        papers = get_papers(abbr)
        papers = [t for t in papers if bool(set(process_subject_fields(t['subjects'])) & set(categories))]
    else:
        papers = get_papers(abbr)
    if interest:
        relevancy, hallucination = generate_relevance_score(papers, query={'interest': interest}, threshold_score=threshold, num_paper_in_prompt=16)
        body = '<br><br>'.join([f'Title: <a href="{paper['main_page']}">{paper['title']}</a><br>Authors: {paper['authors']}<br>Score: {paper['Relevancy score']}<br>Reason: {paper['Reasons for match']}' for paper in relevancy])
        if hallucination:
            body = 'Warning: the model hallucinated some papers. We have tried to remove them, but the scores may not be accurate.<br><br>' + body
    else:
        body = '<br><br>'.join([f'Title: <a href="{paper['main_page']}">{paper['title']}</a><br>Authors: {paper['authors']}' for paper in papers])
    return body

