# Cluster 2

def generate_user_profile(age, gender, mbti, profession, topics):
    prompt = prompt_tem.format(age=age, gender=gender, mbti=mbti, profession=profession, topics=topics)
    user = rag.gen(prompt)
    user_dict = user.dict()
    user_dict['topics'] = topics
    return user_dict

def create_user_profile(i):
    age = weighted_random_age(ages, probabilities)
    print(f'Person {i + 1}: Age={age}, MBTI={mbtis[mbti_index[i]]}, Gender={genders[gender_index[i]]}, Profession={professions[profession_index[i]]}')
    try:
        return generate_user_profile(age, mbtis[mbti_index[i]], genders[gender_index[i]], professions[profession_index[i]], [topics[x] for x in topic_index[i]])
    except Exception as e:
        print(e)
        retry = 5
        while retry > 0:
            try:
                return generate_user_profile(age, mbtis[mbti_index[i]], genders[gender_index[i]], professions[profession_index[i]], [topics[x] for x in topic_index[i]])
            except Exception as e:
                print(f'{retry} times', e)
                retry -= 1
        return None

def weighted_random_age(ages, probabilities):
    ranges = []
    for age_range in ages:
        if '+' in age_range:
            start = int(age_range[:-1])
            end = start + 20
        else:
            start, end = map(int, age_range.split('-'))
        ranges.append((start, end))
    total_weight = sum(probabilities)
    rnd = random.uniform(0, total_weight)
    cumulative_weight = 0
    for i, weight in enumerate(probabilities):
        cumulative_weight += weight
        if rnd < cumulative_weight:
            start, end = ranges[i]
            return random.randint(start, end)
    return None

def create_user_profile():
    while True:
        try:
            gender = get_random_gender()
            age = get_random_age()
            mbti = get_random_mbti()
            country = get_random_country()
            profession = get_random_profession()
            topic_index_lst = get_interested_topics(mbti, age, gender, country, profession)
            topics = index_to_topics(topic_index_lst)
            profile = generate_user_profile(age, gender, mbti, profession, topics)
            profile['age'] = age
            profile['gender'] = gender
            profile['mbti'] = mbti
            profile['country'] = country
            profile['profession'] = profession
            profile['interested_topics'] = topics
            return profile
        except Exception as e:
            print(f'Profile generation failed: {e}. Retrying...')

def get_random_gender():
    return random.choices(genders, gender_ratio)[0]

def get_random_age():
    group = random.choices(age_groups, age_ratio)[0]
    if group == 'underage':
        return random.randint(10, 17)
    elif group == '18-29':
        return random.randint(18, 29)
    elif group == '30-49':
        return random.randint(30, 49)
    elif group == '50-64':
        return random.randint(50, 64)
    else:
        return random.randint(65, 100)

def get_random_mbti():
    return random.choices(mbti_types, p_mbti)[0]

def get_random_country():
    country = random.choices(countries, country_ratio)[0]
    if country == 'Other':
        response = client.chat.completions.create(model='gpt-3.5-turbo', messages=[{'role': 'system', 'content': 'Select a real country name randomly:'}])
        return response.choices[0].message.content.strip()
    return country

def get_random_profession():
    return random.choices(professions, p_professions)[0]

def get_interested_topics(mbti, age, gender, country, profession):
    prompt = f"Based on the provided personality traits, age, gender and profession, please select 2-3 topics of interest from the given list.\n    Input:\n        Personality Traits: {mbti}\n        Age: {age}\n        Gender: {gender}\n        Country: {country}\n        Profession: {profession}\n    Available Topics:\n        1. Economics: The study and management of production, distribution, and consumption of goods and services. Economics focuses on how individuals, businesses, governments, and nations make choices about allocating resources to satisfy their wants and needs, and tries to determine how these groups should organize and coordinate efforts to achieve maximum output.\n        2. IT (Information Technology): The use of computers, networking, and other physical devices, infrastructure, and processes to create, process, store, secure, and exchange all forms of electronic data. IT is commonly used within the context of business operations as opposed to personal or entertainment technologies.\n        3. Culture & Society: The way of life for an entire society, including codes of manners, dress, language, religion, rituals, norms of behavior, and systems of belief. This topic explores how cultural expressions and societal structures influence human behavior, relationships, and social norms.\n        4. General News: A broad category that includes current events, happenings, and trends across a wide range of areas such as politics, business, science, technology, and entertainment. General news provides a comprehensive overview of the latest developments affecting the world at large.\n        5. Politics: The activities associated with the governance of a country or other area, especially the debate or conflict among individuals or parties having or hoping to achieve power. Politics is often a battle over control of resources, policy decisions, and the direction of societal norms.\n        6. Business: The practice of making one's living through commerce, trade, or services. This topic encompasses the entrepreneurial, managerial, and administrative processes involved in starting, managing, and growing a business entity.\n        7. Fun: Activities or ideas that are light-hearted or amusing. This topic covers a wide range of entertainment choices and leisure activities that bring joy, laughter, and enjoyment to individuals and groups.\n    Output:\n    [list of topic numbers]\n    Ensure your output could be parsed to **list**, don't output anything else."
    response = client.chat.completions.create(model='gpt-3.5-turbo', messages=[{'role': 'system', 'content': prompt}])
    topics = response.choices[0].message.content.strip()
    return json.loads(topics)

def index_to_topics(index_lst):
    topic_dict = {'1': 'Economics', '2': 'Information Technology', '3': 'Culture & Society', '4': 'General News', '5': 'Politics', '6': 'Business', '7': 'Fun'}
    result = []
    for index in index_lst:
        topic = topic_dict[str(index)]
        result.append(topic)
    return result

