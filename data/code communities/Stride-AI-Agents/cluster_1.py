# Cluster 1

def create_website_folder(url):
    domain = urlparse(url).netloc
    folder_name = domain.split('.')[0]
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def analyze_website_content(content):
    """
    Analyze the scraped website content using OpenAI.
    This function demonstrates how to use AI for content analysis.
    """
    logging.info('Analyzing website content')
    analysis = generate_completion('marketing analyst', 'Analyze the following website content and provide key insights for marketing strategy.', content)
    return {'analysis': analysis}

def generate_completion(role, task, content):
    """
    Generate a completion using OpenAI's GPT model.
    This function demonstrates how to interact with OpenAI's API.
    """
    logging.info(f'Generating completion for {role}')
    response = client.chat.completions.create(model='gpt-4o', messages=[{'role': 'system', 'content': f'You are a {role}. {task}'}, {'role': 'user', 'content': content}])
    return response.choices[0].message.content

def create_campaign_idea(target_audience, goals):
    """
    Create a campaign idea based on target audience and goals using OpenAI.
    This function demonstrates AI's capability in strategic planning.
    """
    logging.info('Creating campaign idea')
    campaign_idea = generate_completion('marketing strategist', 'Create an innovative campaign idea based on the target audience and goals provided.', f'Target Audience: {target_audience}\nGoals: {goals}')
    return {'campaign_idea': campaign_idea}

def generate_copy(brief):
    """
    Generate marketing copy based on a brief using OpenAI.
    This function shows how AI can be used for content creation.
    """
    logging.info('Generating marketing copy')
    copy = generate_completion('copywriter', 'Create compelling marketing copy based on the following brief.', brief)
    return {'copy': copy}

