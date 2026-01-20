# Cluster 11

@solver
def contrastive_matching(num_options: int=4) -> Solver:
    """Generate contrastive matching questions for blueprint identification."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata.get('blueprint', {})
        if 'titles' not in state.metadata or 'purposes' not in state.metadata:
            title_purpose_solver = generate_blueprint_title_and_purpose()
            state = await title_purpose_solver(state, generate)
        correct_title = state.metadata.get('title', 'Unknown Blueprint')
        correct_purpose = state.metadata.get('purpose', 'No description available')
        options = [{'title': correct_title, 'purpose': correct_purpose}]
        dummy_options = [{'title': 'Belt Balancer', 'purpose': 'Distributes items evenly across multiple belt lanes'}, {'title': 'Train Station', 'purpose': 'Automated loading and unloading point for trains'}, {'title': 'Power Plant', 'purpose': 'Generates electricity using steam engines and boilers'}]
        for i in range(min(num_options - 1, len(dummy_options))):
            options.append(dummy_options[i])
        import random
        correct_index = 0
        random.shuffle(options)
        for i, option in enumerate(options):
            if option['title'] == correct_title:
                correct_index = i
                break
        prompt = Templates.contrastive_matching(options=options)
        state.messages = [ChatMessageUser(content=prompt)]
        state.metadata['contrastive_options'] = options
        state.metadata['correct_answer'] = correct_index + 1
        return state
    return solve

@solver
def generate_blueprint_title_and_purpose() -> Solver:
    """Generate both title and purpose description for blueprints."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        blueprint = state.metadata.get('blueprint', {})
        prompt = Templates.blueprint_title_purpose(blueprint=blueprint)
        state.messages[-1] = ChatMessageUser(content=prompt)
        response = await generate(state)
        completion = response.output.completion
        pattern = '```json\\s*\\n(.*?)\\n```'
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            json_content = match.group(1)
            data = json.loads(json_content)
            title = data.get('title')
            purpose = data.get('purpose')
            state.metadata['title'] = title
            state.metadata['purpose'] = purpose
        return state
    return solve

