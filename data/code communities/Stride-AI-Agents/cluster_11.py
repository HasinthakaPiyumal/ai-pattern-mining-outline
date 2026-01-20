# Cluster 11

@pytest.mark.parametrize('query,function_name', [('I want to make a refund!', 'transfer_to_refunds'), ('I want to talk to sales.', 'transfer_to_sales')])
def test_triage_agent_calls_correct_function(query, function_name):
    tool_calls = run_and_get_tool_calls(triage_agent, query)
    assert len(tool_calls) == 1
    assert tool_calls[0]['function']['name'] == function_name

def run_and_get_tool_calls(agent, query):
    message = {'role': 'user', 'content': query}
    response = client.run(agent=agent, messages=[message], execute_tools=False)
    return response.messages[-1].get('tool_calls')

@pytest.mark.parametrize('query', ["What's the weather in NYC?", 'Tell me the weather in London.', "Do I need an umbrella today? I'm in chicago."])
def test_calls_weather_when_asked(query):
    tool_calls = run_and_get_tool_calls(weather_agent, query)
    assert len(tool_calls) == 1
    assert tool_calls[0]['function']['name'] == 'get_weather'

@pytest.mark.parametrize('query', ["Who's the president of the United States?", 'What is the time right now?', 'Hi!'])
def test_does_not_call_weather_when_not_asked(query):
    tool_calls = run_and_get_tool_calls(weather_agent, query)
    assert not tool_calls

