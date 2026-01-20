# Cluster 17

def test_should_return_the_original_command_when_read_throw_exception(mocker):
    mocker.patch.object(ClientConnector, 'send', side_effect=socket.error(), autospec=True)
    command_to_execute, _ = process_command_from_user(ANY_ID, ANY_USER, ANY_INPUT_COMMAND)
    assert command_to_execute == ANY_INPUT_COMMAND

def process_command_from_user(command_id, user_name, command_to_check):
    command_to_execute = send_command(command_id=command_id, user_name=user_name, command_to_check=command_to_check)
    if command_to_execute.origin_command == STOP_COMMAND:
        print(Colorize().info().append('Clai has been stopped').to_console())
        return (NOOP_COMMAND, False)
    if command_to_execute.description and command_to_execute.suggested_command == ':':
        print(command_to_execute.description)
    command_accepted_by_the_user = ask_user_prompt(command_to_execute)
    return (command_accepted_by_the_user, command_to_execute.pending_actions)

def test_should_not_print_the_suggested_dialog_when_suggested_is_the_same(mocker):
    spy_print(mocker)
    mocker.patch.object(ClaiClient, 'send', return_value=ANY_NO_ACTION, autospec=True)
    process_command_from_user(ANY_ID, ANY_USER, ANY_NO_ACTION.origin_command)
    assert print.call_count == 0

def spy_print(mocker):
    return mocker.spy(builtins, 'print')

def test_should_print_the_suggested_dialog_when_suggested_is_different(mocker):
    spy_print(mocker)
    mock_input_console(mocker, 'n')
    mocker.patch.object(ClaiClient, 'send', return_value=SUGGESTED_ACTION, autospec=True)
    process_command_from_user(ANY_ID, ANY_USER, ANY_INPUT_COMMAND)
    assert print.call_count == 1

def mock_input_console(mocker, value):
    mocker.patch('builtins.input', return_value=value)

def test_should_not_print_the_suggested_dialog_when_suggested_is_different_and_execute_is_enable(mocker):
    spy_print(mocker)
    mocker.patch.object(ClaiClient, 'send', return_value=EXECUTABLE_ACTION, autospec=True)
    process_command_from_user(ANY_ID, ANY_USER, ANY_NO_ACTION.origin_command)
    assert print.call_count == 0

def test_should_not_print_the_suggested_dialog_when_the_action_only_contains_original_command(mocker):
    spy_print(mocker)
    mocker.patch.object(ClaiClient, 'send', return_value=BASIC_ACTION, autospec=True)
    process_command_from_user(ANY_ID, ANY_USER, ANY_INPUT_COMMAND)
    assert print.call_count == 0

def test_should_not_print_the_suggested_dialog_when_suggested_command_is_empty(mocker):
    spy_print(mocker)
    empty_suggested_command = Action(origin_command=ANY_INPUT_COMMAND, suggested_command='')
    mocker.patch.object(ClaiClient, 'send', return_value=empty_suggested_command, autospec=True)
    process_command_from_user(ANY_ID, ANY_USER, ANY_INPUT_COMMAND)
    assert print.call_count == 0

def test_should_return_the_original_command_when_the_command_is_the_same(mocker):
    mocker.patch.object(ClaiClient, 'send', return_value=ANY_NO_ACTION, autospec=True)
    command_to_execute, _ = process_command_from_user(ANY_ID, ANY_USER, ANY_NO_ACTION.origin_command)
    assert command_to_execute == ANY_NO_ACTION.origin_command

def test_should_return_the_suggested_command_when_the_user_press_yes(mocker):
    mock_input_console(mocker, 'y')
    mocker.patch.object(ClaiClient, 'send', return_value=SUGGESTED_ACTION, autospec=True)
    command_to_execute, _ = process_command_from_user(ANY_ID, ANY_USER, SUGGESTED_ACTION.origin_command)
    assert command_to_execute == SUGGESTED_ACTION.suggested_command

def test_should_return_the_original_command_when_the_user_press_no(mocker):
    mock_input_console(mocker, 'n')
    mocker.patch.object(ClaiClient, 'send', return_value=SUGGESTED_ACTION, autospec=True)
    command_to_execute, _ = process_command_from_user(ANY_ID, ANY_USER, SUGGESTED_ACTION.origin_command)
    assert command_to_execute == SUGGESTED_ACTION.origin_command

def test_should_return_the_suggested_command_when_the_action_execute_true(mocker):
    mocker.patch.object(ClaiClient, 'send', return_value=EXECUTABLE_ACTION, autospec=True)
    command_to_execute, _ = process_command_from_user(ANY_ID, ANY_USER, EXECUTABLE_ACTION.origin_command)
    assert command_to_execute == EXECUTABLE_ACTION.suggested_command

def test_should_return_the_original_command_when_the_action_not_contains_suggestion(mocker):
    mocker.patch.object(ClaiClient, 'send', return_value=BASIC_ACTION, autospec=True)
    command_to_execute, _ = process_command_from_user(ANY_ID, ANY_USER, BASIC_ACTION.origin_command)
    assert command_to_execute == BASIC_ACTION.origin_command

def ask_user_prompt(command_to_execute: Action) -> Optional[str]:
    if command_to_execute.is_same_command():
        return command_to_execute.origin_command
    if command_to_execute.execute:
        return command_to_execute.suggested_command
    print(Colorize().emoji(Colorize.EMOJI_ROBOT).info().append('  Suggests: ').info().append(f'{command_to_execute.suggested_command} ').append('(y/n/e)').to_console())
    while True:
        command_input = input()
        if is_yes_command(command_input):
            return command_to_execute.suggested_command
        if is_not_command(command_input):
            return command_to_execute.origin_command
        if is_explain_command(command_input):
            print(Colorize().warning().append(f'Description: {command_to_execute.description}').to_console())
        print(Colorize().info().append('choose yes[y] or no[n] or explain[e]').to_console())

def is_yes_command(input_command):
    return input_command in ('y', 'yes')

def is_not_command(input_command):
    return input_command in ('n', 'no')

def is_explain_command(input_command):
    return input_command in ('e', 'explain')

