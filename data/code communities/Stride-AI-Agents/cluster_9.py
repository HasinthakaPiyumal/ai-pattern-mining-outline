# Cluster 9

def pretty_print_messages(messages):
    for message in messages:
        if message['content'] is None:
            continue
        print(f'{message['sender']}: {message['content']}')

