# Cluster 3

def render_message(role, content, avatar_path):
    message_class = 'user-message' if role == 'user' else 'assistant-message'
    avatar_base64 = get_image_base64(avatar_path)
    return f'\n    <div class="chat-message {message_class}">\n        <div class="avatar">\n            <img src="data:image/png;base64,{avatar_base64}" alt="{role} avatar">\n        </div>\n        <div class="message-content">{content}</div>\n    </div>\n    '

def get_image_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode()

