# Cluster 73

class BedrockConverter:
    """Converts MCP message types to Amazon Bedrock API format."""

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        """Check if the given MIME type is supported by Bedrock's image API."""
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @staticmethod
    def convert_to_bedrock(multipart_msg: PromptMessageMultipart) -> MessageUnionTypeDef:
        """
        Convert a PromptMessageMultipart message to Bedrock API format.
        """
        role = multipart_msg.role
        if not multipart_msg.content:
            return {'role': role, 'content': []}
        bedrock_blocks = BedrockConverter._convert_content_items(multipart_msg.content)
        return {'role': role, 'content': bedrock_blocks}

    @staticmethod
    def convert_prompt_message_to_bedrock(message: PromptMessage) -> MessageUnionTypeDef:
        """
        Convert a standard PromptMessage to Bedrock API format.
        """
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])
        return BedrockConverter.convert_to_bedrock(multipart)

    @staticmethod
    def _convert_content_items(content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]]) -> List[ContentBlockUnionTypeDef]:
        """
        Convert a list of content items to Bedrock content blocks.
        """
        bedrock_blocks: List[ContentBlockUnionTypeDef] = []
        for content_item in content_items:
            if is_text_content(content_item):
                text = get_text(content_item)
                bedrock_blocks.append({'text': text})
            elif is_image_content(content_item):
                image_content = content_item
                if not BedrockConverter._is_supported_image_type(image_content.mimeType):
                    data_size = len(image_content.data) if image_content.data else 0
                    bedrock_blocks.append({'text': f"Image with unsupported format '{image_content.mimeType}' ({data_size} bytes)"})
                else:
                    image_data = get_image_data(image_content)
                    bedrock_blocks.append({'image': {'format': image_content.mimeType, 'source': image_data}})
            elif is_resource_content(content_item):
                block = BedrockConverter._convert_embedded_resource(content_item)
                bedrock_blocks.append(block)
        return bedrock_blocks

    @staticmethod
    def _convert_embedded_resource(resource: EmbeddedResource) -> ContentBlockUnionTypeDef:
        """
        Convert EmbeddedResource to appropriate Bedrock block type.
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = getattr(resource_content, 'uri', None)
        mime_type = BedrockConverter._determine_mime_type(resource_content)
        title = extract_title_from_uri(uri) if uri else 'resource'
        if mime_type == 'image/svg+xml':
            return BedrockConverter._convert_svg_resource(resource_content)
        elif is_image_mime_type(mime_type):
            if not BedrockConverter._is_supported_image_type(mime_type):
                return BedrockConverter._create_fallback_text(f"Image with unsupported format '{mime_type}'", resource)
            image_data = get_image_data(resource)
            if image_data:
                return {'image': {'format': mime_type, 'source': {'bytes': image_data}}}
            return BedrockConverter._create_fallback_text('Image missing data', resource)
        elif mime_type == 'application/pdf':
            if hasattr(resource_content, 'blob'):
                return {'document': {'format': 'pdf', 'name': title, 'source': {'bytes': resource_content.blob}}}
            return {'text': f'[PDF resource missing data: {title}]'}
        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if not text:
                return {'text': f'[Text content could not be extracted from {title}]'}
            return {'text': text}
        text = get_text(resource)
        if text:
            return {'text': text}
        if isinstance(resource.resource, BlobResourceContents) and hasattr(resource.resource, 'blob'):
            blob_length = len(resource.resource.blob)
            return {'text': f'Embedded Resource {getattr(uri, '_url', uri_str)} with unsupported format {mime_type} ({blob_length} characters)'}
        return BedrockConverter._create_fallback_text(f'Unsupported resource ({mime_type})', resource)

    @staticmethod
    def _determine_mime_type(resource: Union[TextResourceContents, BlobResourceContents]) -> str:
        """
        Determine the MIME type of a resource.
        """
        if getattr(resource, 'mimeType', None):
            return resource.mimeType
        if getattr(resource, 'uri', None):
            return guess_mime_type(str(resource.uri))
        if hasattr(resource, 'blob'):
            return 'application/octet-stream'
        return 'text/plain'

    @staticmethod
    def _convert_svg_resource(resource_content) -> ContentBlockUnionTypeDef:
        """
        Convert SVG resource to text block with XML code formatting.
        """
        if hasattr(resource_content, 'text'):
            svg_content = resource_content.text
            return {'text': f'```xml\n{svg_content}\n```'}
        return {'text': '[SVG content could not be extracted]'}

    @staticmethod
    def _create_fallback_text(message: str, resource: Union[TextContent, ImageContent, EmbeddedResource]) -> ContentBlockUnionTypeDef:
        """
        Create a fallback text block for unsupported resource types.
        """
        if isinstance(resource, EmbeddedResource) and hasattr(resource.resource, 'uri'):
            uri = resource.resource.uri
            return {'text': f'[{message}: {getattr(uri, '_url', str(uri))}]'}
        return {'text': f'[{message}]'}

    @staticmethod
    def convert_tool_result_to_bedrock(tool_result: CallToolResult, tool_use_id: str) -> ToolResultBlockTypeDef:
        """
        Convert an MCP CallToolResult to a Bedrock ToolResultBlockTypeDef.
        """
        bedrock_content = BedrockConverter._convert_content_items(tool_result.content)
        if not bedrock_content:
            bedrock_content = [{'text': '[No content in tool result]'}]
        return {'toolResult': {'toolUseId': tool_use_id, 'content': bedrock_content, 'status': 'error' if tool_result.isError else 'success'}}

    @staticmethod
    def create_tool_results_message(tool_results: List[tuple[str, CallToolResult]]) -> MessageUnionTypeDef:
        """
        Create a user message containing tool results.
        """
        content_blocks = []
        for tool_use_id, result in tool_results:
            bedrock_content = BedrockConverter._convert_content_items(result.content)
            if not bedrock_content:
                bedrock_content = [{'text': '[No content in tool result]'}]
            content_blocks.append({'toolResult': {'toolUseId': tool_use_id, 'content': bedrock_content, 'status': 'error' if result.isError else 'success'}})
        return {'role': 'user', 'content': content_blocks}

    @staticmethod
    def convert_mixed_messages_to_bedrock(message: MessageTypes) -> List[MessageUnionTypeDef]:
        """
        Convert a list of mixed messages to a list of Bedrock-compatible messages.

        Args:
            messages: List of mixed message objects

        Returns:
            A list of Bedrock-compatible MessageParam objects
        """
        messages: list[MessageUnionTypeDef] = []
        if isinstance(message, str):
            messages.append({'role': 'user', 'content': [{'text': message}]})
        elif isinstance(message, PromptMessage):
            messages.append(BedrockConverter.convert_prompt_message_to_bedrock(message))
        elif isinstance(message, list):
            for m in message:
                if isinstance(m, PromptMessage):
                    messages.append(BedrockConverter.convert_prompt_message_to_bedrock(m))
                elif isinstance(m, str):
                    messages.append({'role': 'user', 'content': [{'text': m}]})
                else:
                    messages.append(m)
        else:
            messages.append(message)
        return messages

def guess_mime_type(file_path: str) -> str:
    """
    Guess the MIME type of a file based on its extension.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

class OpenAIConverter:
    """Converts MCP message types to OpenAI API format."""

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        """
        Check if the given MIME type is supported by OpenAI's image API.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if the MIME type is generally supported, False otherwise
        """
        return mime_type is not None and is_image_mime_type(mime_type) and (mime_type != 'image/svg+xml')

    @staticmethod
    def convert_to_openai(multipart_msg: PromptMessageMultipart, concatenate_text_blocks: bool=False) -> Dict[str, str | ContentBlock | List[ContentBlock]]:
        """
        Convert a PromptMessageMultipart message to OpenAI API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI API message object
        """
        role = multipart_msg.role
        if not multipart_msg.content:
            return {'role': role, 'content': ''}
        if 1 == len(multipart_msg.content) and is_text_content(multipart_msg.content[0]):
            return {'role': role, 'content': get_text(multipart_msg.content[0])}
        content_blocks: List[ContentBlock] = []
        for item in multipart_msg.content:
            try:
                if is_text_content(item):
                    text = get_text(item)
                    content_blocks.append({'type': 'text', 'text': text})
                elif is_image_content(item):
                    content_blocks.append(OpenAIConverter._convert_image_content(item))
                elif is_resource_content(item):
                    block = OpenAIConverter._convert_embedded_resource(item)
                    if block:
                        content_blocks.append(block)
                else:
                    _logger.warning(f'Unsupported content type: {type(item)}')
                    fallback_text = f'[Unsupported content type: {type(item).__name__}]'
                    content_blocks.append({'type': 'text', 'text': fallback_text})
            except Exception as e:
                _logger.warning(f'Error converting content item: {e}')
                fallback_text = f'[Content conversion error: {str(e)}]'
                content_blocks.append({'type': 'text', 'text': fallback_text})
        if not content_blocks:
            return {'role': role, 'content': ''}
        if concatenate_text_blocks:
            content_blocks = OpenAIConverter._concatenate_text_blocks(content_blocks)
        return {'role': role, 'content': content_blocks}

    @staticmethod
    def _concatenate_text_blocks(blocks: List[ContentBlock]) -> List[ContentBlock]:
        """
        Combine adjacent text blocks into single blocks.

        Args:
            blocks: List of content blocks

        Returns:
            List with adjacent text blocks combined
        """
        if not blocks:
            return []
        combined_blocks: List[ContentBlock] = []
        current_text = ''
        for block in blocks:
            if block['type'] == 'text':
                if current_text:
                    current_text += ' ' + block['text']
                else:
                    current_text = block['text']
            else:
                if current_text:
                    combined_blocks.append({'type': 'text', 'text': current_text})
                    current_text = ''
                combined_blocks.append(block)
        if current_text:
            combined_blocks.append({'type': 'text', 'text': current_text})
        return combined_blocks

    @staticmethod
    def convert_prompt_message_to_openai(message: PromptMessage, concatenate_text_blocks: bool=False) -> ChatCompletionMessageParam:
        """
        Convert a standard PromptMessage to OpenAI API format.

        Args:
            message: The PromptMessage to convert
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            An OpenAI API message object
        """
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])
        return OpenAIConverter.convert_to_openai(multipart, concatenate_text_blocks)

    @staticmethod
    def _convert_image_content(content: ImageContent) -> ContentBlock:
        """Convert ImageContent to OpenAI image_url content block."""
        image_data = get_image_data(content)
        if not image_data:
            return {'type': 'text', 'text': f'[Image missing data for {content.mimeType}]'}
        image_url = {'url': f'data:{content.mimeType};base64,{image_data}'}
        if hasattr(content, 'annotations') and content.annotations:
            if hasattr(content.annotations, 'detail'):
                detail = content.annotations.detail
                if detail in ('auto', 'low', 'high'):
                    image_url['detail'] = detail
        return {'type': 'image_url', 'image_url': image_url}

    @staticmethod
    def _determine_mime_type(resource_content) -> str:
        """
        Determine the MIME type of a resource.

        Args:
            resource_content: The resource content to check

        Returns:
            The determined MIME type as a string
        """
        if hasattr(resource_content, 'mimeType') and resource_content.mimeType:
            return resource_content.mimeType
        if hasattr(resource_content, 'uri') and resource_content.uri:
            mime_type = guess_mime_type(str(resource_content.uri))
            return mime_type
        if hasattr(resource_content, 'blob'):
            return 'application/octet-stream'
        return 'text/plain'

    @staticmethod
    def _convert_embedded_resource(resource: EmbeddedResource) -> Optional[ContentBlock]:
        """
        Convert EmbeddedResource to appropriate OpenAI content block.

        Args:
            resource: The embedded resource to convert

        Returns:
            An appropriate OpenAI content block or None if conversion failed
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = getattr(resource_content, 'uri', None)
        is_url = uri and str(uri).startswith(('http://', 'https://'))
        title = extract_title_from_uri(uri) if uri else 'resource'
        mime_type = OpenAIConverter._determine_mime_type(resource_content)
        if OpenAIConverter._is_supported_image_type(mime_type):
            if is_url and uri_str:
                return {'type': 'image_url', 'image_url': {'url': uri_str}}
            image_data = get_image_data(resource)
            if image_data:
                return {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_data}'}}
            else:
                return {'type': 'text', 'text': f'[Image missing data: {title}]'}
        elif mime_type == 'application/pdf':
            if is_url and uri_str:
                return {'type': 'text', 'text': f'[PDF URL: {uri_str}]\nOpenAI requires PDF files to be uploaded or provided as base64 data.'}
            elif hasattr(resource_content, 'blob'):
                return {'type': 'file', 'file': {'filename': title or 'document.pdf', 'file_data': f'data:application/pdf;base64,{resource_content.blob}'}}
        elif mime_type == 'image/svg+xml':
            text = get_text(resource)
            if text:
                file_text = f'<mcp-agent:file title="{title}" mimetype="{mime_type}">\n{text}\n</mcp-agent:file>'
                return {'type': 'text', 'text': file_text}
        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if text:
                file_text = f'<mcp-agent:file title="{title}" mimetype="{mime_type}">\n{text}\n</mcp-agent:file>'
                return {'type': 'text', 'text': file_text}
        text = get_text(resource)
        if text:
            return {'type': 'text', 'text': text}
        elif hasattr(resource_content, 'blob'):
            return {'type': 'text', 'text': f'[Binary resource: {title} ({mime_type})]'}
        return {'type': 'text', 'text': f'[Unsupported resource: {title} ({mime_type})]'}

    @staticmethod
    def _extract_text_from_content_blocks(content: Union[str, List[ContentBlock]]) -> str:
        """
        Extract and combine text from content blocks.

        Args:
            content: Content blocks or string

        Returns:
            Combined text as a string
        """
        if isinstance(content, str):
            return content
        if not content:
            return ''
        text_parts = []
        for block in content:
            if block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
        return ' '.join(text_parts) if text_parts else '[Complex content converted to text]'

    @staticmethod
    def convert_tool_result_to_openai(tool_result: CallToolResult, tool_call_id: str, concatenate_text_blocks: bool=False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Convert a CallToolResult to an OpenAI tool message.

        If the result contains non-text elements, those are converted to separate user messages
        since OpenAI tool messages can only contain text.

        Args:
            tool_result: The tool result from a tool call
            tool_call_id: The ID of the associated tool use
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            Either a single OpenAI message for the tool response (if text only),
            or a tuple containing the tool message and a list of additional messages for non-text content
        """
        if not tool_result.content:
            return {'role': 'tool', 'tool_call_id': tool_call_id, 'content': '[No content in tool result]'}
        text_content = []
        non_text_content = []
        for item in tool_result.content:
            if isinstance(item, TextContent):
                text_content.append(item)
            else:
                non_text_content.append(item)
        tool_message_content = ''
        if text_content:
            temp_multipart = PromptMessageMultipart(role='user', content=text_content)
            converted = OpenAIConverter.convert_to_openai(temp_multipart, concatenate_text_blocks=concatenate_text_blocks)
            tool_message_content = OpenAIConverter._extract_text_from_content_blocks(converted.get('content', ''))
        if not tool_message_content:
            tool_message_content = '[Tool returned non-text content]'
        tool_message = {'role': 'tool', 'tool_call_id': tool_call_id, 'content': tool_message_content}
        if not non_text_content:
            return tool_message
        non_text_multipart = PromptMessageMultipart(role='user', content=non_text_content)
        user_message = OpenAIConverter.convert_to_openai(non_text_multipart)
        user_message['tool_call_id'] = tool_call_id
        return (tool_message, [user_message])

    @staticmethod
    def convert_function_results_to_openai(results: List[Tuple[str, CallToolResult]], concatenate_text_blocks: bool=False) -> List[Dict[str, Any]]:
        """
        Convert a list of function call results to OpenAI messages.

        Args:
            results: List of (tool_call_id, result) tuples
            concatenate_text_blocks: If True, adjacent text blocks will be combined

        Returns:
            List of OpenAI API messages for tool responses
        """
        messages = []
        for tool_call_id, result in results:
            converted = OpenAIConverter.convert_tool_result_to_openai(tool_result=result, tool_call_id=tool_call_id, concatenate_text_blocks=concatenate_text_blocks)
            if isinstance(converted, tuple):
                tool_message, additional_messages = converted
                messages.append(tool_message)
                messages.extend(additional_messages)
            else:
                messages.append(converted)
        return messages

    @staticmethod
    def convert_mixed_messages_to_openai(message: MessageTypes) -> List[ChatCompletionMessageParam]:
        """
        Convert a list of mixed messages to a list of OpenAI-compatible messages.

        Args:
            messages: List of mixed message objects

        Returns:
            A list of OpenAI-compatible MessageParam objects
        """
        messages: list[ChatCompletionMessageParam] = []
        if isinstance(message, str):
            messages.append(ChatCompletionUserMessageParam(role='user', content=message))
        elif isinstance(message, PromptMessage):
            messages.append(OpenAIConverter.convert_prompt_message_to_openai(message))
        elif isinstance(message, list):
            for m in message:
                if isinstance(m, PromptMessage):
                    messages.append(OpenAIConverter.convert_prompt_message_to_openai(m))
                elif isinstance(m, str):
                    messages.append(ChatCompletionUserMessageParam(role='user', content=m))
                else:
                    messages.append(m)
        else:
            messages.append(message)
        return messages

class AzureConverter:
    """Converts MCP message types to Azure API format."""

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @staticmethod
    def convert_to_azure(multipart_msg: PromptMessageMultipart) -> UserMessage | AssistantMessage:
        """
        Convert a PromptMessageMultipart message to Azure API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert

        Returns:
            An Azure UserMessage or AssistantMessage object
        """
        role = multipart_msg.role
        if not multipart_msg.content:
            if role == 'assistant':
                return AssistantMessage(content='')
            else:
                return UserMessage(content='')
        azure_blocks = AzureConverter._convert_content_items(multipart_msg.content)
        if role == 'assistant':
            text_blocks = []
            for block in azure_blocks:
                if isinstance(block, TextContentItem):
                    text_blocks.append(block.text)
                else:
                    _logger.warning(f'Removing non-text block from assistant message: {type(block)}')
            content = '\n'.join(text_blocks)
            return AssistantMessage(content=content)
        else:
            content = azure_blocks
            return UserMessage(content=content)

    @staticmethod
    def convert_prompt_message_to_azure(message: PromptMessage) -> UserMessage | AssistantMessage:
        """
        Convert a standard PromptMessage to Azure API format.

        Args:
            message: The PromptMessage to convert

        Returns:
            An Azure UserMessage or AssistantMessage object
        """
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])
        return AzureConverter.convert_to_azure(multipart)

    @staticmethod
    def _convert_content_items(content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]]) -> List[ContentItem]:
        """
        Convert a list of content items to Azure content blocks.

        Args:
            content_items: Sequence of MCP content items

        Returns:
            List of Azure ContentItem
        """
        azure_blocks: List[ContentItem] = []
        for content_item in content_items:
            if is_text_content(content_item):
                text = get_text(content_item)
                if text:
                    azure_blocks.append(TextContentItem(text=text))
            elif is_image_content(content_item):
                image_content = content_item
                if not AzureConverter._is_supported_image_type(image_content.mimeType):
                    data_size = len(image_content.data) if image_content.data else 0
                    azure_blocks.append(TextContentItem(text=f"Image with unsupported format '{image_content.mimeType}' ({data_size} bytes)"))
                else:
                    image_data = get_image_data(image_content)
                    data_url = f'data:{image_content.mimeType};base64,{image_data}'
                    azure_blocks.append(ImageContentItem(image_url=ImageUrl(url=data_url)))
            elif is_resource_content(content_item):
                block = AzureConverter._convert_embedded_resource(content_item)
                if block is not None:
                    azure_blocks.append(block)
        return azure_blocks

    @staticmethod
    def _convert_embedded_resource(resource: EmbeddedResource) -> Optional[ContentItem]:
        """
        Convert EmbeddedResource to appropriate Azure ContentItem.

        Args:
            resource: The embedded resource to convert

        Returns:
            An appropriate ContentItem for the resource, or None if not convertible
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = getattr(resource_content, 'uri', None)
        is_url: bool = uri and getattr(uri, 'scheme', None) in ('http', 'https')
        mime_type = AzureConverter._determine_mime_type(resource_content)
        title = extract_title_from_uri(uri) if uri else 'resource'
        if mime_type == 'image/svg+xml':
            return AzureConverter._convert_svg_resource(resource_content)
        elif is_image_mime_type(mime_type):
            if not AzureConverter._is_supported_image_type(mime_type):
                return AzureConverter._create_fallback_text(f"Image with unsupported format '{mime_type}'", resource)
            if is_url and uri_str:
                return ImageContentItem(image_url=ImageUrl(url=uri_str))
            image_data = get_image_data(resource)
            if image_data:
                data_url = f'data:{mime_type};base64,{image_data}'
                return ImageContentItem(image_url=ImageUrl(url=data_url))
            return AzureConverter._create_fallback_text('Image missing data', resource)
        elif mime_type == 'application/pdf':
            return TextContentItem(text=f'[PDF resource: {title}]')
        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if not text:
                return TextContentItem(text=f'[Text content could not be extracted from {title}]')
            return TextContentItem(text=text)
        text = get_text(resource)
        if text:
            return TextContentItem(text=text)
        if isinstance(resource.resource, BlobResourceContents) and hasattr(resource.resource, 'blob'):
            blob_length = len(resource.resource.blob)
            return TextContentItem(text=f'Embedded Resource {getattr(uri, '_url', '')} with unsupported format {mime_type} ({blob_length} characters)')
        return AzureConverter._create_fallback_text(f'Unsupported resource ({mime_type})', resource)

    @staticmethod
    def _determine_mime_type(resource: Union[TextResourceContents, BlobResourceContents]) -> str:
        if getattr(resource, 'mimeType', None):
            return resource.mimeType
        if getattr(resource, 'uri', None):
            return guess_mime_type(str(resource.uri))
        if hasattr(resource, 'blob'):
            return 'application/octet-stream'
        return 'text/plain'

    @staticmethod
    def _convert_svg_resource(resource_content) -> TextContentItem:
        if hasattr(resource_content, 'text'):
            svg_content = resource_content.text
            return TextContentItem(text=f'```xml\n{svg_content}\n```')
        return TextContentItem(text='[SVG content could not be extracted]')

    @staticmethod
    def _create_fallback_text(message: str, resource: Union[TextContent, ImageContent, EmbeddedResource]) -> TextContentItem:
        if isinstance(resource, EmbeddedResource) and hasattr(resource.resource, 'uri'):
            uri = resource.resource.uri
            return TextContentItem(text=f'[{message}: {getattr(uri, '_url', '')}]')
        return TextContentItem(text=f'[{message}]')

    @staticmethod
    def convert_tool_result_to_azure(tool_result: CallToolResult, tool_use_id: str) -> ToolMessage:
        """
        Convert an MCP CallToolResult to an Azure ToolMessage.

        Args:
            tool_result: The tool result from a tool call
            tool_use_id: The ID of the associated tool use

        Returns:
            An Azure ToolMessage containing the tool result content as text.
        """
        azure_content = []
        for item in tool_result.content:
            if isinstance(item, EmbeddedResource):
                resource_block = AzureConverter._convert_embedded_resource(item)
                if resource_block is not None:
                    azure_content.append(resource_block)
            elif isinstance(item, (TextContent, ImageContent)):
                blocks = AzureConverter._convert_content_items([item])
                azure_content.extend(blocks)
        if not azure_content:
            azure_content = [TextContentItem(text='[No content in tool result]')]
        content_text = AzureConverter._extract_text_from_azure_content_blocks(azure_content)
        return ToolMessage(tool_call_id=tool_use_id, content=content_text)

    @staticmethod
    def _extract_text_from_azure_content_blocks(blocks: list[TextContentItem | ImageContentItem | AudioContentItem]) -> str:
        """
        Extract and concatenate text from Azure content blocks for ToolMessage.
        """
        texts = []
        for block in blocks:
            if hasattr(block, 'text') and isinstance(block.text, str):
                texts.append(block.text)
            elif hasattr(block, 'image_url'):
                url = getattr(block.image_url, 'url', None)
                if url:
                    texts.append(f'[Image: {url}]')
                else:
                    texts.append('[Image]')
            else:
                texts.append(str(block))
        return '\n'.join(texts)

    @staticmethod
    def create_tool_results_message(tool_results: List[tuple[str, CallToolResult]]) -> List[ToolMessage]:
        """
        Create a list of ToolMessage objects for tool results.

        Args:
            tool_results: List of (tool_use_id, tool_result) tuples

        Returns:
            A list of ToolMessage objects, one for each tool result.
        """
        tool_messages = []
        for tool_use_id, result in tool_results:
            tool_message = AzureConverter.convert_tool_result_to_azure(result, tool_use_id)
            tool_messages.append(tool_message)
        return tool_messages

    @staticmethod
    def convert_mixed_messages_to_azure(message: MessageTypes) -> List[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage, DeveloperMessage]]:
        """
        Convert a list of mixed messages to a list of Azure-compatible messages.

        Args:
            messages: List of mixed message objects

        Returns:
            A list of Azure-compatible MessageParam objects
        """
        messages = []
        if isinstance(message, str):
            messages.append(UserMessage(content=message))
        elif isinstance(message, PromptMessage):
            messages.append(AzureConverter.convert_prompt_message_to_azure(message))
        elif isinstance(message, list):
            for m in message:
                if isinstance(m, PromptMessage):
                    messages.append(AzureConverter.convert_prompt_message_to_azure(m))
                elif isinstance(m, str):
                    messages.append(UserMessage(content=m))
                else:
                    messages.append(m)
        else:
            messages.append(message)
        return messages

class AnthropicConverter:
    """Converts MCP message types to Anthropic API format."""

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        """Check if the given MIME type is supported by Anthropic's image API.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if the MIME type is supported, False otherwise
        """
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @staticmethod
    def convert_to_anthropic(multipart_msg: PromptMessageMultipart) -> MessageParam:
        """
        Convert a PromptMessageMultipart message to Anthropic API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert

        Returns:
            An Anthropic API MessageParam object
        """
        role = multipart_msg.role
        if not multipart_msg.content:
            return MessageParam(role=role, content=[])
        anthropic_blocks = AnthropicConverter._convert_content_items(multipart_msg.content, document_mode=True)
        if role == 'assistant':
            text_blocks = []
            for block in anthropic_blocks:
                if block.get('type') == 'text':
                    text_blocks.append(block)
                else:
                    _logger.warning(f'Removing non-text block from assistant message: {block.get('type')}')
            anthropic_blocks = text_blocks
        return MessageParam(role=role, content=anthropic_blocks)

    @staticmethod
    def convert_prompt_message_to_anthropic(message: PromptMessage) -> MessageParam:
        """
        Convert a standard PromptMessage to Anthropic API format.

        Args:
            message: The PromptMessage to convert

        Returns:
            An Anthropic API MessageParam object
        """
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])
        return AnthropicConverter.convert_to_anthropic(multipart)

    @staticmethod
    def _convert_content_items(content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]], document_mode: bool=True) -> List[ContentBlockParam]:
        """
        Convert a list of content items to Anthropic content blocks.

        Args:
            content_items: Sequence of MCP content items
            document_mode: Whether to convert text resources to document blocks (True) or text blocks (False)

        Returns:
            List of Anthropic content blocks
        """
        anthropic_blocks: List[ContentBlockParam] = []
        for content_item in content_items:
            if is_text_content(content_item):
                text = get_text(content_item)
                if text:
                    anthropic_blocks.append(TextBlockParam(type='text', text=text))
            elif is_image_content(content_item):
                image_content = content_item
                if not AnthropicConverter._is_supported_image_type(image_content.mimeType):
                    data_size = len(image_content.data) if image_content.data else 0
                    anthropic_blocks.append(TextBlockParam(type='text', text=f"Image with unsupported format '{image_content.mimeType}' ({data_size} bytes)"))
                else:
                    image_data = get_image_data(image_content)
                    if image_data:
                        anthropic_blocks.append(ImageBlockParam(type='image', source=Base64ImageSourceParam(type='base64', media_type=image_content.mimeType, data=image_data)))
                    else:
                        anthropic_blocks.append(TextBlockParam(type='text', text=f'[Image missing data for {image_content.mimeType}]'))
            elif is_resource_content(content_item):
                block = AnthropicConverter._convert_embedded_resource(content_item, document_mode)
                anthropic_blocks.append(block)
        return anthropic_blocks

    @staticmethod
    def _convert_embedded_resource(resource: EmbeddedResource, document_mode: bool=True) -> ContentBlockParam:
        """
        Convert EmbeddedResource to appropriate Anthropic block type.

        Args:
            resource: The embedded resource to convert
            document_mode: Whether to convert text resources to Document blocks (True) or Text blocks (False)

        Returns:
            An appropriate ContentBlockParam for the resource
        """
        resource_content = resource.resource
        uri_str = get_resource_uri(resource)
        uri = getattr(resource_content, 'uri', None)
        is_url: bool = uri and uri.scheme in ('http', 'https')
        mime_type = AnthropicConverter._determine_mime_type(resource_content)
        title = extract_title_from_uri(uri) if uri else 'resource'
        if mime_type == 'image/svg+xml':
            return AnthropicConverter._convert_svg_resource(resource_content)
        elif is_image_mime_type(mime_type):
            if not AnthropicConverter._is_supported_image_type(mime_type):
                return AnthropicConverter._create_fallback_text(f"Image with unsupported format '{mime_type}'", resource)
            if is_url and uri_str:
                return ImageBlockParam(type='image', source=URLImageSourceParam(type='url', url=uri_str))
            image_data = get_image_data(resource)
            if image_data:
                return ImageBlockParam(type='image', source=Base64ImageSourceParam(type='base64', media_type=mime_type, data=image_data))
            return AnthropicConverter._create_fallback_text('Image missing data', resource)
        elif mime_type == 'application/pdf':
            if is_url and uri_str:
                return DocumentBlockParam(type='document', title=title, source=URLPDFSourceParam(type='url', url=uri_str))
            elif hasattr(resource_content, 'blob'):
                return DocumentBlockParam(type='document', title=title, source=Base64PDFSourceParam(type='base64', media_type='application/pdf', data=resource_content.blob))
            return TextBlockParam(type='text', text=f'[PDF resource missing data: {title}]')
        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if not text:
                return TextBlockParam(type='text', text=f'[Text content could not be extracted from {title}]')
            if document_mode:
                return DocumentBlockParam(type='document', title=title, source=PlainTextSourceParam(type='text', media_type='text/plain', data=text))
            return TextBlockParam(type='text', text=text)
        text = get_text(resource)
        if text:
            return TextBlockParam(type='text', text=text)
        if isinstance(resource.resource, BlobResourceContents) and hasattr(resource.resource, 'blob'):
            blob_length = len(resource.resource.blob)
            return TextBlockParam(type='text', text=f'Embedded Resource {str(uri)} with unsupported format {mime_type} ({blob_length} characters)')
        return AnthropicConverter._create_fallback_text(f'Unsupported resource ({mime_type})', resource)

    @staticmethod
    def _determine_mime_type(resource: Union[TextResourceContents, BlobResourceContents]) -> str:
        """
        Determine the MIME type of a resource.

        Args:
            resource: The resource to check

        Returns:
            The MIME type as a string
        """
        if getattr(resource, 'mimeType', None):
            return resource.mimeType
        if getattr(resource, 'uri', None):
            return guess_mime_type(str(resource.uri))
        if hasattr(resource, 'blob'):
            return 'application/octet-stream'
        return 'text/plain'

    @staticmethod
    def _convert_svg_resource(resource_content) -> TextBlockParam:
        """
        Convert SVG resource to text block with XML code formatting.

        Args:
            resource_content: The resource content containing SVG data

        Returns:
            A TextBlockParam with formatted SVG content
        """
        if hasattr(resource_content, 'text'):
            svg_content = resource_content.text
            return TextBlockParam(type='text', text=f'```xml\n{svg_content}\n```')
        return TextBlockParam(type='text', text='[SVG content could not be extracted]')

    @staticmethod
    def _create_fallback_text(message: str, resource: Union[TextContent, ImageContent, EmbeddedResource]) -> TextBlockParam:
        """
        Create a fallback text block for unsupported resource types.

        Args:
            message: The fallback message
            resource: The resource that couldn't be converted

        Returns:
            A TextBlockParam with the fallback message
        """
        if isinstance(resource, EmbeddedResource) and hasattr(resource.resource, 'uri'):
            uri = resource.resource.uri
            return TextBlockParam(type='text', text=f'[{message}: {str(uri)}]')
        return TextBlockParam(type='text', text=f'[{message}]')

    @staticmethod
    def convert_tool_result_to_anthropic(tool_result: CallToolResult, tool_use_id: str) -> ToolResultBlockParam:
        """
        Convert an MCP CallToolResult to an Anthropic ToolResultBlockParam.

        Args:
            tool_result: The tool result from a tool call
            tool_use_id: The ID of the associated tool use

        Returns:
            An Anthropic ToolResultBlockParam ready to be included in a user message
        """
        anthropic_content = []
        for item in tool_result.content:
            if isinstance(item, EmbeddedResource):
                resource_block = AnthropicConverter._convert_embedded_resource(item, document_mode=False)
                anthropic_content.append(resource_block)
            elif isinstance(item, (TextContent, ImageContent)):
                blocks = AnthropicConverter._convert_content_items([item], document_mode=False)
                anthropic_content.extend(blocks)
        if not anthropic_content:
            anthropic_content = [TextBlockParam(type='text', text='[No content in tool result]')]
        return ToolResultBlockParam(type='tool_result', tool_use_id=tool_use_id, content=anthropic_content, is_error=tool_result.isError)

    @staticmethod
    def create_tool_results_message(tool_results: List[tuple[str, CallToolResult]]) -> MessageParam:
        """
        Create a user message containing tool results.

        Args:
            tool_results: List of (tool_use_id, tool_result) tuples

        Returns:
            A MessageParam with role='user' containing all tool results
        """
        content_blocks = []
        for tool_use_id, result in tool_results:
            tool_result_blocks = []
            separate_blocks = []
            for item in result.content:
                if isinstance(item, (TextContent, ImageContent)):
                    blocks = AnthropicConverter._convert_content_items([item], document_mode=False)
                    tool_result_blocks.extend(blocks)
                elif isinstance(item, EmbeddedResource):
                    resource_content = item.resource
                    if isinstance(resource_content, TextResourceContents):
                        block = AnthropicConverter._convert_embedded_resource(item, document_mode=False)
                        tool_result_blocks.append(block)
                    else:
                        block = AnthropicConverter._convert_embedded_resource(item, document_mode=True)
                        separate_blocks.append(block)
            if tool_result_blocks:
                content_blocks.append(ToolResultBlockParam(type='tool_result', tool_use_id=tool_use_id, content=tool_result_blocks, is_error=result.isError))
            else:
                content_blocks.append(ToolResultBlockParam(type='tool_result', tool_use_id=tool_use_id, content=[TextBlockParam(type='text', text='[No content in tool result]')], is_error=result.isError))
            content_blocks.extend(separate_blocks)
        return MessageParam(role='user', content=content_blocks)

    @staticmethod
    def convert_mixed_messages_to_anthropic(message: MessageTypes) -> List[MessageParam]:
        """
        Convert a list of mixed messages to a list of Anthropic-compatible messages.

        Args:
            messages: List of mixed message objects

        Returns:
            A list of Anthropic-compatible MessageParam objects
        """
        messages: list[MessageParam] = []
        if isinstance(message, str):
            messages.append(MessageParam(role='user', content=message))
        elif isinstance(message, PromptMessage):
            messages.append(AnthropicConverter.convert_prompt_message_to_anthropic(message))
        elif isinstance(message, list):
            for m in message:
                if isinstance(m, PromptMessage):
                    messages.append(AnthropicConverter.convert_prompt_message_to_anthropic(m))
                elif isinstance(m, str):
                    messages.append(MessageParam(role='user', content=m))
                else:
                    messages.append(m)
        else:
            messages.append(message)
        return messages

class GoogleConverter:
    """Converts MCP message types to Google API format."""

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        """Check if the given MIME type is supported by Google's image API.

        Args:
            mime_type: The MIME type to check

        Returns:
            True if the MIME type is supported, False otherwise
        """
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @staticmethod
    def convert_to_google(multipart_msg: PromptMessageMultipart) -> types.Content:
        """
        Convert a PromptMessageMultipart message to Google API format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert

        Returns:
            A Google API Content object
        """
        role = multipart_msg.role
        if not multipart_msg.content:
            return types.Content(role=role, parts=[])
        google_parts = GoogleConverter._convert_content_items(multipart_msg.content)
        return types.Content(role=role, parts=google_parts)

    @staticmethod
    def convert_prompt_message_to_google(message: PromptMessage) -> types.Content:
        """
        Convert a standard PromptMessage to Google API format.

        Args:
            message: The PromptMessage to convert

        Returns:
            A Google API Content object
        """
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])
        return GoogleConverter.convert_to_google(multipart)

    @staticmethod
    def _convert_content_items(content_items: Sequence[Union[TextContent, ImageContent, EmbeddedResource]]) -> List[types.Part]:
        """
        Convert a list of content items to Google content parts.

        Args:
            content_items: Sequence of MCP content items

        Returns:
            List of Google content parts
        """
        google_parts: List[types.Part] = []
        for content_item in content_items:
            if is_text_content(content_item):
                text = get_text(content_item)
                google_parts.append(types.Part.from_text(text=text))
            elif is_image_content(content_item):
                image_content = content_item
                if not GoogleConverter._is_supported_image_type(image_content.mimeType):
                    data_size = len(image_content.data) if image_content.data else 0
                    google_parts.append(types.Part.from_text(text=f"Image with unsupported format '{image_content.mimeType}' ({data_size} bytes)"))
                else:
                    image_data = get_image_data(image_content)
                    if image_data:
                        google_parts.append(types.Part.from_bytes(data=base64.b64decode(image_data), mime_type=image_content.mimeType))
                    else:
                        google_parts.append(types.Part.from_text(text=f"Image missing data for '{image_content.mimeType}'"))
            elif is_resource_content(content_item):
                part = GoogleConverter._convert_embedded_resource(content_item)
                google_parts.append(part)
        return google_parts

    @staticmethod
    def _convert_embedded_resource(resource: EmbeddedResource) -> types.Part:
        """
        Convert EmbeddedResource to appropriate Google Part.

        Args:
            resource: The embedded resource to convert

        Returns:
            A Google Part for the resource
        """
        resource_content = resource.resource
        uri = getattr(resource_content, 'uri', None)
        mime_type = GoogleConverter._determine_mime_type(resource_content)
        title = extract_title_from_uri(uri) if uri else 'resource'
        if mime_type == 'image/svg+xml':
            return GoogleConverter._convert_svg_resource(resource_content)
        elif is_image_mime_type(mime_type):
            if not GoogleConverter._is_supported_image_type(mime_type):
                return GoogleConverter._create_fallback_text(f"Image with unsupported format '{mime_type}'", resource)
            image_data = get_image_data(resource)
            if image_data:
                return types.Part.from_bytes(data=base64.b64decode(image_data), mime_type=mime_type)
            else:
                return GoogleConverter._create_fallback_text('Image missing data', resource)
        elif mime_type == 'application/pdf':
            if hasattr(resource_content, 'blob'):
                return types.Part.from_bytes(data=base64.b64decode(resource_content.blob), mime_type='application/pdf')
            return types.Part.from_text(text=f'[PDF resource missing data: {title}]')
        elif is_text_mime_type(mime_type):
            text = get_text(resource)
            if text:
                return types.Part.from_text(text=text)
            else:
                return types.Part.from_text(text=f'[Text content could not be extracted from {title}]')
        text = get_text(resource)
        if text:
            return types.Part.from_text(text=text)
        if isinstance(resource.resource, BlobResourceContents) and hasattr(resource.resource, 'blob'):
            blob_length = len(resource.resource.blob)
            return types.Part.from_text(text=f'Embedded Resource {str(uri)} with unsupported format {mime_type} ({blob_length} characters)')
        return GoogleConverter._create_fallback_text(f'Unsupported resource ({mime_type})', resource)

    @staticmethod
    def _determine_mime_type(resource: Union[TextResourceContents, BlobResourceContents]) -> str:
        """
        Determine the MIME type of a resource.

        Args:
            resource: The resource to check

        Returns:
            The MIME type as a string
        """
        if getattr(resource, 'mimeType', None):
            return resource.mimeType
        if getattr(resource, 'uri', None):
            return guess_mime_type(str(resource.uri))
        if hasattr(resource, 'blob'):
            return 'application/octet-stream'
        return 'text/plain'

    @staticmethod
    def _convert_svg_resource(resource_content) -> types.Part:
        """
        Convert SVG resource to text part with XML code formatting.

        Args:
            resource_content: The resource content containing SVG data

        Returns:
            A types.Part with formatted SVG content
        """
        if hasattr(resource_content, 'text'):
            svg_content = resource_content.text
            return types.Part.from_text(text=f'```xml\n{svg_content}\n```')
        return types.Part.from_text(text='[SVG content could not be extracted]')

    @staticmethod
    def _create_fallback_text(message: str, resource: Union[TextContent, ImageContent, EmbeddedResource]) -> types.Part:
        """
        Create a fallback text part for unsupported resource types.

        Args:
            message: The fallback message
            resource: The resource that couldn't be converted

        Returns:
            A types.Part with the fallback message
        """
        if isinstance(resource, EmbeddedResource) and hasattr(resource.resource, 'uri'):
            uri = resource.resource.uri
            return types.Part.from_text(text=f'[{message}: {str(uri)}]')
        return types.Part.from_text(text=f'[{message}]')

    @staticmethod
    def convert_tool_result_to_google(tool_result: CallToolResult, tool_use_id: str) -> types.Part:
        """
        Convert an MCP CallToolResult to a Google function response part.

        Args:
            tool_result: The tool result from a tool call
            tool_use_id: The ID of the associated tool use

        Returns:
            A Google function response part
        """
        google_content = []
        for item in tool_result.content:
            if isinstance(item, EmbeddedResource):
                part = GoogleConverter._convert_embedded_resource(item)
                google_content.append(part)
            elif isinstance(item, (TextContent, ImageContent)):
                parts = GoogleConverter._convert_content_items([item])
                google_content.extend(parts)
        if not google_content:
            google_content = [types.Part.from_text(text='[No content in tool result]')]
        serialized_parts = [part.to_json_dict() for part in google_content]
        function_response = {'content': serialized_parts}
        if tool_result.isError:
            function_response['error'] = str(tool_result.content)
        return types.Part.from_function_response(name=tool_use_id, response=function_response)

    @staticmethod
    def create_tool_results_message(tool_results: List[tuple[str, CallToolResult]]) -> types.Content:
        """
        Create a user message containing tool results.

        Args:
            tool_results: List of (tool_use_id, tool_result) tuples

        Returns:
            A Content with role='user' containing all tool results
        """
        parts = []
        for tool_use_id, result in tool_results:
            part = GoogleConverter.convert_tool_result_to_google(result, tool_use_id)
            parts.append(part)
        return types.Content(role='user', parts=parts)

    @staticmethod
    def convert_mixed_messages_to_google(message: MessageTypes) -> List[types.Content]:
        """
        Convert a list of mixed messages to a list of Google-compatible messages.

        Args:
            messages: List of mixed message objects

        Returns:
            A list of Google-compatible message objects
        """
        messages: list[types.Content] = []
        if isinstance(message, str):
            messages.append(types.Content(role='user', parts=[types.Part.from_text(text=message)]))
        elif isinstance(message, PromptMessage):
            messages.append(GoogleConverter.convert_prompt_message_to_google(message))
        elif isinstance(message, list):
            for m in message:
                if isinstance(m, PromptMessage):
                    messages.append(GoogleConverter.convert_prompt_message_to_google(m))
                elif isinstance(m, str):
                    messages.append(types.Content(role='user', parts=[types.Part.from_text(text=m)]))
                else:
                    messages.append(m)
        else:
            messages.append(message)
        return messages

class TestGuessMimeType:

    def test_guess_mime_type_python_file(self):
        assert guess_mime_type('script.py') == 'text/x-python'

    def test_guess_mime_type_json_file(self):
        assert guess_mime_type('data.json') == 'application/json'

    def test_guess_mime_type_txt_file(self):
        assert guess_mime_type('readme.txt') == 'text/plain'

    def test_guess_mime_type_html_file(self):
        assert guess_mime_type('index.html') == 'text/html'

    def test_guess_mime_type_png_file(self):
        assert guess_mime_type('image.png') == 'image/png'

    def test_guess_mime_type_webp_file(self):
        assert guess_mime_type('image.webp') == 'image/webp'

    def test_guess_mime_type_unknown_extension(self):
        assert guess_mime_type('file.unknown') == 'application/octet-stream'

    def test_guess_mime_type_no_extension(self):
        assert guess_mime_type('filename') == 'application/octet-stream'

