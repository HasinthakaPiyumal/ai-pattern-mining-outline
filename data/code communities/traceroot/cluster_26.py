# Cluster 26

def collect_spans_latency_recursively(spans: list[Span], spans_latency_dict: dict[str, float]) -> None:
    """Recursively collect span latencies from all
    spans and their children.
    """
    for span in spans:
        spans_latency_dict[span.id] = span.duration
        if span.spans:
            collect_spans_latency_recursively(span.spans, spans_latency_dict)

class CreateGitIssue(Tool):

    def __init__(self, **kwargs) -> None:
        self.name = 'CreateGitIssue'
        self.description = 'This tool helps you to create a github issue.'
        self.github_token = kwargs.get('github_token', None)
        if not self.github_token:
            raise Exception('No Github token is set, can not create issues.')

    def run(self) -> dict[str, Any]:
        github_client = GitHubClient()
        issue_number = github_client.create_issue(title=self.values['title'], body=self.values['body'], owner=self.values['owner'], repo_name=self.values['repo_name'], github_token=self.github_token)
        url = f'https://github.com/{self.values['owner']}/{self.values['repo_name']}/issues/{issue_number}'
        content = f'Issue created: {url}'
        action_type = ActionType.GITHUB_CREATE_ISSUE.value
        return {'content': content, 'action_type': action_type}

    def get_parameters(self) -> dict[str, Any]:
        self.parameters['body'] = {'type': str, 'description': 'The body of the issue.'}
        self.parameters['title'] = {'type': str, 'description': 'The title of the issue.'}
        self.parameters['owner'] = {'type': str, 'description': 'The owner of the repository.'}
        self.parameters['repo_name'] = {'type': str, 'description': 'The name of the repository.'}
        return self.parameters

def parse_github_url(url: str) -> tuple[str, str, str, str, int]:
    """
    Parse a GitHub URL and extract components.

    Args:
        url: GitHub URL in format:
            https://github.com/owner/repo/tree/ref/path/to/file#L123

    Returns:
        Tuple of (owner, repo_name, ref, file_path, line_number)
            Line number defaults to 1 if not specified in URL.

    Raises:
        ValueError: If URL format is invalid

    Example:
        >>> parse_github_url("https://github.com/traceroot-ai/traceroot-sdk/tree/main/examples/simple_example.py#L1")  # type: ignore  # noqa: E501
        ('traceroot-ai', 'traceroot-sdk', 'main',
        'examples/simple_example.py', 1)
    """
    line_number = 1
    if '#L' in url:
        url_part, fragment = url.split('#L', 1)
        try:
            line_number = int(fragment)
        except ValueError:
            line_number = 1
        url = url_part
    if '?plain=1' in url:
        url = url.replace('?plain=1', '')
    pattern = 'https://github\\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.*)'
    match = re.match(pattern, url)
    if not match:
        raise ValueError(f'Invalid GitHub URL format: {url}')
    owner, repo_name, ref, file_path = match.groups()
    return (owner, repo_name, ref, file_path, line_number)

def set_github_related(github_related_output: GithubRelatedOutput) -> GithubRelatedOutput:
    if github_related_output.is_github_issue:
        github_related_output.source_code_related = True
    if github_related_output.is_github_pr:
        github_related_output.source_code_related = True
    return github_related_output

