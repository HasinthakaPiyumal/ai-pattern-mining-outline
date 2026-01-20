# Cluster 10

class HTTPAgent(AgentClient):

    def __init__(self, url, proxies=None, body=None, headers=None, return_format='{response}', prompter=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.proxies = proxies or {}
        self.headers = headers or {}
        self.body = body or {}
        self.return_format = return_format
        self.prompter = Prompter.get_prompter(prompter)
        if not self.url:
            raise Exception("Please set 'url' parameter")

    def _handle_history(self, history: List[dict]) -> Dict[str, Any]:
        return self.prompter(history)

    def inference(self, history: List[dict]) -> str:
        for _ in range(3):
            try:
                body = self.body.copy()
                body.update(self._handle_history(history))
                with no_ssl_verification():
                    resp = requests.post(self.url, json=body, headers=self.headers, proxies=self.proxies, timeout=120)
                if resp.status_code != 200:
                    if check_context_limit(resp.text):
                        raise AgentContextLimitException(resp.text)
                    else:
                        raise Exception(f'Invalid status code {resp.status_code}:\n\n{resp.text}')
            except AgentClientException as e:
                raise e
            except Exception as e:
                print('Warning: ', e)
                pass
            else:
                resp = resp.json()
                return self.return_format.format(response=resp)
            time.sleep(_ + 2)
        raise Exception('Failed.')

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        opened_adapters.add(self.get_adapter(url))
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings
    requests.Session.merge_environment_settings = merge_environment_settings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings
        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

def check_context_limit(content: str):
    content = content.lower()
    and_words = [['prompt', 'context', 'tokens'], ['limit', 'exceed', 'max', 'long', 'much', 'many', 'reach', 'over', 'up', 'beyond']]
    rule = AndRule([OrRule([ContainRule(word) for word in and_words[i]]) for i in range(len(and_words))])
    return rule.check(content)

