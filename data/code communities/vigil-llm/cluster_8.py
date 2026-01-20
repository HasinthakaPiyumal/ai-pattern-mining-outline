# Cluster 8

class Scanner:

    def __init__(self, scanners: List[BaseScanner]):
        self.name = 'dispatch:scan'
        self.scanners = scanners

    def run(self, prompt: str, scan_id: uuid.uuid4, prompt_response: str=None) -> Dict:
        response = {}
        for scanner in self.scanners:
            scan_obj = ScanModel(prompt=prompt, prompt_response=prompt_response)
            try:
                logger.info(f'Running scanner: {scanner.name}; id={scan_id}')
                updated = scanner.analyze(scan_obj, scan_id)
                response[scanner.name] = [res.dict() for res in updated.results]
                logger.success(f'Successfully ran scanner: {scanner.name} id={scan_id}')
            except Exception as err:
                logger.error(f'Failed to run scanner: {scanner.name}, Error: {str(err)} id={scan_id}')
                response[scanner.name] = {'error': str(err)}
        return response

