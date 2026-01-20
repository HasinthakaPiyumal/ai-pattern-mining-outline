# Cluster 40

class HowDoIAgent(Agent):

    def __init__(self):
        super(HowDoIAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)
        self.questionIdentifier = QuestionDetection()

    def compute_simple_token_similarity(self, src_sequence, tgt_sequence):
        src_tokens = set([x.lower().strip() for x in src_sequence.split()])
        tgt_tokens = set([x.lower().strip() for x in tgt_sequence.split()])
        return len(src_tokens & tgt_tokens) / len(src_tokens)

    def get_next_action(self, state: State) -> Action:
        logger.info('================== In HowDoI Bot:get_next_action ========================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        is_question = self.questionIdentifier.is_question(state.command)
        if not is_question:
            return Action(suggested_command=state.command, confidence=0.0)
        apis: OrderedDict = self.store.get_apis()
        helpWasFound = False
        for provider in apis:
            if provider == 'manpages':
                logger.info(f"Skipping search provider 'manpages'")
                continue
            thisAPI: Provider = apis[provider]
            if not thisAPI.can_run_on_this_os():
                logger.info(f"Skipping search provider '{provider}'")
                logger.info(f'==> Excluded on platforms: {str(thisAPI.get_excludes())}')
                continue
            logger.info(f"Processing search provider '{provider}'")
            if thisAPI.has_variants():
                logger.info(f'==> Has search variants: {str(thisAPI.get_variants())}')
                variants: List = thisAPI.get_variants()
            else:
                logger.info(f'==> Has no search variants')
                variants: List = [None]
            for variant in variants:
                if variant is not None:
                    logger.info(f"==> Searching variant '{variant}'")
                    data = self.store.search(state.command, service=provider, size=1, searchType=variant)
                else:
                    data = self.store.search(state.command, service=provider, size=1)
                if data:
                    logger.info(f'==> Success!!! Found a result in the {thisAPI}')
                    searchResult = thisAPI.extract_search_result(data)
                    manpages = self.store.search(searchResult, service='manpages', size=5)
                    if manpages:
                        logger.info('==> Success!!! found relevant manpages.')
                        command = manpages['commands'][-1]
                        confidence = manpages['dists'][-1]
                        logger.info('==> Command: {} \t Confidence:{}'.format(command, confidence))
                        suggested_command = 'man {}'.format(command)
                        description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I did little bit of Internet searching for you, ').append(f'and found this in the {thisAPI}:\n').info().append(thisAPI.get_printable_output(data)).warning().append('Do you want to try: man {}'.format(command)).to_console()
                        helpWasFound = True
                        break
            if helpWasFound:
                break
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"Sorry. It looks like you have stumbled across a problem that even the Internet doesn't have answer to.\n").info().append(f'Have you tried turning it OFF and ON again. ;)').to_console()
            confidence = 0.0
        return Action(suggested_command=suggested_command, description=description, confidence=confidence)

class MsgCodeAgent(Agent):

    def __init__(self):
        super(MsgCodeAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        logger.info('==================== In zMsgCode Bot:post_execute ============================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        stderr = state.stderr.strip()
        matches = re.compile(REGEX_ZMSG).match(stderr)
        if matches is None:
            logger.info(f"No Z message ID found in '{stderr}'")
            return Action(suggested_command=state.command)
        logger.info(f"Analyzing error message '{matches[0]}'")
        msgid: str = matches[2]
        helpWasFound = False
        bpx_matches: List[str] = self.__search(matches[0], REGEX_BPX)
        if bpx_matches is not None:
            reason_code: str = bpx_matches[1]
            logger.info(f'==> Reason Code: {reason_code}')
            result: CompletedProcess = subprocess.run(['bpxmtext', reason_code], stdout=subprocess.PIPE)
            if result.returncode == 0:
                messageText = result.stdout.decode('UTF8')
                if self.__search(messageText, REGEX_BPX_BADANSWER) is None:
                    suggested_command = state.command
                    description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I asked bpxmtext about that message:\n').info().append(messageText).warning().to_console()
                    helpWasFound = True
        if not helpWasFound:
            kc_api: Provider = self.store.get_apis()['ibm_kc']
            if kc_api is not None and kc_api.can_run_on_this_os():
                data = self.store.search(msgid, service='ibm_kc', size=1)
                if data:
                    logger.info(f'==> Success!!! Found information for msgid {msgid}')
                    suggested_command = state.command
                    description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I looked up {msgid} in the IBM KnowledgeCenter for you:\n').info().append(kc_api.get_printable_output(data)).warning().to_console()
                    helpWasFound = True
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"I couldn't find any help for message code '{msgid}'\n").info().to_console()
        return Action(suggested_command=suggested_command, description=description, confidence=1.0)

    def __search(self, target: str, regex_list: List[str]) -> List[str]:
        """Check all possible regexes in a list, return the first match encountered"""
        for regex in regex_list:
            this_match = re.compile(regex).match(target)
            if this_match is not None:
                return this_match
        return None

class HelpMeAgent(Agent):

    def __init__(self):
        super(HelpMeAgent, self).__init__()
        inifile_path = os.path.join(str(Path(__file__).parent.absolute()), 'config.ini')
        self.store = Datastore(inifile_path)

    def compute_simple_token_similarity(self, src_sequence, tgt_sequence):
        src_tokens = set([x.lower().strip() for x in src_sequence.split()])
        tgt_tokens = set([x.lower().strip() for x in tgt_sequence.split()])
        return len(src_tokens & tgt_tokens) / len(src_tokens)

    def compute_confidence(self, query, forum, manpage):
        """
        Computes the confidence based on query, stack-exchange post answer and manpage

        Algorithm:
            1. Compute token-wise similarity b/w query and forum text
            2. Compute token-wise similarity b/w forum text and manpage description
            3. Return product of two similarities


        Args:
            query (str): standard error captured in state variable
            forum (str): answer text from most relevant stack exchange post w.r.t query
            manpage (str): manpage description for most relevant manpage w.r.t. forum

        Returns:
             confidence (float): confidence on the returned manpage w.r.t. query
        """
        query_forum_similarity = self.compute_simple_token_similarity(query, forum[0]['Content'])
        forum_manpage_similarity = self.compute_simple_token_similarity(forum[0]['Answer'], manpage)
        confidence = query_forum_similarity * forum_manpage_similarity
        return confidence

    def get_next_action(self, state: State) -> Action:
        return Action(suggested_command=state.command)

    def post_execute(self, state: State) -> Action:
        logger.info('==================== In Helpme Bot:post_execute ============================')
        logger.info('State:\n\tCommand: {}\n\tError Code: {}\n\tStderr: {}'.format(state.command, state.result_code, state.stderr))
        logger.info('============================================================================')
        if state.result_code == '0':
            return Action(suggested_command=state.command)
        apis: OrderedDict = self.store.get_apis()
        helpWasFound = False
        for provider in apis:
            if provider == 'manpages':
                logger.info(f"Skipping search provider 'manpages'")
                continue
            thisAPI: Provider = apis[provider]
            if not thisAPI.can_run_on_this_os():
                logger.info(f"Skipping search provider '{provider}'")
                logger.info(f'==> Excluded on platforms: {str(thisAPI.get_excludes())}')
                continue
            logger.info(f"Processing search provider '{provider}'")
            if thisAPI.has_variants():
                logger.info(f'==> Has search variants: {str(thisAPI.get_variants())}')
                variants: List = thisAPI.get_variants()
            else:
                logger.info(f'==> Has no search variants')
                variants: List = [None]
            for variant in variants:
                if variant is not None:
                    logger.info(f"==> Searching variant '{variant}'")
                    data = self.store.search(state.stderr, service=provider, size=1, searchType=variant)
                else:
                    data = self.store.search(state.stderr, service=provider, size=1)
                if data:
                    apiString = str(thisAPI)
                    if variant is not None:
                        apiString = f"{apiString} '{variant}' variant"
                    logger.info(f'==> Success!!! Found a result in the {apiString}')
                    searchResult = thisAPI.extract_search_result(data)
                    manpages = self.store.search(searchResult, service='manpages', size=5)
                    if manpages:
                        logger.info('==> Success!!! found relevant manpages.')
                        command = manpages['commands'][-1]
                        confidence = manpages['dists'][-1]
                        confidence = 1.0
                        logger.info('==> Command: {} \t Confidence:{}'.format(command, confidence))
                        suggested_command = 'man {}'.format(command)
                        description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f'I did little bit of Internet searching for you, ').append(f'and found this in the {thisAPI}:\n').info().append(thisAPI.get_printable_output(data)).warning().append('Do you want to try: man {}'.format(command)).to_console()
                        helpWasFound = True
                        break
            if helpWasFound:
                break
        if not helpWasFound:
            logger.info('Failure: Unable to be helpful')
            logger.info('============================================================================')
            suggested_command = NOOP_COMMAND
            description = Colorize().emoji(Colorize.EMOJI_ROBOT).append(f"Sorry. It looks like you have stumbled across a problem that even the Internet doesn't have answer to.\n").info().append(f'Have you tried turning it OFF and ON again. ;)').to_console()
            confidence = 0.0
        return Action(suggested_command=suggested_command, description=description, confidence=confidence)

