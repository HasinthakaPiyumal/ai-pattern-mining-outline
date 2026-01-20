# Cluster 31

class ExternalJSUtilts:

    @staticmethod
    def ensure_valid_protocol(protocol):
        if protocol not in ('https', 'http'):
            raise InvalidProtocolException('Invalid protocol: %s.  Protocol must be either http or https.' % protocol)

