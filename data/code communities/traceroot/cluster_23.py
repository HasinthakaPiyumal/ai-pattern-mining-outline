# Cluster 23

class ProviderFactory:
    """Factory for creating log and trace client instances on-the-fly.

    Creates fresh client instances for each request, supporting dynamic
    provider selection (e.g., per-request Tencent logs with default AWS traces).
    """

    @staticmethod
    def create_log_client(provider: ObservabilityProviderType | str, **kwargs: Any) -> LogClient:
        """Create a fresh log client instance for the specified provider.

        Args:
            provider: Provider type (enum or string like 'aws', 'tencent', 'jaeger')
            **kwargs: Provider-specific configuration

        Returns:
            New LogClient instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = ObservabilityProviderType(provider)
        if provider == ObservabilityProviderType.AWS:
            return AWSLogClient(aws_region=kwargs.get('region'))
        elif provider == ObservabilityProviderType.TENCENT:
            return TencentLogClient(tencent_region=kwargs.get('region'), secret_id=kwargs.get('secret_id'), secret_key=kwargs.get('secret_key'), cls_topic_id=kwargs.get('cls_topic_id'))
        elif provider == ObservabilityProviderType.JAEGER:
            return JaegerLogClient(jaeger_url=kwargs.get('url'))
        else:
            raise ValueError(f'Unknown log provider: {provider}')

    @staticmethod
    def create_trace_client(provider: ObservabilityProviderType | str, **kwargs: Any) -> TraceClient:
        """Create a fresh trace client instance for the specified provider.

        Args:
            provider: Provider type (enum or string like 'aws', 'tencent', 'jaeger')
            **kwargs: Provider-specific configuration

        Returns:
            New TraceClient instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = ObservabilityProviderType(provider)
        if provider == ObservabilityProviderType.AWS:
            return AWSTraceClient(aws_region=kwargs.get('region'))
        elif provider == ObservabilityProviderType.TENCENT:
            return TencentTraceClient(tencent_region=kwargs.get('region'), secret_id=kwargs.get('secret_id'), secret_key=kwargs.get('secret_key'), apm_instance_id=kwargs.get('apm_instance_id'))
        elif provider == ObservabilityProviderType.JAEGER:
            return JaegerTraceClient(jaeger_url=kwargs.get('url'))
        else:
            raise ValueError(f'Unknown trace provider: {provider}')

