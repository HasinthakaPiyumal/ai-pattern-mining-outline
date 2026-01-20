# Cluster 17

class DataProcessor:
    """
    Base class for data processors.
    """
    fitted = False

    def check_fitted(self):
        """Check if the processor is fitted.

        Raises:
            SynthesizerProcessorError: If the processor is not fitted.
        """
        if not self.fitted:
            raise SynthesizerProcessorError('Processor NOT fitted.')

    def fit(self, metadata: Metadata | None=None, **kwargs: Dict[str, Any]):
        self._fit(metadata, **kwargs)
        self.fitted = True

    def _fit(self, metadata: Metadata | None=None, **kwargs: Dict[str, Any]):
        """Fit the data processor.

        Called before ``convert`` and ``reverse_convert``.

        Args:
            metadata (Metadata, optional): Metadata. Defaults to None.
        """
        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert raw data into processed data.

        Args:
            raw_data (pd.DataFrame): Raw data

        Returns:
            pd.DataFrame: Processed data
        """
        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Convert processed data into raw data.

        Args:
            processed_data (pd.DataFrame): Processed data

        Returns:
            pd.DataFrame: Raw data
        """
        return processed_data

    @staticmethod
    def remove_columns(tabular_data: pd.DataFrame, column_name_to_remove: list) -> pd.DataFrame:
        """
        Remove specified columns from the input tabular data.

        Args:
            - tabular_data (pd.DataFrame): Processed tabular data
            - column_name_to_remove (list): List of column names to be removed

        Returns:
            - result_data (pd.DataFrame): Tabular data with specified columns removed
        """
        result_data = tabular_data.copy()
        try:
            result_data = result_data.drop(columns=column_name_to_remove)
        except KeyError:
            logger.warning('Duplicate column removal occurred, which might lead to unintended consequences.')
        return result_data

    @staticmethod
    def attach_columns(tabular_data: pd.DataFrame, new_columns: pd.DataFrame) -> pd.DataFrame:
        """
        Attach additional columns to an existing DataFrame.

        Args:
            - tabular_data (pd.DataFrame): The original DataFrame.
            - new_columns (pd.DataFrame): The DataFrame containing additional columns to be attached.

        Returns:
            - result_data (pd.DataFrame): The DataFrame with new_columns attached.

        Raises:
            - ValueError: If the number of rows in tabular_data and new_columns are not the same.
        """
        if tabular_data.shape[0] != new_columns.shape[0]:
            raise ValueError('Number of rows in tabular_data and new_columns must be the same.')
        result_data = pd.concat([tabular_data, new_columns], axis=1)
        return result_data

