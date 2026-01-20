# Cluster 87

class AbstractPredictor(abc.ABC):
    """
    Interface for a generic agent predictor.
    """
    requires_scenario: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> AbstractPredictor:
        """
        Define attributes needed by all predictors, take care when overriding.
        :param cls: class being constructed.
        :param args: arguments to constructor.
        :param kwargs: keyword arguments to constructor.
        """
        instance: AbstractPredictor = super().__new__(cls)
        instance._compute_predictions_runtimes = []
        return instance

    @abstractmethod
    def name(self) -> str:
        """
        :return string describing name of the predictor.
        """
        pass

    @abc.abstractmethod
    def initialize(self, initialization: PredictorInitialization) -> None:
        """
        Initialize predictor.
        :param initialization: Necessary data to initialize predictor.
        """
        pass

    @abc.abstractmethod
    def observation_type(self) -> Type[Observation]:
        """
        :return Type of observation that is expected in compute_predictions.
        """
        pass

    @abc.abstractmethod
    def compute_predicted_trajectories(self, current_input: PredictorInput) -> DetectionsTracks:
        """
        Computes the agent predictions.
        :param current_input: input to the predictor.
        :return: Detections updated with agents' predicted future trajectories.
        """
        pass

    def compute_predictions(self, current_input: PredictorInput) -> DetectionsTracks:
        """
        Computes the predicted trajectories for input agents and populates updated detections with predictions.
        :param current_input: Predictor input data. Includes observations (tracked objects) for which future
            trajectories will be predicted.
        :return: Detections updated with agents' predicted future trajectories.
        """
        start_time = time.perf_counter()
        try:
            return self.compute_predicted_trajectories(current_input)
        finally:
            self._compute_predictions_runtimes.append(time.perf_counter() - start_time)

    def generate_predictor_report(self, clear_stats: bool=True) -> PredictorReport:
        """
        Generate a report containing runtime stats from the predictor.
        By default, returns a report containing the time-series of compute_predictions runtimes.
        :param clear_stats: whether to clear stored stats after creating report.
        :return: report containing predictor runtime stats.
        """
        report = PredictorReport(compute_predictions_runtimes=self._compute_predictions_runtimes)
        if clear_stats:
            self._compute_predictions_runtimes: List[float] = []
        return report

