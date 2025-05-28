
class BaseExperiment():
    """
    Base class for experiments.
    """

    def __init__(self, config):
        self.config = config

    def run(self):
        """
        Run the experiment.
        """
        raise NotImplementedError("Subclasses should implement this method.")