class Calibration:
    """
    Base class for calibration methods.
    """

    def __init__(self, name: str):
        self.name = name

    def calibrate(self):
        """
        Calibrate the predictions based on the targets.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
