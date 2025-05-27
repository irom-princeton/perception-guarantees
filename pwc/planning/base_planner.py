class BasePlanner:
    """
    Base class for planners.
    """

    def __init__(self, planner_config):
        """
        Initialize the planner with the given configuration.

        Args:
            planner_config (dict): Configuration for the planner.
        """
        self.planner_config = planner_config

    def plan(self, *args, **kwargs):
        """
        Plan method to be implemented by subclasses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method.")