from abc import ABC, abstractmethod

class ProgressBarInterface(ABC):
    """
    Abstract base class for a progress bar interface.
    """

    @abstractmethod
    def __init__(self, total_items: int, message: str, update_interval: int):
        """
        Initializes the progress bar.

        Args:
            total_items: The total number of items to track.
            message: The message to display with the progress bar.
            update_interval: The interval at which to update the progress bar.
        """
        pass

    @abstractmethod
    def update(self, current_item: int, additional_msg: str = "", force: bool = False):
        """
        Updates the progress bar to a new state.

        Args:
            current_item: The index of the current item being processed.
            additional_msg: An optional message to display with the progress bar.
            force: If True, forces an immediate redraw of the progress bar.
        """
        pass

    @abstractmethod
    def close(self, final_message: str = ""):
        """
        Closes or finalizes the progress bar.

        Args:
            final_message: An optional final message to display.
        """
        pass
