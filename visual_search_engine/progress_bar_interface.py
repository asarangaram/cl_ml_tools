from abc import ABC, abstractmethod

class ProgressBarInterface(ABC):
    """
    Abstract base class for a progress bar interface.
    """

    @abstractmethod
    def __init__(self, total_items: int, message: str, update_interval: int):
        pass

    @abstractmethod
    def update(self, current_item: int, additional_msg: str = "", force: bool = False):
        """Update the progress bar."""
        pass

    @abstractmethod
    def close(self, final_message: str = ""):
        """Close the progress bar."""
        pass
