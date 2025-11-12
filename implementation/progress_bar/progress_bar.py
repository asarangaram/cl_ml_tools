import time
import sys
import itertools
from contextlib import contextmanager
from visual_search_engine.progress_bar_interface import ProgressBarInterface


class ProgressBar(ProgressBarInterface):
    """
    A simple command-line progress bar with a spinner that updates on specific intervals.
    """

    def __init__(self, total_items: int, message: str, update_interval: int):
        self.total = total_items
        self.interval = update_interval
        self.message = message
        self.spinner = itertools.cycle(["|", "/", "-", "\\"])
        self.start_time = time.time()
        self._current_index = 0
        self._last_line_length = 0

    def update(self, current_item: int, additional_msg: str = "", force: bool = False):
        self._current_index = current_item
        # Only update the display if it's the right interval or the very last item
        if (
            (current_item % self.interval == 0)
            or (current_item == self.total - 1)
            or force
        ):
            next_char = next(self.spinner)
            elapsed_time = time.time() - self.start_time
            display_message = (
                f"\r{self.message}... {current_item + 1}/{self.total} completed "
                f"[{elapsed_time:.1f}s, {elapsed_time*1000/(current_item + 1):0.1f}ms/peritem] {additional_msg} {next_char}"
            )
            # Pad the line with spaces to ensure we fully overwrite previous output
            spaces_to_pad = self._last_line_length - len(display_message)
            if spaces_to_pad > 0:
                display_message += " " * spaces_to_pad

            sys.stdout.write(display_message)
            sys.stdout.flush()
            self._last_line_length = len(display_message)

    def close(self, final_message="Done!"):
        # Clear the last line and write a final message with a newline
        final_display = f"\r{self.message}... {self.total}/{self.total} completed - {final_message}    \n"
        sys.stdout.write(final_display)
        sys.stdout.flush()


# --- How the implementer uses the wrapper ---
if __name__ == "__main__":
    # 1. Prepare your data
    all_files = [f"file_{n}.txt" for n in range(532)]  # Example list of items

    # 2. Initialize the progress bar before the loop
    progress_bar = ProgressBar(
        total_items=len(all_files), update_interval=25, message="Analyzing data files"
    )

    # 3. Iterate through your items and call .update()
    for i, f in enumerate(all_files):
        # Your processing logic here (e.g., opening a file, running an operation)
        # time.sleep(0.005) # Simulate work

        # Update the progress bar status
        progress_bar.update(i)

    # 4. Close the bar after the loop finishes
    progress_bar.close(final_message="Finished successfully")
