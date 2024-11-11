from qtpy import QtCore


class FlexibleWorker(QtCore.QObject):
    """
    A flexible worker class that runs a given function in a separate thread.
    Provides signals to indicate the start, progress, return value, and completion of the task.
    """

    start_signal = QtCore.Signal()
    finished_signal = QtCore.Signal(object)
    result_signal = QtCore.Signal(object)
    stop_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)

    def __init__(self, task_function, *args, **kwargs):
        """
        Initialize the FlexibleWorker with the function to run and its arguments.

        :param task_function: The function to be executed.
        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        """
        super().__init__()
        self._task_function = task_function
        self._args = args
        self._kwargs = kwargs
        self._is_stopped = False

        # Connect the stop signal to the stop method
        self.stop_signal.connect(self._stop)

    def run(self):
        """
        Executes the task function with the provided arguments.
        Emits signals for result and completion when done.
        """
        self._is_stopped = False
        try:
            result = self._task_function(*self._args, **self._kwargs)
            self.result_signal.emit(result)
            self.finished_signal.emit(result)
        except Exception as e:
            # Optionally handle exceptions and emit an error signal if needed
            self.finished_signal.emit(e)

    def _stop(self):
        """
        Stops the worker by setting the stop flag.
        """
        self._is_stopped = True

    def is_stopped(self):
        """
        Check if the worker has been stopped.

        :return: True if the worker is stopped, otherwise False.
        """
        return self._is_stopped

    def report_progress(self, progress):
        """
        Reports the progress of the task.

        :param progress: An integer representing the progress percentage.
        """
        self.progress_signal.emit(progress)
