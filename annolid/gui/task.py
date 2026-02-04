from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread


class BackgroundTask(QObject):
    """
    A class for running tasks in the background using PyQt5.
    """

    task_completed = pyqtSignal()

    def __init__(self, task_function, *args, **kwargs):
        """
        Initialize a background task.

        :param task_function: The function to run as a background task.
        :param args: Positional arguments to be passed to the task function.
        :param kwargs: Keyword arguments to be passed to the task function.
        """
        super().__init__()
        self.task_function = task_function
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def execute(self):
        """
        Execute the background task.

        This method will run the task function with the provided arguments and emit the task_completed signal when done.
        """
        try:
            self.task_function(*self.args, **self.kwargs)
            self.task_completed.emit()
        except Exception:
            # Handle exceptions as needed
            self.task_completed.emit()


class BackgroundTaskThread(QThread):
    def __init__(self, background_task):
        """
        Initialize a thread to run a background task.

        :param background_task: The BackgroundTask instance to be executed in a separate thread.
        """
        super().__init__()
        self.background_task = background_task

    def run(self):
        """
        Start executing the background task in a separate thread.
        """
        self.background_task.execute()


# Usage Example:
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    def time_consuming_task():
        # Simulate a time-consuming task
        import time

        time.sleep(2)

    app = QApplication([])

    task = BackgroundTask(time_consuming_task)
    task_thread = BackgroundTaskThread(task)

    task.task_completed.connect(app.quit)

    task_thread.start()

    app.exec_()
