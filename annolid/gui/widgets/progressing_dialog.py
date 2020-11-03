import sys
import time
from qtpy import QtWidgets as qtw
from qtpy import QtCore as qtc
from qtpy import QtGui as qtg


class Runner(qtc.QObject):
    running = qtc.Signal(int, str)

    def __init__(self, long_job_gen):
        self.long_job_gen = long_job_gen
        super(Runner, self).__init__()

    @qtc.Slot(str)
    def run(self):
        for lj, content in self.long_job_gen:
            self.running.emit(lj, content)


class ProgressingWindow(qtw.QDialog):
    running_submitted = qtc.Signal(str)

    def __init__(self, long_job_gen):
        super(ProgressingWindow, self).__init__()
        qbtn = qtw.QDialogButtonBox.Ok | qtw.QDialogButtonBox.Cancel
        self.buttonbox = qtw.QDialogButtonBox(qbtn)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        self.long_job_gen = long_job_gen
        form = qtw.QWidget()
        self.setWindowTitle("Background Running Jobs")
        layout = qtw.QFormLayout()
        form.setLayout(layout)

        self.run_btn = qtw.QPushButton('Clear Contents',
                                       clicked=self.clear_running
                                       )
        self.run_btn.animateClick()
        self.progress_bar = qtw.QProgressBar(self)

        self.results = qtw.QTableWidget(0, 2)
        self.results.setHorizontalHeaderLabels(['percentage', 'content'])
        self.results.horizontalHeader().setSectionResizeMode(qtw.QHeaderView.Stretch)
        self.results.setSizePolicy(qtw.QSizePolicy.Expanding,
                                   qtw.QSizePolicy.Expanding
                                   )

        layout.addRow(qtw.QLabel("Job running status"))
        layout.addRow(self.progress_bar)
        layout.addRow('Clear button', self.run_btn)
        layout.addRow(self.results)
        layout.addRow(self.buttonbox)

        self.runner = Runner(self.long_job_gen)
        self.runner_thread = qtc.QThread()
        self.runner.running.connect(self.add_job_to_table)
        self.runner.running.connect(self.progress_bar.setValue)
        self.running_submitted.connect(self.runner.run)

        self.runner.moveToThread(self.runner_thread)
        self.runner_thread.start()

        self.setLayout(layout)

        self.show()

    def clear_running(self):
        while self.results.rowCount() > 0:
            self.results.removeRow(0)

        self.running_submitted.emit('started')

    def add_job_to_table(self, percent, content):
        row = self.results.rowCount()
        self.results.insertRow(row)
        self.results.setItem(row, 0,
                             qtw.QTableWidgetItem(f"{str(percent)}%")
                             )
        self.results.setItem(row, 1,
                             qtw.QTableWidgetItem(str(content)))
