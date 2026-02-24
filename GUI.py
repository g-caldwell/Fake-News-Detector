import os
import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

# constants
os.environ["QT_API"] = "PyQt6"
ICON_PATH = "GUI\\FakeNewsIcon.png"
UPLOAD_MESSAGE = "Uploading text to model..."

# create app instance
app = QApplication(sys.argv)

# global style sheet
app.setStyleSheet("""

    QWidget {
        background-color: #2C2C2C ;
    }         
                  
    QPlainTextEdit {
        background-color: #3f3f3f;
        color: #E4E4E4;
        font-family: 'Montserrat','Helvetica Neue',Helvetica,Arial,sans-serif;
        font: italic 20px;
        letter-spacing: -0.03em;
    }
                  
    QPlainTextEdit:focus {
        border: 0;
    }
                  
    QPushButton {
        font-family: 'Montserrat','Helvetica Neue',Helvetica,Arial,sans-serif;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #E4E4E4;
        background-color: #89779E;
        border-color: #89779E; 
        border-style: solid;
        border-width: 3px;
        border-radius: 6px;
        padding: 5px;
    }                    

    QPushButton:hover {
        background-color: #CCBDDF;
        border-color: #CCBDDF;
    }
""")

# window initialization
window = QWidget()
window.setWindowTitle("Fake News Detector")
window.adjustSize()
window.setWindowIcon(QIcon(ICON_PATH))

# layout initialization
layout = QGridLayout()
layout.setSpacing(20)
layout.setContentsMargins(40, 40, 40, 40)

# pie chart
pieChartCategories = ["real", "fake"]
pieChartConfidenceValues = [81, 19]
pieChartColors = ["#FFC1CC", "#A8DADC"]
fig1 = Figure(facecolor="#3f3f3f")
pieChart = fig1.add_subplot()
pieChart.pie(pieChartConfidenceValues, 
             explode=[0.0, 0.08], 
             labels=pieChartCategories, 
             colors=pieChartColors, 
             startangle=90, 
             shadow=True,
             textprops={"color": "#E4E4E4", "fontname": "sans-serif", "fontsize": 10, "fontweight": "bold"},
             wedgeprops={'width': 0.3, 'linewidth': 3, 'capstyle': "round"})
pieChartCanvas = FigureCanvasQTAgg(fig1)

# bar graph
barGraphCategories = ["PsyOp", "Flint Michigan", "Orwell", "Donald Trump", "Obama"]
barGraphValues = [1783, 2389, 2731, 2786, 1442]
barGraphColors = ["#B39CD0", "#B39CD0", "#B39CD0", "#B39CD0", "#B39CD0"]
fig2 = Figure(facecolor="#3f3f3f")
barGraph = fig2.add_subplot(frameon=False)
barGraph.bar(barGraphCategories, barGraphValues, color=barGraphColors, width=0.5)
barGraph.set_facecolor("#3f3f3f")
barGraph.set_title("frequently used words", color="#E4E4E4", fontname="sans-serif", fontweight="bold")
barGraph.tick_params(axis='x', colors='#E4E4E4')
barGraph.tick_params(axis='y', colors='#E4E4E4')
barGraphCanvas = FigureCanvasQTAgg(fig2)

# Paste text box settings and logic
pasteTextBox = QPlainTextEdit()
pasteTextBox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
pasteTextBox.setPlaceholderText("paste text here...")

# text clearing utility method
def clearText():
    pasteTextBox.clear()

# button sizing utility method
def resizeButton(button):
    button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
    button.adjustSize()

# CSV button
uploadCSVButton = QPushButton("upload csv")
resizeButton(uploadCSVButton)
uploadCSVButton.clicked.connect(QFileDialog.getOpenFileName)

# import button
importReportButton = QPushButton("import")
resizeButton(importReportButton)
importReportButton.clicked.connect(QFileDialog.getOpenFileName)

# detect button
detectButton = QPushButton("detect")
resizeButton(detectButton)
detectButton.clicked.connect(clearText)

#model dropdown
modelType = QComboBox()
modelType.addItem("Linear Regression")
modelType.addItem("Placeholder")
modelType.addItem("Placeholder")

#graph type selector
graphType = QComboBox()
graphType.addItem("Bar Chart")
graphType.addItem("Placeholder")
graphType.addItem("Placeholder")

# add widgets to layout
layout.addWidget(graphType, 0, 2, 1, 2,  alignment=Qt.AlignmentFlag.AlignCenter)
layout.addWidget(modelType, 0,0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
layout.addWidget(pieChartCanvas, 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
layout.addWidget(barGraphCanvas, 1, 2, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
layout.addWidget(pasteTextBox, 3, 0, 2, 4, alignment=Qt.AlignmentFlag.AlignVCenter)
layout.addWidget(importReportButton, 5, 0, alignment=Qt.AlignmentFlag.AlignLeft)
layout.addWidget(uploadCSVButton, 5, 0, alignment=Qt.AlignmentFlag.AlignCenter)
layout.addWidget(detectButton, 5, 3, alignment=Qt.AlignmentFlag.AlignRight)

# adds layout to window, locks window size, and displays window
window.setLayout(layout)
window.move(320, 160)
window.show()
window.setFixedSize(window.size())

# event loop
sys.exit(app.exec())