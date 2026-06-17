"""
This is main root f the project
just run this main file:
    py main.py

Expects this structure:
    project/
        main.py
        src/
            app.py
            neuron.py
            layer.py
            Network.py
            losses.py
            BackPropagation.py
            derivative.py
"""

import sys
import os
#importing src folder 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


libraries=["numpy","PyQt6",]
#checking if user has dependecies
def check_deps():
    missing = []
    for pkg in libraries:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(f"Fix with:  pip install {' '.join(missing)}")
        sys.exit(1)



def main():
    """
    Creates the Qt application and launches the
    Neural Network Playground window.
    
    """

    from PyQt6.QtWidgets import QApplication
    from app import MainWindow
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(application.exec())   


if __name__=="__main__":
    check_deps()
    main()    