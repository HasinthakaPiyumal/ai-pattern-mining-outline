# Cluster 1

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    store = Store()
    anthropic_client = AnthropicClient()
    window = MainWindow(store, anthropic_client)
    window.show()
    sys.exit(app.exec())

