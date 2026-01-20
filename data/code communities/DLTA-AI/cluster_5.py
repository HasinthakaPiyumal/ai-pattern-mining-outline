# Cluster 5

def PopUp():
    """
    Check for updates of DLTA-AI and display a message box with the result.

    The function checks the latest release of DLTA-AI on GitHub and compares it with the current version.
    If the latest release is newer than the current version, a message box is displayed with a button to
    download the latest version. Otherwise, a message box is displayed indicating that the user is using
    the latest version.

    Args:
        None

    Returns:
        None
    """
    from labelme import __version__
    updates = False
    tag = {}
    tag['href'] = None
    try:
        url = 'https://github.com/0ssamaak0/DLTA-AI/releases'
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, 'html.parser')
        tag = soup.find('a', class_='Link--primary')
        lastest_version = tag.text.lower().split('v')[1]
        if lastest_version != __version__:
            text = f'New version of DLTA-AI (v{lastest_version}) is available.\n You are currently using (v{__version__})\n'
            updates = True
        else:
            text = f'you are using the latest version of DLTA-AI (v{__version__})\n'
    except:
        text = f'You are using DLTA-AI (v{__version__})\n There was an error checking for updates.\n'
    msgBox = QMessageBox()
    msgBox.setWindowTitle('Check for Updates')
    msgBox.setFont(QFont('Arial', 10))
    msgBox.setText(text)
    if updates:
        msgBox.addButton(QMessageBox.StandardButton.Yes)
        msgBox.button(QMessageBox.StandardButton.Yes).setText('Get the Latest Version')
        msgBox.button(QMessageBox.StandardButton.Yes).clicked.connect(lambda: open_release(tag['href']))
    msgBox.addButton(QMessageBox.StandardButton.Close)
    msgBox.button(QMessageBox.StandardButton.Close).setText('Close')
    msgBox.exec()

def open_release(link=None):
    """
    Opens the release page for the DLTA-AI project in the default web browser.

    Parameters:
    link (str): The link to the release page. If None, the default link will be used.

    Returns:
    None
    """
    import webbrowser
    if link is None:
        link = 'https://github.com/0ssamaak0/DLTA-AI/releases'
    else:
        link = 'https://github.com/' + link
    webbrowser.open(link)

