# Cluster 8

def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b

def newIcon(icon):
    icons_dir = osp.join(here, '../icons')
    return QtGui.QIcon(osp.join(':/', icons_dir, '%s.png' % icon))

def newAction(parent, text, slot=None, shortcut=None, icon=None, tip=None, checkable=False, enabled=True, checked=False):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtGui.QAction(text, parent)
    if icon is not None:
        a.setIconText(text.replace(' ', '\n'))
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a

