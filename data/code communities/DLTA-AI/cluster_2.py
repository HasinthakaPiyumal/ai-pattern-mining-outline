# Cluster 2

def check_duplicates_editLabel(id_frames_rec, old_group_id, new_group_id, only_this_frame, idChanged, currFrame):
    """
    Summary:
        Check if there are id duplicates in any frame if the id is changed.
        
    Args:
        id_frames_rec: a dictionary of id frames records
        old_group_id: the old id
        new_group_id: the new id
        only_this_frame: a flag to indicate if the id is changed only in the current frame or in all frames
        idChanged: a flag to indicate if the id is changed or not (if False, the function returns False as there is no change)
        currFrame: the current frame index
        
    Returns:
        True if there will be duplicates, False otherwise
    """
    if not idChanged:
        return False
    old_id_frame_record = copy.deepcopy(id_frames_rec['id_' + str(old_group_id)])
    try:
        new_id_frame_record = copy.deepcopy(id_frames_rec['id_' + str(new_group_id)])
    except:
        new_id_frame_record = set()
        pass
    if only_this_frame:
        Intersection = new_id_frame_record.intersection({currFrame})
        if len(Intersection) != 0:
            OKmsgBox('Warning', f'Two shapes with the same ID exists.\nApparantly, a shape with ID ({new_group_id}) already exists with another shape with ID ({old_group_id}) in the CURRENT FRAME and the edit will result in two shapes with the same ID in the same frame.\n\n The edit is NOT performed.')
            return True
    else:
        Intersection = old_id_frame_record.intersection(new_id_frame_record)
        if len(Intersection) != 0:
            reduced_Intersection = reducing_Intersection(Intersection)
            OKmsgBox('ID already exists', f'Two shapes with the same ID exists in at least one frame.\nApparantly, a shape with ID ({new_group_id}) already exists with another shape with ID ({old_group_id}).\nLike in frames ({reduced_Intersection}) and the edit will result in two shapes with the same ID ({new_group_id}).\n\n The edit is NOT performed.')
            return True
    return False

def OKmsgBox(title, text, type='info', turnResult=False):
    """
    Show a message box.

    Args:
        title (str): The title of the message box.
        text (str): The text of the message box.
        type (str, optional): The type of the message box. Can be "info", "warning", or "critical". Defaults to "info".

    Returns:
        int: The result of the message box. This will be the value of the button clicked by the user.
    """
    msgBox = QtWidgets.QMessageBox()
    if type == 'info':
        msgBox.setIcon(QtWidgets.QMessageBox.Icon.Information)
    elif type == 'warning':
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
    elif type == 'critical':
        msgBox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    msgBox.setText(text)
    msgBox.setWindowTitle(title)
    if turnResult:
        msgBox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Ok)
    else:
        msgBox.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
    msgBox.exec()
    return msgBox.result()

def PopUp(self, group_id, text):
    """
    Summary:
        Show a dialog to get a new id from the user.
        check if the id is repeated.
        
    Args:
        self: the main window object to access the canvas
        group_id: the group id
        text: Class name
        
    Returns:
        group_id: the new group id
        text: Class name (False if the user-input id is repeated)
    """
    mainTEXT = 'A Shape with that ID already exists in this frame.\n\n'
    repeated = 0
    while is_id_repeated(self, group_id):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('ID already exists')
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.resize(450, 100)
        if repeated == 0:
            label = QtWidgets.QLabel(mainTEXT + f'Please try a new ID: ')
        if repeated == 1:
            label = QtWidgets.QLabel(mainTEXT + f'OH GOD.. AGAIN? I hpoe you are not doing this on purpose..')
        if repeated == 2:
            label = QtWidgets.QLabel(mainTEXT + f'AGAIN? REALLY? LAST time for you..')
        if repeated == 3:
            text = False
            return (group_id, text)
        properID = QtWidgets.QSpinBox()
        properID.setRange(1, 1000)
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttonBox.accepted.connect(dialog.accept)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(properID)
        layout.addWidget(buttonBox)
        dialog.setLayout(layout)
        result = dialog.exec()
        if result != QtWidgets.QDialog.DialogCode.Accepted:
            text = False
            return (group_id, text)
        group_id = properID.value()
        repeated += 1
    if repeated > 1:
        OKmsgBox('Finally..!', 'OH, Finally..!')
    return (group_id, text)

