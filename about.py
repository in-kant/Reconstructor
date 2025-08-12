# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(Dialog)
        self.plainTextEdit.setEnabled(True)
        self.plainTextEdit.setMouseTracking(False)
        self.plainTextEdit.setInputMethodHints(QtCore.Qt.ImhNone)
        self.plainTextEdit.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plainTextEdit.setFrameShadow(QtWidgets.QFrame.Plain)
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setBackgroundVisible(False)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout.addWidget(self.plainTextEdit)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "About"))
        self.plainTextEdit.setPlainText(_translate("Dialog", "Reconstructor V. 1.31\n"
"\n"
"Reconstructor is a GUI tool for tomopy, whith walks goes through all the standard steps for parallel beam geometry tomography reconstruction, from data loading and normalization to phase retrieval and reconstruction.\n"
"Reconstructor was build for the dxchange file format (a variety of HDF5 file format).\n"
"\n"
"Written by Innokenty Kantor with contributions from many others.\n"
"\n"
"Gitlab:\n"
"Wiki (MAX IV internal): https://wiki.maxiv.lu.se/wiki/DanMAX:_Reconstructor\n"
"Youtube:"))
