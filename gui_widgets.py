# -*- coding: utf-8 -*-
#
# Special widget classes for the Reconstructor GUI widgets
#
# 


#from PyQt5 import QtCore, QtGui, QtWidgets
#import PyQt5
#from silx.gui.plot import Plot2D, PlotWindow
#from silx.gui.colors import Colormap
import numpy as np

class draggable_select_for_sliders:
    #  rectangular selection for silx plotting widget which can be dragged with mouse
    #  requires "parent_plot" - a silx plot object, and slider1 - slider4 - qt5 horizontalSlider objects that has the values for  
    #  top, bottom, left and right selection ranges,
    #  and lineedit1, lineedit2, lineedit3, lineedit4 - qt5 lineEdit objects that display the limits
    #  "crop" is the curve showing the rectangle added to the plot
    def __init__(self, parent_plot, slider1, slider2, slider3, slider4, lineedit1, lineedit2, lineedit3, lineedit4, name, color):
        self.parent_plot = parent_plot
        self.parent_plot.addMarker(50, 10, legend=f'{name}topleft', color=color, selectable=False, draggable=True, symbol='x', constraint=self.topleft_move)
        self.topleftmarker = self.parent_plot.getItems()[-1]
        self.parent_plot.addMarker(450, 10, legend=f'{name}bottomright', color=color, selectable=False, draggable=True, symbol='x', constraint=self.bottomright_move)
        self.bottomrightmarker = self.parent_plot.getItems()[-1]
        self.parent_plot.addMarker(250, 450, legend=f'{name}top', color=color, selectable=False, draggable=True, symbol='+', constraint=self.top_move)
        self.topmarker = self.parent_plot.getItems()[-1]
        self.parent_plot.addMarker(50, 230, legend=f'{name}left', color=color, selectable=False, draggable=True, symbol='+', constraint=self.left_move)
        self.leftmarker = self.parent_plot.getItems()[-1]
        self.parent_plot.addMarker(250, 10, legend=f'{name}bottom', color=color, selectable=False, draggable=True, symbol='+', constraint=self.bottom_move)
        self.bottommarker = self.parent_plot.getItems()[-1]
        self.parent_plot.addMarker(250, 230, legend=f'{name}right', color=color, selectable=False, draggable=True, symbol='+', constraint=self.right_move)
        self.rightmarker = self.parent_plot.getItems()[-1]
        self.slider1 = slider1
        self.slider2 = slider2
        self.slider3 = slider3
        self.slider4 = slider4
        self.lineedit1 = lineedit1
        self.lineedit2 = lineedit2
        self.lineedit3 = lineedit3
        self.lineedit4 = lineedit4
        self.parent_plot.addCurve(x=[], y=[], color=color, legend=f'{name}crop', selectable=False)
        self.curve = self.parent_plot.getItems()[-1]
        
    def topleft_move(self, x, y):
        try:
            self.slider3.setValue(int(x))
            self.slider1.setValue(int(y)-1)
            x = self.slider3.value()
            y = self.slider1.value()+1
        except:
            pass
        return x, y

    def top_move(self, x, y):
        try:
            self.slider1.setValue(int(y)-1)
            y = self.slider1.value()+1
            x = (self.slider3.value()+self.slider4.value())//2
        except:
            pass
        return x, y

    def left_move(self, x, y):
        try:
            self.slider3.setValue(int(x))
            y = (self.slider1.value()+self.slider2.value())//2
            x = self.slider3.value()
        except:
            pass
        return x, y

    def bottomright_move(self, x, y):
        try:
            self.slider4.setValue(int(x)-1)
            self.slider2.setValue(int(y))
            x = self.slider4.value()+1
            y = self.slider2.value()
        except:
            pass
        return x, y

    def bottom_move(self, x, y):
        try:
            self.slider2.setValue(int(y))
            x = (self.slider3.value()+self.slider4.value())//2
            y = self.slider2.value()
        except:
            pass
        return x, y        

    def right_move(self, x, y):
        try:
            self.slider4.setValue(int(x)-1)
            y = (self.slider1.value()+self.slider2.value())//2
            x = self.slider4.value()+1
        except:
            pass
        return x, y
    
    def crop_changed(self):
        if self.slider1.value() < self.slider2.value():
            self.slider1.setValue(self.slider2.value())
        if self.slider4.value() < self.slider3.value():
            self.slider4.setValue(self.slider3.value())
        top = self.slider1.value()
        bottom = self.slider2.value()
        left = self.slider3.value()
        right = self.slider4.value()
        self.lineedit1.setText(str(top))
        self.lineedit2.setText(str(bottom))
        self.lineedit3.setText(str(left))
        self.lineedit4.setText(str(right))
        self.curve.setData(x=[left,left,right+1,right+1,left], y=[top+1,bottom,bottom,top+1,top+1])
        self.topleftmarker.setPosition(left, top+1)
        self.bottomrightmarker.setPosition(right+1, bottom)
        self.topmarker.setPosition((left+right)//2,top+1)
        self.bottommarker.setPosition((left+right)//2,bottom)
        self.leftmarker.setPosition(left, (top+bottom)//2)
        self.rightmarker.setPosition(right+1, (top+bottom)//2)

    def hide(self):
        self.curve.setData(x=[], y=[])
        self.topleftmarker.setVisible(False)
        self.topmarker.setVisible(False)
        self.bottommarker.setVisible(False)
        self.leftmarker.setVisible(False)
        self.rightmarker.setVisible(False)
        self.bottomrightmarker.setVisible(False)

    def show(self):
        self.topleftmarker.setVisible(True)
        self.bottomrightmarker.setVisible(True)
        self.topmarker.setVisible(True)
        self.bottommarker.setVisible(True)
        self.leftmarker.setVisible(True)
        self.rightmarker.setVisible(True)
        self.crop_changed()

class smart_scale_bar:
    #  custom scale bar and ruler for a silx plotting widget
    #  requires "parent_plot" - a silx plot object
    #  "scale" is the pixel size in mm
    #  "update_scale" function recalculates label
    #  "default_scale" function recalculated the default scale bar position for current zoom
    def __init__(self, parent_plot):
        self.parent_plot = parent_plot
        self.x1 = 50
        self.x2 = 250
        self.y1 = 20
        self.y2 = 20
        self.xc = 150
        self.yc = 20
        self.scale = 0.0055
        self.parent_plot.addMarker(self.x1, self.y1, legend='ruler1', color='pink', selectable=False, draggable=True, symbol='None', constraint=self.ruler1_move)
        self.ruler1marker = self.parent_plot.getItems()[-1]
        self.parent_plot.addMarker(self.x2, self.y2, legend='ruler2', color='pink', selectable=False, draggable=True, symbol='None', constraint=self.ruler2_move)
        self.ruler2marker = self.parent_plot.getItems()[-1]
        self.parent_plot.addMarker(self.xc, self.yc, legend='ruler_center', color='pink', selectable=False, draggable=True, symbol='None', text='1.02 mm', constraint=self.rulercenter_move)
        self.rulercenter = self.parent_plot.getItems()[-1]
        self.parent_plot.addCurve(x=[100,200], y=[100,100], color='pink', legend='scale', selectable=False, linewidth=2)

    def ruler1_move(self, x, y):
        try:
            self.ruler2marker.getPosition()
        except:
            return x, y
        if x == self.x1 and y == self.y1:
            #was not moved with the mouse
            return x, y
        else:
            #was actually moved with the mouse: update the center position
            self.xc = (x + self.x2)/2
            self.yc = (y + self.y2)/2
            self.rulercenter.setPosition(self.xc, self.yc)
            self.x1 = x
            self.y1 = y
            self.update_scale()
            return x, y

    def ruler2_move(self, x, y):
        try:
            self.ruler1marker.getPosition()
        except:
            return x, y
        if x == self.x2 and y == self.y2:
            #was not moved with the mouse
            return x, y
        else:
            #was actually moved with the mouse: update the center position
            self.xc = (x + self.x1)/2
            self.yc = (y + self.y1)/2
            self.rulercenter.setPosition(self.xc, self.yc)
            self.x2 = x
            self.y2 = y
            self.update_scale()
            return x, y

    def rulercenter_move(self, x, y):
        try:
            self.ruler2marker.getPosition() #if the plot is ready
        except:
            return x, y
        if x == self.xc and y == self.yc:
            #was not moved with the mouse
            return x, y
        else:
            #was actually moved with the mouse: update the two side marker positions
            xnew1 = x + (self.x1-self.x2)/2
            ynew1 = y + (self.y1-self.y2)/2
            xnew2 = x - (self.x1-self.x2)/2
            ynew2 = y - (self.y1-self.y2)/2
            self.x1 = xnew1
            self.y1 = ynew1
            self.x2 = xnew2
            self.y2 = ynew2
            self.ruler1marker.setPosition(self.x1, self.y1)
            self.ruler2marker.setPosition(self.x2, self.y2)
            self.rulercenter.setPosition(self.xc, self.yc)
            self.update_scale()
            return x, y

    def update_scale(self):
        length = np.sqrt(np.square(self.x1 - self.x2) + np.square(self.y1 - self.y2))
        self.parent_plot.getCurve('scale').setData(x=[self.x1, self.x2], y=[self.y1, self.y2])
        if length*self.scale >= 0.99999:
            label_text = '{0:.2f} mm'.format(length*self.scale)
        if length*self.scale < 0.99999 and length*self.scale > 0.1:
            label_text = '{0:.1f} um'.format(length*self.scale*1000)
        if length*self.scale < 0.1 and length*self.scale > 0.01:
            label_text = '{0:.2f} um'.format(length*self.scale*1000)
        if length*self.scale < 0.01 and length*self.scale > 0.001:
            label_text = '{0:.3f} um'.format(length*self.scale*1000)
        if length*self.scale <= 0.001:
            label_text = '{0:.1f} nm'.format(length*self.scale*1000000)
        if self.scale == 0:
            label_text = '{0:.1f} px'.format(length)
        try:
            self.rulercenter.setText(label_text)
        except:
            return
    
    def default_scale(self):
        h_min = self.parent_plot.getGraphXLimits()[0]
        h_max = self.parent_plot.getGraphXLimits()[1]
        if h_min < 0:
            h_min = 0
        hrange = h_max - h_min
        if self.scale == 0:
            hl = hrange/2
        else:
            hl = self.scale*hrange/2
        #hl is the half horizontal range
        target_scale = (0.5*(abs(0.5*(10**np.round(np.log10(hl))) - hl) > abs(10**np.round(np.log10(hl))-hl)) + 0.5)*(10**np.round(np.log10(hl)))
        #the full or half order (e.g. 100 or 50 or 1 of 0.05) nearest to the hl
        #define the new positions
        self.x1 = h_min + ((self.parent_plot.getGraphXLimits()[1]-self.parent_plot.getGraphXLimits()[0])/10)
        #x1 starts at 10% of horizontal FOV
        self.y1 = self.parent_plot.getGraphYLimits()[0] + ((self.parent_plot.getGraphYLimits()[1]-self.parent_plot.getGraphYLimits()[0])/10)
        #height set to 10% of thr vertical FOV
        self.y2 = self.y1
        if self.scale == 0:
            self.x2 = self.x1 + target_scale
        else:
            self.x2 = self.x1 + (target_scale/self.scale)
        self.xc = (self.x1+self.x2)/2
        self.yc = (self.y1+self.y2)/2
        self.ruler1marker.setPosition(self.x1, self.y1)
        self.ruler2marker.setPosition(self.x2, self.y2)
        self.rulercenter.setPosition(self.xc, self.yc)
        self.update_scale()

    def hide(self):
        self.parent_plot.getCurve('scale').setData(x=[], y=[])
        self.rulercenter.setText(' ')

    def show(self):
        self.default_scale()
