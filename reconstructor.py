#Reconstructor V 1.31 (Aug. 2025)
from PyQt5 import QtCore, QtWidgets #QtGui,
from silx.gui.plot import Plot2D, PlotWindow
from silx.gui.colors import Colormap
from silx.gui.plot.MaskToolsWidget import MaskToolsDockWidget
from reconstructor_gui import Ui_MainWindow
from filter_dialog import Ui_filter_dialog
from dataset_select import Ui_Dialog
from about import Ui_Dialog as Ui_About
import psutil
import numpy as np
import sys
import os
import time
import copy
import h5py
import tomopy
from tomopy.recon.algorithm import allowed_recon_kwargs
ark = copy.deepcopy(allowed_recon_kwargs)
import json
import scipy
from scipy import ndimage
from gui_widgets import draggable_select_for_sliders, smart_scale_bar

ncore = psutil.cpu_count()
print('Reconstructor V1.31')
print('Found {} CPU cores available'.format(ncore))

debug = False#True #(of 2 for even more print outputs)

for a in ark:
    if 'num_gridx' in ark[a]:
        ark[a].remove('num_gridx')
    if 'num_gridy' in ark[a]:
        ark[a].remove('num_gridy')

#global variables definition
last_saved_path = ''#/data/visitors/danmax/20240936/2024042712/raw/sea_urchin' #start path for data loading
tomo=np.array([]) #all projections
flat=np.array([]) #flat filed array
dark=np.array([]) #dark field array
theta=[] #theta valuest(degrees) used for reconstruction
theta_from_file=[] #theta values (degrees) as read from file
theta_min=[] #theta first position (degrees) from file
theta_max=[] #theta last position (degrees) from file
i0_data = [] #beam monitor intensity data
cor_range_slices = [] #an array of slices (same height) with different center of rotation values
cor_range_values = [] #a list of center of rotation values
cor_range_TV = [] #a list of total variance values inside a selected region from cor_range_slices
cor_range_STD = [] #a list of standard deviations inside a selected region from cor_range_slices
phase_range_values = np.array([]) # a list of beta/delta or delta/beta values to calculate a range of retrieved slices and projections
phase_range_slices = np.array([]) #a set of slices (same height), phase retrieved with a range of alpha values
phase_range_projections = np.array([]) #a set of projections (same angle), phase retrieved with a range of alpha values
all_filters_slices = np.array([]) #a set of reconstructions of the central slice with different filters for gridrec
recon=np.array([]) #reconstructed volume

# Creating GUI and adding some GUI elements
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

#make the lists for ui elements of excluded projections
excl_reg_chk = [ui.excl_reg_chk_01, ui.excl_reg_chk_02, ui.excl_reg_chk_03, ui.excl_reg_chk_04, ui.excl_reg_chk_05, ui.excl_reg_chk_06]
excl_reg_from_spin = [ui.excl_reg_01_from_spin, ui.excl_reg_02_from_spin, ui.excl_reg_03_from_spin, ui.excl_reg_04_from_spin, ui.excl_reg_05_from_spin, ui.excl_reg_06_from_spin]
excl_reg_to_spin = [ui.excl_reg_01_to_spin, ui.excl_reg_02_to_spin, ui.excl_reg_03_to_spin, ui.excl_reg_04_to_spin, ui.excl_reg_05_to_spin, ui.excl_reg_06_to_spin]

#add main silx plot
ui.silx_plot = Plot2D(ui.load_data_tab)
ui.silx_plot.setObjectName("silx_plot")
ui.silx_plot.setKeepDataAspectRatio()
ui.silx_plot.setAxesDisplayed(False)
#remove mask button in a hard way...
for i, n in enumerate(ui.silx_plot.toolBar().actions()):
    if isinstance(n.parent(), MaskToolsDockWidget):
        ui.silx_plot.toolBar().actions()[i].setVisible(False)

#ui.silx_plot.colorbarAction.setVisible(True)
ui.silx_plot.addImage(np.zeros(shape=[500,500]), legend='image')
ui.silx_plot.addCurve(x=[], y=[], legend='circle', color='green')
ui.silx_plot.addCurve(x=[], y=[], legend='line1', color='green')
ui.silx_plot.addCurve(x=[], y=[], legend='line2', color='green')
ui.silx_plot.addCurve(x=[], y=[], legend='excl_region1', color='#FF0000', baseline = 0, fill = True)
ui.silx_plot.addCurve(x=[], y=[], legend='excl_region2', color='#FF0000', baseline = 0, fill = True)
ui.silx_plot.addCurve(x=[], y=[], legend='excl_region3', color='#FF0000', baseline = 0, fill = True)
ui.silx_plot.addCurve(x=[], y=[], legend='excl_region4', color='#FF0000', baseline = 0, fill = True)
ui.silx_plot.addCurve(x=[], y=[], legend='excl_region5', color='#FF0000', baseline = 0, fill = True)
ui.silx_plot.addCurve(x=[], y=[], legend='excl_region6', color='#FF0000', baseline = 0, fill = True)
excl_curves = [ui.silx_plot.getCurve('excl_region1'), ui.silx_plot.getCurve('excl_region2'), ui.silx_plot.getCurve('excl_region3'), ui.silx_plot.getCurve('excl_region4'),ui.silx_plot.getCurve('excl_region5'),ui.silx_plot.getCurve('excl_region6')]
for c in excl_curves:
    c.__dict__['_alpha'] = 0.5
    c.__dict__['_visible'] = False

ui.silx_layout.addWidget(ui.silx_plot)
ui.silx_plot.getColorBarWidget().getColormap().setAutoscaleMode('stddev3')
ui.silx_plot.resetZoom()

#main plot image object for simpler refferal
image = ui.silx_plot.getImage(legend='image')

#1D plot for local variance
ui.silx_cor_plot_variance = PlotWindow(ui.cor_tab, roi=False, mask=False, yInverted=False, colormap=False, aspectRatio=False, logScale=False, fit=True)
ui.silx_cor_plot_variance.setObjectName("silx_cor_plot_variance")
ui.silx_cor_plot_variance.setGraphXLabel('Center of rotation (px)')
ui.silx_cor_plot_variance.setGraphYLabel('Total local variance')
ui.plot_variance_layout.addWidget(ui.silx_cor_plot_variance)
ui.silx_cor_plot_variance.addCurve(x=[], y=[], legend='local variance')
ui.silx_cor_plot_variance.addCurve(x=[], y=[], legend='current_position', symbol='o', color='red')
local_var_curve = ui.silx_cor_plot_variance.getCurve('local variance')
local_var_cursor = ui.silx_cor_plot_variance.getCurve('current_position')

#1D silx plot for standad deviations
ui.silx_cor_plot_std = PlotWindow(ui.cor_tab, roi=False, mask=False, yInverted=False, colormap=False, aspectRatio=False, logScale=False, fit=True)
ui.silx_cor_plot_std.setObjectName("silx_cor_plot_std")
ui.silx_cor_plot_std.setGraphXLabel('Center of rotation (px)')
ui.silx_cor_plot_std.setGraphYLabel('Local standard deviation')
ui.plot_std_layout.addWidget(ui.silx_cor_plot_std)
ui.silx_cor_plot_std.addCurve(x=[], y=[], legend='std')
ui.silx_cor_plot_std.addCurve(x=[], y=[], legend='std_current_position', symbol='o', color='red')
local_std_curve = ui.silx_cor_plot_std.getCurve('std')
local_std_cursor = ui.silx_cor_plot_std.getCurve('std_current_position')

#add progress bar to the statusbar
ui.progress_bar = QtWidgets.QProgressBar()
ui.statusbar.addPermanentWidget(ui.progress_bar)
ui.progress_bar.hide()

#add smart ROI select objects
crop_selection = draggable_select_for_sliders(ui.silx_plot, ui.crop_top_slider, ui.crop_bottom_slider, ui.crop_left_slider, ui.crop_right_slider, ui.crop_top_index, ui.crop_bottom_index, ui.crop_left_index, ui.crop_right_index, 'crop_', 'red')
postp_crop_selection = draggable_select_for_sliders(ui.silx_plot, ui.postp_top_slider, ui.postp_bottom_slider, ui.postp_left_slider, ui.postp_right_slider, ui.postp_top_index, ui.postp_bottom_index, ui.postp_left_index, ui.postp_right_index, 'postp_', 'yellow')
postp_crop_selection.hide()
scalebar = smart_scale_bar(ui.silx_plot)
scalebar.scale = 0
scalebar.default_scale()
scalebar.hide()

#fill the list of existing reconstruction algorithms
ui.recon_algorithm_select.clear()
for a in ark:
    ui.recon_algorithm_select.addItem(a)
if 'gridrec' in ark: #we set gridrec as default
    ui.recon_algorithm_select.setCurrentIndex(list(ark).index('gridrec'))

#fill in parameters entries placeholders for recon algorithms
ui.recon_param_labels = []
ui.recon_param_inputs = []
for i in range(0, 7):
    ui.recon_param_labels.append(QtWidgets.QLabel(ui.recon_groupBox_01))
    ui.recon_param_labels[i].setObjectName(f"recon_param_label_{i}")
    ui.gridLayout_7.addWidget(ui.recon_param_labels[i], i+1, 0, 1, 1)
    ui.recon_param_inputs.append(QtWidgets.QLineEdit(ui.recon_groupBox_01))
    ui.recon_param_inputs[i].setMaximumSize(QtCore.QSize(50, 16777215))
    ui.recon_param_inputs[i].setObjectName(f"recon_param_input_{i}")
    ui.gridLayout_7.addWidget(ui.recon_param_inputs[i], i+1, 1, 1, 1)

#disable all tabs before the data is loaded
ui.crop_tab.setEnabled(False)
ui.sino_tab.setEnabled(False)
ui.cor_tab.setEnabled(False)
ui.phase_tab.setEnabled(False)
ui.recon_tab.setEnabled(False)
ui.param_tab.setEnabled(False)
ui.postp_tab.setEnabled(False)


def fresh_tori():
    """generates a default tori dictionary

    Returns
    -------
    tori dictionaty with the default parameters
    """
    tori={
    'file_definitions': {
        'data_file': '',
        'data_path': 'exchange/data',
        'dark_file': '',
        'dark_path': 'exchange/data_dark',
        'flat_file': '',
        'flat_path': 'exchange/data_white',
        'slice_proj': [0,-1,1],
        'slice_ver': [0,-1,1],
        'slice_hor': [0,-1,1],
        'theta_source': 'exchange', #or 'calc'
        'theta_start': 0,
        'theta_end': 180
        },
    'correction_arguments': {
        'normalize_dark': False,
        'normalize_flat': False,
        'normalize_i0': False,
        'normalize_to_unity': False,
        'flatten_images': False,
        'flatten_images_kwargs': {'flatten_size': None},
        'crop': False,
        'crop_kwargs': {'crop_ranges': [0, -1, 0, -1]},
        'outlier_removal': False,
        'outlier_removal_kwargs': {'dif': 0.4,
                           'size': 3,
                           'axis': 0},
        'stripe_removal': False,
        'stripe_function': 'all_stripe',
        'stripe_removal_kwargs': {'snr': 3,
                          'la_size': 61,
                          'sm_size': 21,
                          'dim': 1},
        'phase_retrieval':  False,
        'phase_retrieval_kwargs': {'pixel_size': 0.00055,
                         'dist': 28,
                         'energy': 20,
                         'alpha': 0.002,
                         'pad': True},
        'i0_correction': False,
        'minus_log_before_phase': False,
        'minus_log_after_phase': False},
    'reconstruction_arguments': {
        'recon_kwargs': {
            'center': None,
            'algorithm': 'gridrec',
            'filter_name': 'parzen',
            'ncore': 24,
            'nchunk': 1},
        'pad': True,
        'pad_kwargs': {
            'axis': 2,
            'npad': 'auto',
            'mode': 'edge'},
        'sino_360_to_180': False,
        'sino_360_to_180_old_center': None,
        'excluded_proj': []
            },
    'post_recon_process_arguments': {
        'circ_mask': False,
        'circ_mask_kwargs': {
            'ratio': 0.95,
            'val': 0
            },
        'ring_removal': False,
        'ring_removal_kwargs': {
            'thresh': None,
            'theta_min': None,
            'rwidth': None
            },
        'convert': False,
        'convert_kwargs':
            {'min': None,
            'max': None,
            'dtype': 'float16',
            'mode': 'std',  # or minmax or manual
            },
        'downscale': False,
        'downscale_kwargs': {
            'down_scale_factors': [2,4,8,16],
            'chunks': (100, 100, 100),}
                                    }
        }
    return tori

def fix_skipped_ranges():
    '''Internal macros to check multiple skip ranges for consistency
    '''
    global tori
    ep = tori['reconstruction_arguments']['excluded_proj']
    if len(ep) == 0:
        return
    if len(ep) == 1:
        ep[0].sort()
        return
    ep.sort()
    for r in ep:
        r.sort()
    #fixed_list = []
    for i in range(1, len(ep)):
        if ep[i][0] < ep[i-1][1]:
            ep[i][0] = ep[i-1][0]
            ep[i][1] = np.max([ep[i-1][1], ep[i][1]])
            ep[i-1][0] = -1
    for n, r in enumerate(ep):
        if r[0] == -1:
            del(ep[n])
    tori['reconstruction_arguments']['excluded_proj'] = ep

def get_datasets_in_h5(filename):
    """finds all 2D and 3D datasets inside a h5 file, including external links

    Parameters
    ----------
    filename : full path to a valid h5 file

    Returns
    -------
    keys2d : a list of all h5 path (keys) with 2d datasets
    keys3d : a list of all h5 path (keys) with 2d datasets
    """
    try:
        hf = h5py.File(filename, 'r')
    except:
        print('this is probably not a valid h5 file')
        return None, None
    #read what datasets are in the file
    #a workaround is required, as h5py visit and visititems does not go through the external links :(
    keys = list(hf.keys())
    oldlen = 0
    while len(keys) > oldlen:#keep iterating deeper into keys structure untill all items are found
        oldlen = len(keys)
        for item in keys:
            try:
                newitems = list(hf[item].keys())
                for newitem in newitems:
                    if not item+'/'+newitem in keys:
                        keys.append(item+'/'+newitem)
            except:
                pass
    #now 'keys' is the list of all entries in the file. Lets leave only those with 2D (single image) or 3D (images stack) datasets
    keys2d = []
    keys3d = []
    for i in keys:
        ## ERROR WORKAROUND FREDERIK 21/04-2024 FOR CORRUPTED FILES / LINKS
        try:
            if isinstance(hf[i], h5py.Dataset):
                if len(hf[i].shape) ==2:
                    keys2d.append(i)
                if len(hf[i].shape) ==3:
                    keys3d.append(i)
        except KeyError:
            print('Unable to open ', i)
    return keys2d, keys3d


def get_description_for_param(function, param):
    """returns the description of a "param" parameter of the "function"
    from the tomopy function help doc

    Parameters
    ----------
    function : a (tomopy) function, e.g. tomopy.prep.stripe.remove_stripe_ti
    param : name of the parameter for this function, e.g. 'alpha'

    Returns
    -------
    a text string with the desctiption, e.g. 'Damping factor.'
    """
    try:
        #fullhelp = function.__doc__.split('\n')
        fullhelp = function.__doc__.splitlines()
    except:
        return ""
    fullhelp = [s.strip() for s in fullhelp]
    for i,h in enumerate(fullhelp):
        if param in h:
            try:
                return fullhelp[i+1] + fullhelp[i].split(':')[-1]
            except:
                return ""
    return ""


def grad(img):
    """compute simple gradients in 2D images

    Parameters
    ----------
    img : 2D array_like

    Returns
    -------
    img_grad : 2D gradient image
    img_grad_x : 2D gradient in x direction (axis = 1)
    img_grad_y : 2D gradient in y direction (axis = 0)
    """
    img_grad_x = abs(img[1:,:]-img[0:-1,:])
    img_grad_y = abs(img[:,1:]-img[:,0:-1])
    img_grad = img_grad_x[:,1:] + img_grad_y[1:,:]
    return img_grad, img_grad_x, img_grad_y


def select_data_file(autoload):
    """trying to load everything from a DxChange file or at least a dataset from an h5 file (or also tiff in the future?)
    """
    if autoload == True:
        print('loading from tori')
    else:
        autoload = False
    global tori, last_saved_path, tomo, flat, dark, theta, theta_from_file, i0_data
    if not autoload:
        filename = QtWidgets.QFileDialog.getOpenFileName(MainWindow, 'Open DxChange file', last_saved_path, "h5 files (*.h5)")
        if filename[0]=='':
            print('no file specified')
            return
        loadfilename = filename[0] #REMOVE
        tori['file_definitions']['data_file'] = filename[0]
    try:
        f = h5py.File(tori['file_definitions']['data_file'], 'r')
    except:
        print('selected datafile is probably not a valid h5 file')
        return
    print('selected file: '+tori['file_definitions']['data_file'])
    if not autoload:
        keys2d, keys3d = get_datasets_in_h5(tori['file_definitions']['data_file'])
        dui.dataset_combo.clear()
        dui.dataset_combo.addItems(keys3d)
        #do pre-filtering
        if ui.show_pre_filter_chk.isChecked():
            keys2d, keys3d = get_datasets_in_h5(tori['file_definitions']['data_file'])
            recalc()
            if FilterDialog.exec():
                #define filtered slicing for partial data load
                tori['file_definitions']['slice_proj'] = [dui.proj_range_0_spin.value(), dui.proj_range_1_spin.value(), dui.proj_downsample_spin.value()]
                tori['file_definitions']['slice_ver'] = [dui.ver_range_0_spin.value(), dui.ver_range_1_spin.value(), dui.ver_downsample_spin.value()]
                tori['file_definitions']['slice_hor'] = [dui.hor_range_0_spin.value(), dui.hor_range_1_spin.value(), dui.hor_downsample_spin.value()]
                tori['file_definitions']['data_path'] = dui.dataset_combo.currentText()
            else:
                #canlel was clicked
                return
        else: #no filter dialog opened
            tori['file_definitions']['slice_proj'] = [0, f['exchange/data'].shape[0], 1]
            tori['file_definitions']['slice_ver'] = [0, f['exchange/data'].shape[1], 1]
            tori['file_definitions']['slice_hor'] = [0, f['exchange/data'].shape[2], 1]
    ui.data_filename_display.setText(tori['file_definitions']['data_file'])
    last_saved_path = os.path.dirname(tori['file_definitions']['data_file'])

    myslices = [slice(tori['file_definitions']['slice_proj'][0], tori['file_definitions']['slice_proj'][1], tori['file_definitions']['slice_proj'][2]),
                slice(tori['file_definitions']['slice_ver'][0], tori['file_definitions']['slice_ver'][1], tori['file_definitions']['slice_ver'][2]),
                slice(tori['file_definitions']['slice_hor'][0], tori['file_definitions']['slice_hor'][1], tori['file_definitions']['slice_hor'][2])]

    #'fix' slicing definition in tori: if max range is full sise, set to -1
    #this will avoid confusion when loading new dataset of different dimensions
    f = h5py.File(tori['file_definitions']['data_file'], 'r')
    dshape = f[tori['file_definitions']['data_path']].shape
    if tori['file_definitions']['slice_proj'][1] == dshape[0]:
        tori['file_definitions']['slice_proj'][1] = -1
    if tori['file_definitions']['slice_ver'][1] == dshape[1]:
        tori['file_definitions']['slice_ver'][1] = -1
    if tori['file_definitions']['slice_hor'][1] == dshape[2]:
        tori['file_definitions']['slice_hor'][1] = -1

    tomo = load_3d_data(tori['file_definitions']['data_file'], tori['file_definitions']['data_path'], myslices)
    ui.data_path_edit.setText(tori['file_definitions']['data_path'])

    try:
        ffile = tori['file_definitions']['flat_file']
        if ffile == '':
            ffile = tori['file_definitions']['data_file']
        flat = load_3d_data(ffile, tori['file_definitions']['flat_path'], [slice(0,None,1),myslices[1],myslices[2]])
        tori['file_definitions']['flat_file']=ffile
    except:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("No flat field data found in this file!")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.show()
        retval = msg.exec_()
    try:
        dfile = tori['file_definitions']['dark_file']
        if dfile == '':
            dfile = tori['file_definitions']['data_file']
        dark = load_3d_data(dfile, tori['file_definitions']['dark_path'], [slice(0,None,1),myslices[1],myslices[2]])
        tori['file_definitions']['dark_file'] = dfile
    except:
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("No dxchange dark field data found in this file!")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.show()
        retval = msg.exec_()
    try:
        theta = f['exchange/theta'][myslices[0]]
        if not autoload:
            tori['file_definitions']['theta_source'] = 'exchange'
        for i in range (0, theta.shape[0]):
            if theta[i]<0:
                theta[i] = theta[i]+360
        theta_from_file = theta
    except:
        theta = None
        print('No theta values found in the file')
        ui.cor_use_theta_from_file_rbtn.setEnabled(False)
        ui.cor_calculate_theta_rbtn.setChecked(True)
        tori['file_definitions']['theta_source'] = 'calc'
        theta_from_file = []
    try:
        theta_min = f['process/acquisition/setup/rotation_start_angle'][()]
        tori['file_definitions']['theta_start'] = theta_min
        ui.cor_start_theta_input.setText(str(theta_min))
    except:
        print('No theta start position found in the file')
        ui.cor_start_theta_input.setText('')
    try:
        theta_max = f['process/acquisition/setup/rotation_end_angle'][()]
        tori['file_definitions']['theta_end'] = theta_max
        ui.cor_end_theta_input.setText(str(theta_max))
    except:
        print('No theta end position found in file')
        ui.cor_end_theta_input.setText('')
    try:
        energy = np.round(f['instrument/monochromator/energy'][()], decimals=2)
        if not autoload:
            tori['correction_arguments']['phase_retrieval_kwargs']['energy'] = energy
        ui.phase_energy_inp.setText(str(tori['correction_arguments']['phase_retrieval_kwargs']['energy']))
    except:
        print('No energy values found in file')
        ui.phase_energy_inp.setText('')
    try:
        if not autoload:
            pixel_size = f.get('instrument/detector/actual_pixel_size_x')
            if 'units' in pixel_size.attrs.keys():
                if pixel_size.attrs['units'] == 'um' or pixel_size.attrs['unit'] == 'micron' or pixel_size.attrs['unit'] == 'microns':
                    tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = pixel_size[()]*1e-4
                if pixel_size.attrs['units'] == 'mm':
                    tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = pixel_size[()]*1e-1
                if pixel_size.attrs['units'] == 'cm':
                    tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = pixel_size[()]
                if pixel_size.attrs['units'] == 'm':
                    tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = pixel_size[()]*1e2
            else: #no unit attribute, assume um:
                tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = pixel_size[()]*1e-4
        tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = np.round(tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'], decimals = 7) #precision down to 1 nm
        scalebar.scale = tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size']*10 #scale bar is in mm units
    except:
        print('No pixel size value found in file')
        ui.phase_pixel_size.setText('')
    if not autoload:
        try:
            sdd = f.get('instrument/detector/propagation_distance')
            if 'units' in sdd.attrs.keys():
                if sdd.attrs['units'] == 'mm':
                    tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = sdd[()]/10
                if sdd.attrs['units'] == 'cm':
                    tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = sdd[()]
                if sdd.attrs['units'] == 'm':
                    tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = sdd[()]*100
            else: #no unit attribute, assume mm:
                tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = sdd[()]/10
            tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = np.round(tori['correction_arguments']['phase_retrieval_kwargs']['dist'], decimals = 2) #in cm
        except Exception as error:
            print(error)
            print('No propagation distance value found in file')
    try:
        i0_data = f.get('/instrument/beam_monitor/data')
        i0_data = i0_data[myslices[0]]
        i0_data = i0_data/i0_data[0]
        print(f'i0 data loaded, with {len(i0_data)} points')
        ui.i0_normalize_btn.setEnabled(True)
    except:
        print('No I0 monitor data found in file')
        ui.i0_normalize_btn.setEnabled(False)
    #now do the interface elements updates
    reset_interface()
    unblock_gui()
    image_source_changed()
    plot_slider_update()
    ui.silx_plot.resetZoom()


def totvar(img):
    """compute total variation of 2D image

    Parameters
    ----------
    img : 2D array_like

    Returns
    -------
    TV : the total variation based on image gradients
    """
    img_grad,_,_ = grad(img)
    TV = np.sum( img_grad )
    return TV


def try_all_filters():
    '''reconstructs a central slice with each of the possible filters
    generates an all_filters_slices dataset 
    '''
    global tori, tomo, theta, all_filters_slices
    fltlist = [ui.recon_filter_select.itemText(i) for i in range(ui.recon_filter_select.count())]
    
    all_filters_slices = np.empty(shape = (0, tomo.shape[2], tomo.shape[2]), dtype = 'float32')
    
    ui.plot_selector.clear()
    ui.plot_selector.addItem('XY')
    ui.plot_selector.addItem('XZ')
    ui.plot_selector.addItem('YZ')
    ui.plot_selector.addItem('filters')
    ui.plot_selector.setCurrentIndex(3)
       
    if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
        recon_theta = np.radians(theta)
    else:
        recon_theta = theta
    
    single_slice = np.empty(shape = (tomo.shape[0], 1, tomo.shape[2]), dtype='float32')
    single_slice[:,0,:] = tomo[:,tomo.shape[1]//2,:]
 
    if ui.recon_pad_chk.isChecked():
        padsize = tomo.shape[2]//4
        single_slice = tomopy.misc.morph.pad(single_slice, axis=2, npad=padsize, mode='edge')
        COR = tori['reconstruction_arguments']['recon_kwargs']['center']+padsize
    else:
        COR = tori['reconstruction_arguments']['recon_kwargs']['center']

    mask = theta_mask()
    recargs = {'tomo': single_slice[mask],
                'theta': recon_theta[mask],
                'center': COR,
                'sinogram_order': False,
                'algorithm': 'gridrec',
                'ncore': ncore}
 
    for f in fltlist:
        recargs['filter_name'] = f
        block_gui(f'reconstructing central slice with the {f} filter', 100*all_filters_slices.shape[0]/len(fltlist))
        partial_recon = tomopy.recon(**recargs)
        if ui.recon_pad_chk.isChecked(): #unpad results
            partial_recon = partial_recon[:,padsize:-padsize,padsize:-padsize]
        all_filters_slices = np.append(all_filters_slices, partial_recon, axis = 0)
    image_source_changed()
    ui.silx_plot.resetZoom()
    unblock_gui()


def load_3d_data(filename, h5path, myslices):
    '''Loads a 3D dataset in chunks
    filename and h5path are hdf5 file and dataset path, myslices is an array of three slices for three dimensions, can contain Nones
    '''
    if debug:
        print(f'filename: {filename}')
        print(f'path: {h5path}')
        print(f'slices: {myslices}')
    try:
        f = h5py.File(filename, 'r')
    except:
        print(f'{filename} is not a valid h5 file?')
        return
    try:
        h5dataset = f[h5path]
    except:
        print('dataset is not in the file or corrupted')
        return
    starttime = time.time()
    try: #loading full dataset in 20 chunks
        chunksize = len(np.arange(myslices[0].start, min(x for x in [myslices[0].stop, h5dataset.shape[0]] if x is not None))) #chunk size in binned data, float!
        if chunksize <= 100: #do in a single go
            block_gui('Loading data in a single chunk...', 0)
            dataset = h5dataset[myslices[0],myslices[1],myslices[2]]
            unblock_gui()
            return dataset

        loaded_shape = [len(np.arange(myslices[0].start, myslices[0].stop, myslices[0].step)),
                        len(np.arange(myslices[1].start, myslices[1].stop, myslices[1].step)),
                        len(np.arange(myslices[2].start, myslices[2].stop, myslices[2].step))]
        dim0_mask = np.zeros(shape = h5dataset.shape[0], dtype=bool)
        dim0_mask[myslices[0]] = True #dim_0 is a boolean mask for projections to be loaded, dimension is matching the full dataset dimension
        dataset = np.zeros(shape = loaded_shape, dtype = f[h5path].dtype)
        block_gui(f'Wait for data to load (raw: {h5dataset.nbytes*1e-9:.2f} GB, filtered: {dataset.nbytes*1e-9:.2f} GB)...', 0)
        for i in range (0,1+h5dataset.shape[0]//100):
            ch_start = i*100
            ch_end = (i+1)*100
            dim0_chunk_mask = dim0_mask[ch_start:ch_end]
            if np.sum(dim0_chunk_mask): #if any projections are to be loaded in the current range of raw projections
                if debug:
                    print(f'loading into filtered dataset from {np.sum(dim0_mask[0:ch_start])} to {np.sum(dim0_mask[0:ch_start]) + np.sum(dim0_chunk_mask)}')
                dataset[np.sum(dim0_mask[0:ch_start]):np.sum(dim0_mask[0:ch_start]) + np.sum(dim0_chunk_mask)] = h5dataset[
                int(i*100):int((i+1)*100),
                myslices[1].start:myslices[1].stop:myslices[1].step,
                myslices[2].start:myslices[2].stop:myslices[2].step][dim0_chunk_mask]
            block_gui(f'Wait for data to load (raw: {h5dataset.nbytes*1e-9:.2f} GB, filtered: {dataset.nbytes*1e-9:.2f} GB)...', (i+1)*100*100/(h5dataset.shape[0]))
        loadtime = np.round(time.time() - starttime, decimals = 2)
        print(f'Total loading time was {loadtime} seconds at {dataset.nbytes*1e-6/loadtime:.3f} MB/s')
    except Exception as error:
        print(error)
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(str(error))
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.show()
        retval = msg.exec_()
        unblock_gui()
        return
    return dataset

def theta_mask():
    '''Calculates boolean mask for data with excluded theta regions
    '''
    global tomo, tori
    mask = np.ones(shape = tomo.shape[0], dtype=bool)
    for er in tori['reconstruction_arguments']['excluded_proj']:
        mask[er[0]:er[1]] = False
    return mask

def reset_interface():
    '''Resets GUI interface
    '''
    global tomo, cor_range_slices, cor_range_values, cor_range_TV, local_var_curve, local_var_cursor, local_std_curve, local_std_cursor
    ui.crop_tab.setEnabled(True)
    ui.normalize_btn.setEnabled(True)
    ui.apply_minus_log_btn.setEnabled(True)
    ui.apply_minus_log_btn2.setEnabled(True)
    ui.plot_tab_displaylog_chk.setEnabled(True)
    ui.unity_normalize_btn.setEnabled(True)
    ui.flatten_projections_btn.setEnabled(True)
    ui.sino_tab.setEnabled(True)
    ui.cor_tab_value_slider.setValue(0)
    ui.cor_tab_value_slider.setMaximum(9)
    cor_range_values = np.linspace(0, 9, 10)
    cor_range_TV = np.zeros(shape=10)
    cor_range_STD = np.zeros(shape=10)
    cor_range_slices = np.zeros(shape=[10, 100, 100])
    ui.cor_tab_log_chk.setEnabled(True)
    local_var_curve.setData(x=[], y=[])
    local_var_cursor.setData(x=[], y=[])
    local_std_curve.setData(x=[], y=[])
    local_std_cursor.setData(x=[], y=[])
    ui.phase_slider.setMaximum(tomo.shape[0]-1)
    ui.phase_slider.setValue(0)
    ui.phase_index.setText('0')
    ui.recon_first_slice_slider.setMaximum(tomo.shape[1]-1)
    ui.recon_first_slice_slider.setValue(0)
    ui.recon_first_slice_index.setText('0')
    ui.recon_last_slice_slider.setMaximum(tomo.shape[1]-1)
    ui.recon_last_slice_slider.setValue(tomo.shape[1]-1)
    ui.recon_last_slice_index.setText(str(tomo.shape[1]-1))
    ui.phase_retrieve_btn.setEnabled(True)
    ui.plot_next_btn.setEnabled(True)
    ui.phase_log_chk.setChecked(True)
    ui.phase_log_chk.setEnabled(True)
    ui.cor_tab.setEnabled(True)
    ui.phase_tab.setEnabled(True)
    ui.recon_tab.setEnabled(True)
    ui.param_tab.setEnabled(True)
    ui.convert_360_180_btn.setEnabled(True)
    ui.postp_tab.setEnabled(True)
    
def block_gui(message, progress):
    '''Disables GUI with a given messaga and progress bar value
    '''
    ui.statusbar.showMessage(message)
    ui.centralwidget.setEnabled(False)
    ui.progress_bar.show()
    ui.progress_bar.setValue(int(progress))
    QtCore.QCoreApplication.processEvents()
    
def unblock_gui():
    '''Unblocks GUI
    '''
    global debug
    ui.progress_bar.hide()
    ui.statusbar.showMessage('')
    ui.centralwidget.setEnabled(True)
    if debug:
        print('Current data in memory size:')
        print(f'tomo  shape: {tomo.shape}, dtype={tomo.dtype}, size = {tomo.nbytes*1e-9:.2f} GB')
        print(f'recon shape: {recon.shape}, dtype={recon.dtype}, size = {recon.size*recon.itemsize*1e-9:.2f} GB')
    if debug == 2:
        print(f'dark  shape: {dark.shape}, dtype={dark.dtype}, size = {dark.size*tomo.itemsize*1e-9:.2f} GB')
        print(f'flat  shape: {flat.shape}, dtype={flat.dtype}, size = {flat.size*tomo.itemsize*1e-9:.2f} GB')
        print(f"pixel size (cm) = {tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size']}")

def load_custom_dark():
    '''Loads dark data from a custom h5 file
    '''
    global tori, last_saved_path, dark
    filename = QtWidgets.QFileDialog.getOpenFileName(MainWindow, 'Open file with dark frames', last_saved_path, "h5 files (*.h5)")
    if filename[0]=='':
        print('no file specified')
        return
    loadfilename = filename[0]
    try:
        keys2d, keys3d = get_datasets_in_h5(loadfilename)
    except:
        print('not a valid h5 file?')
        return
    #read what datasets are in the file
    if len(keys3d) == 0:
        print('no suitable datasets are found in this file')
        return
    if len(keys3d) == 1:
        ui.custom_dark_path_edit.setText(keys3d[0])
        ui.custom_dark_filename_display.setText(loadfilename)
    if len(keys3d) > 1:
        #first delete existing fields, if any:
        try:
            for i in dsui.radioButtons:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dim0labels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dim1labels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dim2labels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dtypelabels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.sizelabels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            print('cleared')
        except Exception as error:
            pass#print(error) #no pre-existing idatasets
        dsui.radioButtons = []
        dsui.dim0labels = []
        dsui.dim1labels = []
        dsui.dim2labels = []
        dsui.dtypelabels = []
        dsui.sizelabels = []
        for i in range (0, len(keys3d)):
            h5dataset = h5py.File(filename[0], 'r')[keys3d[i]]
            dsui.radioButtons.append(QtWidgets.QRadioButton(DatasetDialog))
            dsui.radioButtons[i].setObjectName(f"radioButton{i}")
            dsui.radioButtons[i].setText(keys3d[i])
            dsui.gridLayout.addWidget(dsui.radioButtons[i], i+1, 0, 1, 1)
            dsui.dim0labels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dim0labels[i].setObjectName(f"dim0labels{i}")
            dsui.gridLayout.addWidget(dsui.dim0labels[i], i+1, 1, 1, 1)
            dsui.dim0labels[i].setText(str(h5dataset.shape[0]))
            dsui.dim1labels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dim1labels[i].setObjectName(f"dim1labels{i}")
            dsui.gridLayout.addWidget(dsui.dim1labels[i], i+1, 2, 1, 1)
            dsui.dim1labels[i].setText(str(h5dataset.shape[1]))
            dsui.dim2labels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dim2labels[i].setObjectName(f"dim2labels{i}")
            dsui.gridLayout.addWidget(dsui.dim2labels[i], i+1, 3, 1, 1)
            dsui.dim2labels[i].setText(str(h5dataset.shape[2]))
            dsui.dtypelabels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dtypelabels[i].setObjectName(f"dtypelabels{i}")
            dsui.gridLayout.addWidget(dsui.dtypelabels[i], i+1, 4, 1, 1)
            dsui.dtypelabels[i].setText(str(h5dataset.dtype))
            dsui.sizelabels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.sizelabels[i].setObjectName(f"sizelabels{i}")
            dsui.gridLayout.addWidget(dsui.sizelabels[i], i+1, 5, 1, 1)
            dsui.sizelabels[i].setText(str(np.round(h5dataset.nbytes*1e-9, decimals = 2)))
        dsui.radioButtons[0].setChecked(True)
        if DatasetDialog.exec():
            #dataset was selected and OK clicked, let's find out which dataset
            for i in dsui.radioButtons:
                if i.isChecked():
                    ui.custom_dark_path_edit.setText(i.text())
            ui.custom_dark_filename_display.setText(loadfilename)
    #now actually load the data
    myslices = [slice(0,None,1),
                slice(tori['file_definitions']['slice_ver'][0], tori['file_definitions']['slice_ver'][1], tori['file_definitions']['slice_ver'][2]),
                slice(tori['file_definitions']['slice_hor'][0], tori['file_definitions']['slice_hor'][1], tori['file_definitions']['slice_hor'][2])]
    try:
        block_gui('Wait for dark data to load...', 0)
        dark = load_3d_data(loadfilename, ui.custom_dark_path_edit.text(), myslices)
        tori['file_definitions']['dark_file'] = loadfilename
        tori['file_definitions']['dark_path'] = ui.custom_dark_path_edit.text()
    except:
        ui.statusbar.showMessage('something went wrong...')
        unblock_gui()
        return
    unblock_gui()
    image_source_changed()


def load_custom_flat():
    '''Loads flat field data from a custom file
    '''
    global tori, last_saved_path, flat
    filename = QtWidgets.QFileDialog.getOpenFileName(MainWindow, 'Open file with flat frames', last_saved_path, "h5 files (*.h5)")
    if filename[0]=='':
        print('no file specified')
        return
    loadfilename = filename[0]
    try:
        keys2d, keys3d = get_datasets_in_h5(loadfilename)
    except:
        print('not a valid h5 file?')
        return
    #read what datasets are in the file
    if len(keys3d) == 0:
        print('no suitable datasets are found in this file')
        return
    if len(keys3d) == 1:
        ui.custom_flat_path_edit.setText(keys3d[0])
        ui.custom_flat_filename_display.setText(loadfilename)
    if len(keys3d) > 1:
        #first delete existing fields, if any:
        try:
            for i in dsui.radioButtons:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dim0labels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dim1labels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dim2labels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.dtypelabels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            for i in dsui.sizelabels:
                dsui.gridLayout.removeWidget(i)
                i.deleteLater()
            print('cleared')
        except Exception as error:
            pass#print(error) #no pre-existing idatasets
        dsui.radioButtons = []
        dsui.dim0labels = []
        dsui.dim1labels = []
        dsui.dim2labels = []
        dsui.dtypelabels = []
        dsui.sizelabels = []
        for i in range (0, len(keys3d)):
            h5dataset = h5py.File(filename[0], 'r')[keys3d[i]]
            dsui.radioButtons.append(QtWidgets.QRadioButton(DatasetDialog))
            dsui.radioButtons[i].setObjectName(f"radioButton{i}")
            dsui.radioButtons[i].setText(keys3d[i])
            dsui.gridLayout.addWidget(dsui.radioButtons[i], i+1, 0, 1, 1)
            dsui.dim0labels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dim0labels[i].setObjectName(f"dim0labels{i}")
            dsui.gridLayout.addWidget(dsui.dim0labels[i], i+1, 1, 1, 1)
            dsui.dim0labels[i].setText(str(h5dataset.shape[0]))
            dsui.dim1labels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dim1labels[i].setObjectName(f"dim1labels{i}")
            dsui.gridLayout.addWidget(dsui.dim1labels[i], i+1, 2, 1, 1)
            dsui.dim1labels[i].setText(str(h5dataset.shape[1]))
            dsui.dim2labels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dim2labels[i].setObjectName(f"dim2labels{i}")
            dsui.gridLayout.addWidget(dsui.dim2labels[i], i+1, 3, 1, 1)
            dsui.dim2labels[i].setText(str(h5dataset.shape[2]))
            dsui.dtypelabels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.dtypelabels[i].setObjectName(f"dtypelabels{i}")
            dsui.gridLayout.addWidget(dsui.dtypelabels[i], i+1, 4, 1, 1)
            dsui.dtypelabels[i].setText(str(h5dataset.dtype))
            dsui.sizelabels.append(QtWidgets.QLabel(DatasetDialog))
            dsui.sizelabels[i].setObjectName(f"sizelabels{i}")
            dsui.gridLayout.addWidget(dsui.sizelabels[i], i+1, 5, 1, 1)
            dsui.sizelabels[i].setText(str(np.round(h5dataset.nbytes*1e-9, decimals = 2)))
        dsui.radioButtons[0].setChecked(True)
        if DatasetDialog.exec():
            #dataset was selected and OK clicked, let's find out which dataset
            for i in dsui.radioButtons:
                if i.isChecked():
                    ui.custom_flat_path_edit.setText(i.text())
            ui.custom_flat_filename_display.setText(loadfilename)
        #now actually load the data
    myslices = [slice(0,None,1),
                slice(tori['file_definitions']['slice_ver'][0], tori['file_definitions']['slice_ver'][1], tori['file_definitions']['slice_ver'][2]),
                slice(tori['file_definitions']['slice_hor'][0], tori['file_definitions']['slice_hor'][1], tori['file_definitions']['slice_hor'][2])]
    try:
        block_gui('Wait for flat data to load...', 0)
        flat = load_3d_data(loadfilename, ui.custom_flat_path_edit.text(), myslices)
        tori['file_definitions']['flat_file'] = loadfilename
        tori['file_definitions']['flat_path'] = ui.custom_flat_path_edit.text()
    except:
        ui.statusbar.showMessage('something went wrong...')
        unblock_gui()
        return
    unblock_gui()
    image_source_changed()

def normalize():
    '''Flat and dark field normalization of data
    '''
    global tomo, dark, flat, history, ncore, tori
    if ui.dark_correction_check.isChecked() + ui.flat_correction_check.isChecked() == False: #no correction is checked
        print('no correction selected, skip')
        return
    #tori['correction_arguments']['normalize'] = True
    block_gui('wait for normalization...', 0)
    tori['correction_arguments']['normalize_dark'] = False
    tori['correction_arguments']['normalize_flat'] = False
    if ui.flat_correction_check.isChecked():# and flat != None: #prepare flats for normalization
        eff_flat = flat
        if ui.flat_correction_mean_check.isChecked() and len(flat.shape) == 3: #use mean of all flats it flats are a 3d array
            block_gui('calculating mean of all projections...', 0)
            eff_flat = np.mean(tomo, axis = 0, dtype = 'float32')
            unblock_gui()
        if not ui.flat_correction_mean_check.isChecked() and flat.shape != tomo.shape and len(flat.shape) == 3:#there are multiple flats, but not as many as projections, and flats are a 3d array
            eff_flat = np.mean(flat, axis = 0, dtype = 'float32')
    if ui.dark_correction_check.isChecked():# and dark != None: #prepare darks for normalization
        eff_dark = dark
        if len(dark.shape) == 3 and dark.shape != tomo.shape:#there are multiple flats, but not as many as projections
            eff_dark = np.mean(dark, axis = 0, dtype = 'float32')

    if not ui.dark_correction_check.isChecked(): #only flat field correction
        try:
            flat.shape
        except:
            ui.statusbar.showMessage("No flat field loaded yet")
            unblock_gui()
            return
        #split the task into 20 smaller pieces for progress bar indication
        chunksize = tomo.shape[0]/20
        tomonorm = np.zeros(shape=tomo.shape, dtype='float32')
        for i in range (0,20): #do steps of 5% till 95%
            tomonorm[int(np.round(i*chunksize)):int(np.round((i+1)*chunksize))] = np.divide(tomo[int(np.round(i*chunksize)):int(np.round((i+1)*chunksize))], eff_flat, dtype='float32')
            block_gui('wait for normalization...', int(i*100/20))
        tomo = tomonorm
        tomonorm = []
        tori['correction_arguments']['normalize_flat'] = True
        unblock_gui()
        ui.normalize_btn.setEnabled(False) #to prevent double normalization
        plot_slider_update()
        return
    if not ui.flat_correction_check.isChecked(): #only dark field correction
        try:
            dark.shape
        except:
            ui.statusbar.showMessage("No dark field loaded yet")
            unblock_gui()
            return
        tomo = np.subtract(tomo, eff_dark, dtype='float32')
        unblock_gui()
        ui.normalize_btn.setEnabled(False) #to prevent double normalization
        tori['correction_arguments']['normalize_dark'] = True
        plot_slider_update()
        return 
    #full normalization happens here
    try:
        flat.shape
    except:
        ui.statusbar.showMessage("No flat field loaded yet")
        unblock_gui()
        return
    try:
        dark.shape
    except:
        ui.statusbar.showMessage("No dark field loaded yet")
        unblock_gui()
        return
    #prepare flats
    eff_flat = flat
    if ui.flat_correction_mean_check.isChecked() and len(flat.shape) == 3: #use mean of all flats it flats are a 3d array
        block_gui('calculating mean of all projections...', 0)
        eff_flat = np.mean(tomo, axis = 0, dtype = 'float32')
        unblock_gui()
    if not ui.flat_correction_mean_check.isChecked() and flat.shape != tomo.shape and len(flat.shape) == 3:#there are multiple flats, but not as many as projections, and flats are a 3d array
        eff_flat = np.mean(flat, axis = 0, dtype = 'float32')
    #prepare darks for normalization
    eff_dark = dark
    if len(dark.shape) == 3 and dark.shape != tomo.shape:#there are multiple flats, but not as many as projections
        eff_dark = np.mean(dark, axis = 0, dtype = 'float32')

    #split the task into 20 smaller pieces for progress bar indication
    chunksize = tomo.shape[0]/20
    tomonorm = np.zeros(shape=tomo.shape, dtype='float32')
    for i in range (0,20):
        tomonorm[int(np.round(i*chunksize)):int(np.round((i+1)*chunksize))] = tomopy.prep.normalize.normalize(tomo[int(np.round(i*chunksize)):int(np.round((i+1)*chunksize))], eff_flat, eff_dark, ncore=ncore)
        block_gui('wait for normalization...', int(i*100/20))
    tomo = tomonorm
    tomonorm = []
    tori['correction_arguments']['normalize_dark'] = True
    tori['correction_arguments']['normalize_flat'] = True
    print('normalized!')
    unblock_gui()
    #normalization_slider_update()
    plot_slider_update()
    ui.silx_plot.resetZoom()
    ui.normalize_btn.setEnabled(False) #to prevent double normalization

def normalize_to_unity():
    '''Normalize each frame to unity (experimental feature to compensate for inhomogeneous environment in local tomo of a "pure phase" object)
    '''
    global tomo
    block_gui('Normalizing each projection to unity...', 0)
    for i in range (0, tomo.shape[0]):
        block_gui('Normalizing each projection to unity...', int(100*i/tomo.shape[0]))
        tomo[i] = tomo[i]/np.mean(tomo[i])
    ui.unity_normalize_btn.setEnabled(False)
    tori['correction_arguments']['normalize_to_unity'] = True
    unblock_gui()

def i0_correction():
    '''
    corrects each frame to the i0 monitor data from the dxchange file
    '''
    global tori, tomo, i0_data
    block_gui('Normalizing data to I0 monitor...', 0)
    for i in range (0, tomo.shape[0]):
        block_gui('Normalizing each projection to unity...', int(100*i/tomo.shape[0]))
        tomo[i] = tomo[i]/i0_data[i]
    tori['correction_arguments']['normalize_i0'] = True
    unblock_gui()
    ui.i0_normalize_btn.setEnabled(False)

def phase_proj_slice_switched():
    '''updating the GUI parameters according to data shape
    '''
    if ui.phase_proj_slice_select.currentText() == 'projection':
        ui.phase_slider.setMaximum(tomo.shape[0]-1)
        ui.phase_slider.setMinimum(0)
        ui.phase_slider.setValue(0)
        ui.phase_index.setText(str(ui.phase_slider.value()))
    if ui.phase_proj_slice_select.currentText() == 'slice':
        ui.phase_slider.setMaximum(tomo.shape[1]-6)
        ui.phase_slider.setMinimum(5)
        ui.phase_slider.setValue(tomo.shape[1]//2)
        ui.phase_index.setText(str(ui.phase_slider.value()))
    image.setData(np.zeros(shape=(500,500), dtype = 'float32'))
    if ui.phase_autoupd_chk.isChecked():
        single_phase_retrieve()
    ui.silx_plot.resetZoom()

def flatten_preview():
    '''Tlattens a current image, a low-pass filter
    '''
    try:
        filter_size = abs(int(ui.flatten_filter_size_input.text()))
        ui.flatten_filter_size_input.setText(str(filter_size))
    except:
        ui.statusbar.showMessage('Provide a valid filter size')
        return
    block_gui('Flattening image...', 0)
    curim = tomo[ui.plot_slider.value()]
    filteredim = ndimage.uniform_filter(curim, size = filter_size)
    resim = curim - filteredim + 1
    image.setData(resim)
    unblock_gui()

def flatten_all():
    '''Flatten all projections (a low-pass filter)
    '''
    try:
        filter_size = abs(int(ui.flatten_filter_size_input.text()))
        ui.flatten_filter_size_input.setText(str(filter_size))
    except:
        ui.statusbar.showMessage('Provide a valid filter size')
        return
    block_gui('Flattening all images...', 0)
    for i in range (0, tomo.shape[0]):
        block_gui('Flattening all images...', int(100*i/tomo.shape[0]))
        filteredim = ndimage.uniform_filter(tomo[i], size = filter_size)
        tomo[i] = tomo[i] - filteredim + 1
    tori['correction_arguments']['flatten_images'] = True
    tori['correction_arguments']['flatten_images_kwargs']['flatten_size'] = filter_size
    unblock_gui()
    ui.flatten_projections_btn.setEnabled(False)
    plot_slider_update()

def tab_selected():
    '''takes care of GUI when a different tab is selected
    '''
    global tori, theta, history, reconhistory
    fix_skipped_ranges()
    tabindex = ui.controls_tabs.currentIndex()
    for c in excl_curves:
        #c.__dict__['_visible'] = False
        c.setVisible(False)
    if tabindex == 0: #load data tab
        ui.controls_tabs.setMaximumHeight(136)
        ui.plots_verticalLayout_01.hide()
        ui.silx_plot.resetZoom()
        ui.plot_selector.clear()
        ui.plot_selector.addItem('Projections')
        ui.plot_selector.addItem('Flat field')
        ui.plot_selector.addItem('Dark field')
        ui.plot_selector.show()#setHidden(True)
        ui.plot_slider.show()#setHidden(True)
        ui.plot_index.show()#setHidden(True)
        crop_selection.hide()
        postp_crop_selection.hide()
        plot_slider_update()
        #scalebar_load.default_scale()
    if tabindex == 1: #normalization and cropping tab
        ui.controls_tabs.setMaximumHeight(210)
        ui.plots_verticalLayout_01.hide()
        ui.silx_plot.resetZoom()
        ui.plot_selector.clear()
        ui.plot_selector.addItem('Projections')
        ui.plot_selector.show()#setHidden(True)
        ui.plot_slider.show()#setHidden(True)
        ui.plot_index.show()#setHidden(True)
        postp_crop_selection.hide()
        try:
            if tomo.shape[0] == 0:
                return
        except:
            ui.statusbar.showMessage("No data loaded yet")
            return
        crop_selection.show()
        ui.crop_bottom_slider.setMaximum(tomo.shape[1]-1)
        ui.crop_top_slider.setMaximum(tomo.shape[1]-1)
        if ui.crop_top_slider.value() == 0:
            ui.crop_top_slider.setValue(tomo.shape[1]-1)
        ui.crop_left_slider.setMaximum(tomo.shape[2]-1)
        ui.crop_right_slider.setMaximum(tomo.shape[2]-1)
        if ui.crop_right_slider.value() == 0:
            ui.crop_right_slider.setValue(tomo.shape[2]-1)
        crop_sliders_update()
    if tabindex == 2: #sinogram tab
        ui.controls_tabs.setMaximumHeight(240)
        ui.plots_verticalLayout_01.hide()
        crop_selection.hide()
        postp_crop_selection.hide()
        ui.plot_selector.clear()
        ui.plot_selector.show()#setHidden(True)
        ui.plot_slider.show()#setHidden(True)
        ui.plot_index.show()#setHidden(True)
        ui.silx_plot.resetZoom()
        stripe_algorithm()
        try:
            tomo.shape
        except:
            ui.statusbar.showMessage("No data loaded yet")
            return
        if len(tomo) == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        ui.plot_slider.setMaximum(tomo.shape[1]-1)
        ui.plot_slider.setValue(int(tomo.shape[1]/2))
        ui.plot_index.setText(str(int(tomo.shape[1]/2)))
        plot_slider_update()
        #the tab always opens in the sinogram (not slice) view
        #create the excluded regions GUI elements from tori
        init_ranges = tori['reconstruction_arguments']['excluded_proj']
        for e in excl_reg_chk:
            e.setChecked(False)
        for n in range (0, len(init_ranges)):
            excl_reg_chk[n].setChecked(True)
            excl_reg_from_spin[n].setValue(int(init_ranges[n][0]))
            excl_reg_to_spin[n].setValue(int(init_ranges[n][1]))
            excl_curves[n].setVisible(True)
        ui.plot_selector.addItem('Height (sinogram)')
        ui.plot_selector.addItem('Height (slice)')
        ui.silx_plot.resetZoom()
    if tabindex == 3: #center of rotation tab
        ui.controls_tabs.setMaximumHeight(398)
        ui.plots_verticalLayout_01.show()
        crop_selection.hide()
        postp_crop_selection.hide()
        cor_algorithm_selected()
        ui.plot_selector.hide()
        ui.plot_slider.hide()
        ui.plot_index.hide()
        try:
            tomo.shape
        except:
            ui.statusbar.showMessage("No data loaded yet")
            return
        if len(tomo) == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        ui.cor_tab_slider.setMaximum(tomo.shape[1]-1) # slice selector set to vertical images dimentions
        ui.cor_tab_slider.setValue(int(tomo.shape[1]/2))
        ui.cor_tab_index.setText(str(int(tomo.shape[1]/2)))
        if tori['reconstruction_arguments']['recon_kwargs']['center'] == None:
            ui.cor_value.setText(str(tomo.shape[2]/2))
        else:
            ui.cor_value.setText(str(tori['reconstruction_arguments']['recon_kwargs']['center']))
        #now lets deal with theta
        try:
            ui.cor_start_theta_input.setText(str(tori['file_definitions']['theta_start']))
        except:
            ui.cor_start_theta_input.setText('0')
        try:
            ui.cor_end_theta_input.setText(str(tori['file_definitions']['theta_end']))
        except:
            ui.cor_end_theta_input.setText('180')
        if np.array(theta).size == 0: #in case there is no theta in memory
            ui.cor_use_theta_from_file_rbtn.setEnabled(False)
            ui.cor_calculate_theta_rbtn.setChecked(True)
        if ui.cor_use_theta_from_file_rbtn.isChecked():
            switch_to_theta_from_file()
        if ui.cor_calculate_theta_rbtn.isChecked():
            switch_to_calc_theta()
        image.setData(np.zeros(shape=[500,500], dtype = 'float32'))
        ui.silx_plot.resetZoom()
    if tabindex == 4: #phase retrieval tab is selected
        ui.controls_tabs.setMaximumHeight(322)
        ui.plots_verticalLayout_01.hide()
        ui.plot_selector.hide()
        ui.plot_slider.hide()
        ui.plot_index.hide()
        crop_selection.hide()
        postp_crop_selection.hide()
        pixel_unit_changed()
        sdd_unit_changed()
        if tori['correction_arguments']['phase_retrieval_kwargs']['energy'] == None:
            ui.phase_energy_inp.setText('20')
        else:
            ui.phase_energy_inp.setText(str(tori['correction_arguments']['phase_retrieval_kwargs']['energy']))
        try:
            tomo.shape
        except: 
            ui.statusbar.showMessage("No data loaded yet")
            return
        if len(tomo) == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        if ui.phase_proj_slice_select.currentText() == 'projection':
            ui.phase_slider.setMaximum(tomo.shape[0]-1)
            ui.phase_slider.setMinimum(0)
            ui.phase_slider.setValue(0)
            ui.phase_index.setText(str(ui.phase_slider.value()))
        if ui.phase_proj_slice_select.currentText() == 'slice':
            ui.phase_slider.setMaximum(tomo.shape[1]-6)
            ui.phase_slider.setMinimum(5)
            ui.phase_slider.setValue(tomo.shape[1]//2)
            ui.phase_index.setText(str(ui.phase_slider.value()))
        image.setData(np.zeros(shape=[500,500], dtype = 'float32'))
        ui.silx_plot.resetZoom()
    if tabindex == 5: #reconstruction tab
        ui.controls_tabs.setMaximumHeight(266)
        ui.plots_verticalLayout_01.hide()
        ui.silx_plot.resetZoom()
        ui.plot_selector.clear()
        ui.plot_selector.addItem('XY')
        ui.plot_selector.addItem('XZ')
        ui.plot_selector.addItem('YZ')
        ui.plot_selector.show()#setHidden(True)
        ui.plot_slider.show()#setHidden(True)
        ui.plot_index.show()#setHidden(True)
        crop_selection.hide()
        postp_crop_selection.hide()
        recon_algorithm_selected()
        try:
            tomo.shape
        except:
            ui.statusbar.showMessage("No data loaded yet")
            return
        if len(tomo) == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        ui.recon_first_slice_slider.setMaximum(tomo.shape[1]-1)
        ui.recon_last_slice_slider.setMaximum(tomo.shape[1]-1)
        if ui.recon_last_slice_slider.value == 0:
            ui.recon_last_slice_slider.setValue(tomo.shape[1]-1)
        image.setData(np.zeros(shape=[500,500], dtype = 'float32'))
        ui.recon_cor_label.setText(f"center of rotation: {tori['reconstruction_arguments']['recon_kwargs']['center']}")
        ui.silx_plot.resetZoom()
        plot_slider_update()
    if tabindex == 6: #postproc!
        ui.controls_tabs.setMaximumHeight(298)
        ui.plots_verticalLayout_01.hide()
        ui.plot_selector.clear()
        ui.plot_selector.addItem('XY')
        ui.plot_selector.addItem('XZ')
        ui.plot_selector.addItem('YZ')
        ui.postp_save_path.setText(tori['file_definitions']['data_file'].replace('raw','process/recon').replace('.h5', '_recon.h5'))
        image_source_changed()
        ui.silx_plot.resetZoom()
        return##!!
    if tabindex == 7: #param
        ui.controls_tabs.setMaximumHeight(301)
        ui.plots_verticalLayout_01.hide()
        postp_crop_selection.hide()
        crop_selection.hide()
        ui.silx_plot.resetZoom()
        param_type_changed()
        return##!!

def excl_regions_changed():
    '''Takes care of the excluded theta regions display
    '''
    global tori
    excl_list = []
    for n in range (0, 6):
        if excl_reg_chk[n].isChecked():
            excl_list.append([excl_reg_from_spin[n].value(),excl_reg_to_spin[n].value()])
            excl_curves[n].setData(x=[0, tomo.shape[2]], y=[excl_reg_to_spin[n].value(),excl_reg_to_spin[n].value()])
            excl_curves[n].__dict__['_baseline'] = excl_reg_from_spin[n].value()
            excl_curves[n].setVisible(True)
        else:
            excl_curves[n].setVisible(False)
    tori['reconstruction_arguments']['excluded_proj'] = excl_list
    ui.silx_plot.resetZoom()

def param_type_changed():
    '''switch between tori/python script view. would be removed in future versions
    '''
    ui.param_text_edit.clear()
    global tori
    if ui.param_tori_radio.isChecked():
        ui.param_text_edit.insertPlainText(json.dumps(tori, indent = 4))
    if ui.param_python_radio.isChecked():
        script = generate_python_script(tori)
        script_line = ''
        for l in script:
            script_line = script_line + l + '\n'
        ui.param_text_edit.insertPlainText(script_line)

def plot_slider_update():
    '''Updates the 2D plot content
    '''
    global tomo, recon, tori
    if ui.plot_selector.currentText() == 'Projections':
        if tomo.size == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        image.setData(tomo[ui.plot_slider.value()])
    if ui.plot_selector.currentText() == 'Flat field':
        if flat.size == 0:
            ui.statusbar.showMessage("No flat field loaded yet")
            ui.plot_selector.setCurrentIndex(0)
            ui.plot_selector.update()
            return
        image.setData(flat[ui.plot_slider.value()])
    if ui.plot_selector.currentText() == 'Dark field':
        if dark.size == 0:
            ui.statusbar.showMessage("No dark field loaded yet")
            ui.plot_selector.setCurrentIndex(0)
            ui.plot_selector.update()
            unblock_gui()
            return
        image.setData(dark[ui.plot_slider.value()])
    if ui.plot_selector.currentText() == 'Height (sinogram)':
        if tomo.size == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        image.setData(tomo[:,ui.plot_slider.value(),:])
        excl_regions_changed()
    if ui.plot_selector.currentText() == 'Height (slice)':
        if tomo.size == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        block_gui('Reconstructing single slice...', 0)
        single_slice=np.empty(shape=[tomo.shape[0],1,tomo.shape[2]])
        single_slice[:,0,:] = tomo[:,ui.plot_slider.value(),:]
        try:
            COR = float(ui.cor_value.text())
        except:
            ui.statusbar.showMessage("Center of rotation is not specified yet")
            unblock_gui()
            return
        if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
            recon_theta = np.radians(theta)
        else:
            recon_theta = theta

        if tori['correction_arguments']['minus_log_before_phase'] == False and tori['correction_arguments']['minus_log_after_phase'] == False:
            single_slice = -np.log(single_slice)
        single_slice = tomopy.misc.corr.remove_nan(single_slice, ncore=ncore)
        padsize = tomo.shape[2]//4
        single_slice = tomopy.misc.morph.pad(single_slice, axis=2, npad=padsize, mode='edge')
        COR = COR + padsize
        mask = theta_mask()
        test_slice = tomopy.recon(single_slice[mask], recon_theta[mask], COR, algorithm='gridrec', filter_name='ramlak', sinogram_order=False)
        test_slice = test_slice[:,padsize:-padsize,padsize:-padsize]
        for c in excl_curves:
            c.setVisible(False)
        image.setData(test_slice[0])
    if ui.plot_selector.currentText() == 'XY':
        if recon.size == 0:
            ui.statusbar.showMessage("Do the reconstruction first")
            return
        image.setData(recon[ui.plot_slider.value(),:,:])
    if ui.plot_selector.currentText() == 'YZ':
        if recon.size == 0:
            ui.statusbar.showMessage("Do the reconstruction first")
            return
        image.setData(recon[:,:,ui.plot_slider.value()])
    if ui.plot_selector.currentText() == 'XZ':
        if recon.size == 0:
            ui.statusbar.showMessage("Do the reconstruction first")
            return
        image.setData(recon[:,ui.plot_slider.value(),:])
    ui.plot_index.setText(str(ui.plot_slider.value()))
    if ui.plot_selector.currentText() == 'filters':
        try:
            image.setData(all_filters_slices[ui.plot_slider.value()])
            ui.plot_index.setText([ui.recon_filter_select.itemText(i) for i in range(ui.recon_filter_select.count())][ui.plot_slider.value()])
        except Exception as error:
            print(error)
    draw_a_circular_mask()
    unblock_gui()
    #ui.silx_load_data.resetZoom()

def plot_index_update():
    '''Update the 2D image content if an intex has been changed
    '''
    try:
        ui.plot_slider.setValue(int(ui.plot_index.text()))
    except ValueError:
        #input is incorrect, read the current slider value
        pass
    plot_slider_update()

def crop_recon():
    '''cropping the reconstructed data
    '''
    global recon, tori
    tori['post_recon_process_arguments']['crop']={}
    if ui.plot_selector.currentText() == 'XY':
        block_gui('cropping reconstruction in XY range...', 0)
        recon = recon[:,ui.postp_bottom_slider.value():ui.postp_top_slider.value()+1,ui.postp_left_slider.value():ui.postp_right_slider.value()+1]
        tori['post_recon_process_arguments']['crop']['range_1'] = [ui.postp_bottom_slider.value(), ui.postp_top_slider.value()+1]
        tori['post_recon_process_arguments']['crop']['range_2'] = [ui.postp_left_slider.value(), ui.postp_right_slider.value()+1]
    if ui.plot_selector.currentText() == 'YZ':
        block_gui('cropping reconstruction in YZ range...', 0)
        recon = recon[ui.postp_bottom_slider.value():ui.postp_top_slider.value()+1,ui.postp_left_slider.value():ui.postp_right_slider.value()+1,:]
        tori['post_recon_process_arguments']['crop']['range_0'] = [ui.postp_bottom_slider.value(), ui.postp_top_slider.value()+1]
        tori['post_recon_process_arguments']['crop']['range_1'] = [ui.postp_left_slider.value(), ui.postp_right_slider.value()+1]
    if ui.plot_selector.currentText() == 'XZ':
        block_gui('cropping reconstruction in XZ range...', 0)
        recon = recon[ui.postp_bottom_slider.value():ui.postp_top_slider.value()+1,:,ui.postp_left_slider.value():ui.postp_right_slider.value()+1]
        tori['post_recon_process_arguments']['crop']['range_0'] = [ui.postp_bottom_slider.value(), ui.postp_top_slider.value()+1]
        tori['post_recon_process_arguments']['crop']['range_2'] = [ui.postp_left_slider.value(), ui.postp_right_slider.value()+1]
    image_source_changed()
    unblock_gui()


def image_source_changed():
    '''Updated the 2D plot if a different image source option is selected
    '''
    global tomo, recon
    if ui.plot_selector.currentText() == 'Projections':
        if tomo.size == 0:
            ui.statusbar.showMessage("No data loaded yet")
            return
        ui.plot_slider.setValue(0)
        ui.plot_slider.setMinimum(0)
        ui.plot_slider.setMaximum(tomo.shape[0]-1)
    if ui.plot_selector.currentText() == 'Flat field':
        if flat.size == 0:
            ui.statusbar.showMessage("No flat field loaded yet")
            return
        ui.plot_slider.setValue(0)
        ui.plot_slider.setMinimum(0)
        ui.plot_slider.setMaximum(flat.shape[0]-1)
    if ui.plot_selector.currentText() == 'Dark field':
        if dark.size == 0:
            ui.statusbar.showMessage("No dark field loaded yet")
            return
        ui.plot_slider.setValue(0)
        ui.plot_slider.setMinimum(0)
        ui.plot_slider.setMaximum(dark.shape[0]-1)
    if ui.plot_selector.currentText() == 'XY':
        if recon.size == 0:
            ui.statusbar.showMessage("Do the reconstruction first")
            return
        ui.plot_slider.setValue(recon.shape[0]//2)
        ui.plot_slider.setMinimum(0)
        ui.plot_slider.setMaximum(recon.shape[0]-1)
        ui.postp_top_slider.setMaximum(recon.shape[1]-1)
        ui.postp_top_slider.setValue(recon.shape[1]-1)
        ui.postp_bottom_slider.setMaximum(recon.shape[1]-1)
        ui.postp_bottom_slider.setValue(0)
        ui.postp_left_slider.setMaximum(recon.shape[2]-1)
        ui.postp_left_slider.setValue(0)
        ui.postp_right_slider.setMaximum(recon.shape[2]-1)
        ui.postp_right_slider.setValue(recon.shape[2]-1)
        postp_crop_checked()
        image.setData(recon[ui.plot_slider.value(),:,:])
    if ui.plot_selector.currentText() == 'YZ':
        if recon.size == 0:
            ui.statusbar.showMessage("Do the reconstruction first")
            return
        ui.plot_slider.setMinimum(0)
        ui.plot_slider.setMaximum(recon.shape[2]-1)
        ui.plot_slider.setValue(recon.shape[2]//2)
        ui.postp_top_slider.setMaximum(recon.shape[0]-1)
        ui.postp_top_slider.setValue(recon.shape[0]-1)
        ui.postp_bottom_slider.setMaximum(recon.shape[0]-1)
        ui.postp_bottom_slider.setValue(0)
        ui.postp_left_slider.setMaximum(recon.shape[1]-1)
        ui.postp_left_slider.setValue(0)
        ui.postp_right_slider.setMaximum(recon.shape[1]-1)
        ui.postp_right_slider.setValue(recon.shape[1]-1)
        postp_crop_checked()
        image.setData(recon[:,:,ui.plot_slider.value()])
    if ui.plot_selector.currentText() == 'XZ':
        if recon.size == 0:
            ui.statusbar.showMessage("Do the reconstruction first")
            return
        ui.plot_slider.setMinimum(0)
        ui.plot_slider.setMaximum(recon.shape[1]-1)
        ui.plot_slider.setValue(recon.shape[1]//2)
        ui.postp_top_slider.setMaximum(recon.shape[0]-1)
        ui.postp_top_slider.setValue(recon.shape[0]-1)
        ui.postp_bottom_slider.setMaximum(recon.shape[0]-1)
        ui.postp_bottom_slider.setValue(0)
        ui.postp_left_slider.setMaximum(recon.shape[2]-1)
        ui.postp_left_slider.setValue(0)
        ui.postp_right_slider.setMaximum(recon.shape[2]-1)
        ui.postp_right_slider.setValue(recon.shape[2]-1)
        postp_crop_checked()
        image.setData(recon[:,ui.plot_slider.value(),:])
    if ui.plot_selector.currentText() == 'filters':
        ui.plot_slider.setMaximum(all_filters_slices.shape[0]-1)
        ui.plot_slider.setMinimum(0)
        ui.plot_slider.setValue(0)
        try:
            image.setData(all_filters_slices[ui.plot_slider.value()])
        except Exception as error:
            print(error)
    ui.excl_reg_01_from_spin.setMaximum(tomo.shape[0])
    ui.excl_reg_02_from_spin.setMaximum(tomo.shape[0])
    ui.excl_reg_03_from_spin.setMaximum(tomo.shape[0])
    ui.excl_reg_01_to_spin.setMaximum(tomo.shape[0])
    ui.excl_reg_02_to_spin.setMaximum(tomo.shape[0])
    ui.excl_reg_03_to_spin.setMaximum(tomo.shape[0])
    plot_slider_update()
    ui.silx_plot.resetZoom()

def crop_top_index_changed():
    try:
        ui.crop_top_slider.setValue(int(ui.crop_top_index.text()))
    except:
        pass
    crop_sliders_update()


def crop_bottom_index_changed():
    try:
        ui.crop_bottom_slider.setValue(int(ui.crop_bottom_index.text()))
    except:
        pass
    crop_sliders_update()

def crop_left_index_changed():
    try:
        ui.crop_left_slider.setValue(int(ui.crop_left_index.text()))
    except:
        pass
    crop_sliders_update()

def crop_right_index_changed():
    try:
        ui.crop_right_slider.setValue(int(ui.crop_right_index.text()))
    except:
        pass
    crop_sliders_update()

def postp_crop_top_index_changed():
    try:
        ui.crop_top_slider.setValue(int(ui.crop_top_index.text()))
    except:
        pass
    postp_crop_sliders_update()


def postp_crop_bottom_index_changed():
    try:
        ui.crop_bottom_slider.setValue(int(ui.crop_bottom_index.text()))
    except:
        pass
    postp_crop_sliders_update()

def postp_crop_left_index_changed():
    try:
        ui.crop_left_slider.setValue(int(ui.crop_left_index.text()))
    except:
        pass
    postp_crop_sliders_update()

def postp_crop_right_index_changed():
    try:
        ui.crop_right_slider.setValue(int(ui.crop_right_index.text()))
    except:
        pass
    postp_crop_sliders_update()


def crop_sliders_update():
    #don't let top be below bottom, same for hor
    if ui.crop_top_slider.value() <= ui.crop_bottom_slider.value():
        ui.crop_top_slider.setValue(ui.crop_bottom_slider.value())
        ui.crop_bottom_slider.setValue(ui.crop_top_slider.value())
    if ui.crop_right_slider.value() <= ui.crop_left_slider.value():
        ui.crop_right_slider.setValue(ui.crop_left_slider.value())
        ui.crop_left_slider.setValue(ui.crop_right_slider.value())
    #update index fields texts
    ui.crop_top_index.setText(str(ui.crop_top_slider.value()))
    ui.crop_bottom_index.setText(str(ui.crop_bottom_slider.value()))
    ui.crop_left_index.setText(str(ui.crop_left_slider.value()))
    ui.crop_right_index.setText(str(ui.crop_right_slider.value()))
    #update crop rectangle
    crop_selection.crop_changed()
#    cropcurve = ui.silx_crop.getCurve('crop')
#    cropcurve.setData(x=[ui.crop_left_slider.value(),ui.crop_right_slider.value()+1,ui.crop_right_slider.value()+1,ui.crop_left_slider.value(),ui.crop_left_slider.value()], y=[ui.crop_top_slider.value()+1,ui.crop_top_slider.value()+1,ui.crop_bottom_slider.value(),ui.crop_bottom_slider.value(),ui.crop_top_slider.value()+1])
    ui.silx_plot.resetZoom()


def postp_crop_sliders_update():
    #don't let top be below bottom, same for hor
    if ui.postp_top_slider.value() <= ui.postp_bottom_slider.value():
        ui.postp_top_slider.setValue(ui.postp_bottom_slider.value())
        ui.postp_bottom_slider.setValue(ui.postp_top_slider.value())
    if ui.postp_right_slider.value() <= ui.postp_left_slider.value():
        ui.postp_right_slider.setValue(ui.postp_left_slider.value())
        ui.postp_left_slider.setValue(ui.postp_right_slider.value())
    #update index fields texts
    ui.postp_top_index.setText(str(ui.postp_top_slider.value()))
    ui.postp_bottom_index.setText(str(ui.postp_bottom_slider.value()))
    ui.postp_left_index.setText(str(ui.postp_left_slider.value()))
    ui.postp_right_index.setText(str(ui.postp_right_slider.value()))
    #update crop rectangle
    postp_crop_selection.crop_changed()
    postp_crop_checked()
#    cropcurve = ui.silx_crop.getCurve('crop')
#    cropcurve.setData(x=[ui.crop_left_slider.value(),ui.crop_right_slider.value()+1,ui.crop_right_slider.value()+1,ui.crop_left_slider.value(),ui.crop_left_slider.value()], y=[ui.crop_top_slider.value()+1,ui.crop_top_slider.value()+1,ui.crop_bottom_slider.value(),ui.crop_bottom_slider.value(),ui.crop_top_slider.value()+1])
    ui.silx_plot.resetZoom()


def crop_data():
    '''Croping the projections in X and Y after loading
    '''
    global tomo, dark, flat, history, tori
    try:
        tomo.shape
    except:
        ui.statusbar.showMessage("No data loaded yet")
        return    
    block_gui('wait for data cropping...', 0)
    tomo = tomo[:,ui.crop_bottom_slider.value():ui.crop_top_slider.value()+1,ui.crop_left_slider.value():ui.crop_right_slider.value()+1]
    tori['correction_arguments']['crop'] = True
    tori['correction_arguments']['crop_kwargs']['crop_ranges'] = [ui.crop_bottom_slider.value(), ui.crop_top_slider.value()+1,ui.crop_left_slider.value(), ui.crop_right_slider.value()+1]
    try:
        dark.shape
        print('cropping dark')
        #dark is loaded, cropping dark as well
        dark = dark[:,ui.crop_bottom_slider.value():ui.crop_top_slider.value()+1,ui.crop_left_slider.value():ui.crop_right_slider.value()+1]
        history.append('dark = dark[:,'+str(ui.crop_bottom_slider.value())+':'+str(ui.crop_top_slider.value()+1)+','+str(ui.crop_left_slider.value())+':'+str(ui.crop_right_slider.value()+1)+']')
    except:
        pass
    try:
        flat.shape
        print('cropping flat')
        #flat field is loaded, cropping it
        flat = flat[:,ui.crop_bottom_slider.value():ui.crop_top_slider.value()+1,ui.crop_left_slider.value():ui.crop_right_slider.value()+1]
        history.append('flat = flat[:,'+str(ui.crop_bottom_slider.value())+':'+str(ui.crop_top_slider.value()+1)+','+str(ui.crop_left_slider.value())+':'+str(ui.crop_right_slider.value()+1)+']')
    except:
        pass
    ui.crop_bottom_slider.setValue(0)
    ui.crop_left_slider.setValue(0)
    tab_selected()
    #print(history)
    unblock_gui()

def stripe_algorithm():
    '''Fixing GUI and parameters depending on which stripe removal method is selected
    '''
    global stripe_arguments
    stripe_command = getattr(tomopy, ui.stripe_algorithm_select.currentText())#  eval(f'tomopy.{ui.stripe_algorithm_select.currentText()}')
    ui.stripe_algorithm_select.setToolTip(stripe_command.__doc__)
    helpstring = stripe_command.__doc__.splitlines()
    stripe_arguments = {}
    for i, s in enumerate(helpstring):
        if ' : ' in s:
            stripe_arguments[s.split(':')[0].strip()] = {}
            stripe_arguments[s.split(':')[0].strip()]['type'] = s.split(':')[1].split(',')[0].strip()
            stripe_arguments[s.split(':')[0].strip()]['description'] = helpstring[i+1].strip()
    #delete irrelevant arguments
    if 'tomo' in stripe_arguments.keys():
        del(stripe_arguments['tomo'])
    if 'ncore' in stripe_arguments.keys():
        del(stripe_arguments['ncore'])
    if 'nchunk' in stripe_arguments.keys():
        del(stripe_arguments['nchunk'])
    if 'arr' in stripe_arguments.keys():
        del(stripe_arguments['arr'])
    ui.stripe_pad_chk.setVisible('pad' in stripe_arguments.keys())
    if 'pad' in stripe_arguments.keys():
        del(stripe_arguments['pad'])
    ui.stripe_norm_chk.setVisible('norm' in stripe_arguments.keys())
    if 'norm' in stripe_arguments.keys():
        del(stripe_arguments['norm'])
    print('##################################################')
    print(stripe_arguments)
    print('##################################################')
    #hide non-relevant input fiends
    ui.stripe_param2_label.setVisible(len(stripe_arguments) > 1)
    ui.stripe_param2_input.setVisible(len(stripe_arguments) > 1)
    ui.stripe_param3_label.setVisible(len(stripe_arguments) > 2)
    ui.stripe_param3_input.setVisible(len(stripe_arguments) > 2)
    ui.stripe_param4_label.setVisible(len(stripe_arguments) > 3)
    ui.stripe_param4_input.setVisible(len(stripe_arguments) > 3)
    try:
        ui.stripe_param1_label.setText(list(stripe_arguments.keys())[0])
        ui.stripe_param1_input.setToolTip(stripe_arguments[list(stripe_arguments.keys())[0]]['description'])
        ui.stripe_param2_label.setText(list(stripe_arguments.keys())[1])
        ui.stripe_param2_input.setVisible(True)
        ui.stripe_param2_input.setToolTip(stripe_arguments[list(stripe_arguments.keys())[1]]['description'])
        ui.stripe_param3_label.setText(list(stripe_arguments.keys())[2])
        ui.stripe_param3_input.setVisible(True)
        ui.stripe_param3_input.setToolTip(stripe_arguments[list(stripe_arguments.keys())[2]]['description'])
        ui.stripe_param4_label.setText(list(stripe_arguments.keys())[3])
        ui.stripe_param4_input.setVisible(True)
        ui.stripe_param4_input.setToolTip(stripe_arguments[list(stripe_arguments.keys())[3]]['description'])
    except Exception as error:
        print(error)
        pass

def convert_arg(argument, raw_text):
    '''Fixing arguments for stripe removal
    '''
    print(f'argument = {argument}')
    if 'tuple' in argument['type']:
        if 'float' in argument['type']:
            res = (float(raw_text.split(',')[0]), float(raw_text.split(',')[1]))
        else:
            res = (int(raw_text.split(',')[0]), int(raw_text.split(',')[1]))
    elif '{' in argument['type']:
        res = (int(raw_text.split(',')[0]), int(raw_text.split(',')[1]))
    elif 'int' in argument['type']:
        res = int(raw_text)
    elif 'float' in argument['type']:
        res = float(raw_text)
    elif 'str' in argument['type']:
        res = raw_text
    print(f'parsed argument = {res}, type = {type(res)}')
    return res


def stripe_one_slice():
    '''Trying stripe removal on a single sinogram
    '''
    global stripe_function_name, tomo, stripe_arguments
    block_gui('Removing stripes (single frame)...', 0)
    stripe_func = getattr(tomopy.prep.stripe, ui.stripe_algorithm_select.currentText())
    single_sino=np.empty(shape=[tomo.shape[0],1,tomo.shape[2]])
    single_sino[:,0,:] = tomo[:,ui.plot_slider.value(),:]
    print('')
    #getting arguments for the stripe function
    args = {}
    args['tomo']=single_sino
    if ui.stripe_param1_input.text() != '':
        args[list(stripe_arguments.keys())[0]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[0]], ui.stripe_param1_input.text())
    if ui.stripe_param2_input.text() != '' and ui.stripe_param2_input.isVisible():
        args[list(stripe_arguments.keys())[1]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[1]], ui.stripe_param2_input.text())
    if ui.stripe_param3_input.text() != '' and ui.stripe_param3_input.isVisible():
        args[list(stripe_arguments.keys())[2]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[2]], ui.stripe_param3_input.text())
    if ui.stripe_param4_input.text() != '' and ui.stripe_param4_input.isVisible():
        args[list(stripe_arguments.keys())[3]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[3]], ui.stripe_param4_input.text())
    if ui.stripe_pad_chk.isVisible():
        args['pad'] = ui.stripe_pad_chk.isChecked()
    if ui.stripe_norm_chk.isVisible():
        args['norm'] = ui.stripe_norm_chk.isChecked()
    test_stripe = stripe_func(**args)
    #update plot
    if ui.plot_selector.currentText() == 'Height (sinogram)':
        image.setData(test_stripe[:,0,:])
    if ui.plot_selector.currentText() == 'Height (slice)':
        block_gui('Reconstructing single slice...', 0)
        try:
            COR = float(ui.cor_value.text())
        except:
            ui.statusbar.showMessage("Center of rotation is not specified yet")
            unblock_gui()
            return
        if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
            recon_theta = np.radians(theta)
        else:
            recon_theta = theta
        if tori['correction_arguments']['minus_log_before_phase'] == False and tori['correction_arguments']['minus_log_after_phase'] == False:
            test_stripe = -np.log(test_stripe)
        test_stripe = tomopy.misc.corr.remove_nan(test_stripe, ncore=ncore)
        padsize = tomo.shape[2]//4
        test_stripe = tomopy.misc.morph.pad(test_stripe, axis=2, npad=padsize, mode='edge')
        COR = COR + padsize
        mask = theta_mask()
        test_slice = tomopy.recon(test_stripe[mask], recon_theta[mask], COR, algorithm='gridrec', filter_name='ramlak', sinogram_order=False)
        test_slice = test_slice[:,padsize:-padsize,padsize:-padsize]
        image.setData(test_slice[0])
    unblock_gui()
    ui.plot_index.setText(str(ui.plot_slider.value()))
    #ui.silx_plot.resetZoom()

def remove_stripes():
    '''Apply selected stripe removal algorithm to the whole dataset
    '''
    global tomo, history, tor, stripe_arguments
    block_gui('wait for stripe artifacts removal...', 0)

    stripe_func = getattr(tomopy, ui.stripe_algorithm_select.currentText())

    #getting arguments for the stripe function
    myargs = {}
    if ui.stripe_param1_input.text() != '':
        myargs[list(stripe_arguments.keys())[0]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[0]], ui.stripe_param1_input.text())
    if ui.stripe_param2_input.text() != '' and ui.stripe_param2_input.isVisible():
        myargs[list(stripe_arguments.keys())[1]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[1]], ui.stripe_param2_input.text())
    if ui.stripe_param3_input.text() != '' and ui.stripe_param3_input.isVisible():
        myargs[list(stripe_arguments.keys())[2]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[2]], ui.stripe_param3_input.text())
    if ui.stripe_param4_input.text() != '' and ui.stripe_param4_input.isVisible():
        myargs[list(stripe_arguments.keys())[3]] = convert_arg(stripe_arguments[list(stripe_arguments.keys())[3]], ui.stripe_param4_input.text())
    if ui.stripe_pad_chk.isVisible():
        myargs['pad'] = ui.stripe_pad_chk.isChecked()
    if ui.stripe_norm_chk.isVisible():
        myargs['norm'] = ui.stripe_norm_chk.isChecked()

    print(f'stripe arguments were {myargs}')
    chunksize = tomo.shape[1]/20
    print(50*'!')
    if chunksize < 10:
        #do in a single go
        block_gui('wait for stripe artifacts removal in a signle chunk...', 0)
        #args['tomo'] = tomo
        tomo = stripe_func(tomo, **myargs)#stripe_func(tomo = tomo, **myargs)
        #print('')
    else:
        #unstriped_proj = np.zeros(shape = tomo.shape, dtype = 'float32')
        for i in range (0,20):
            block_gui('wait for stripe artifacts removal...', int(i*5))
            myslice = slice(int(np.round(i*chunksize)), int(np.round((i+1)*chunksize)))
            #args['tomo'] = tomo[:,myslice,:]
            tomo[:,myslice,:] = stripe_func(tomo[:,myslice,:], **myargs)
        #del(myslice)
    print(f'stripe my arguments were {myargs}')
    #print(f'stripe arguments were {args}')
    #print(f'args tomo shape is {args["tomo"].shape}')
    #del(args['tomo'])
    tori['correction_arguments']['stripe_removal'] = True
    tori['correction_arguments']['stripe_function'] = ui.stripe_algorithm_select.currentText()
    tori['correction_arguments']['stripe_removal_kwargs'] = myargs
    unblock_gui()
    plot_slider_update()

def outliers_one_slice():
    '''preview outliers removal on a single sinogram
    '''
    try:
        difference = abs(float(ui.outlier_dif_input.text()))
    except:
        ui.statusbar.showMessage("provide a valid intensity difference value")
        return
    block_gui('Removing outliers (single frame)...', 0)
    single_sino=np.empty(shape=[tomo.shape[0],1,tomo.shape[2]])
    single_sino[:,0,:] = tomo[:,ui.plot_slider.value(),:]
    test_stripe = tomopy.misc.corr.remove_outlier(single_sino, difference, ui.outlier_filter_size_spin.value(), ncore=ncore)
    #update plot
    if ui.plot_selector.currentText() == 'Height (sinogram)':
        image.setData(test_stripe[:,0,:])
    if ui.plot_selector.currentText() == 'Height (slice)':
        block_gui('Reconstructing single slice...', 0)
        try:
            COR = float(ui.cor_value.text())
        except:
            ui.statusbar.showMessage("Center of rotation is not specified yet")
            unblock_gui()
            return
        if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
            recon_theta = np.radians(theta)
        else:
            recon_theta = theta

        test_stripe = -np.log(test_stripe)
        test_stripe = tomopy.misc.corr.remove_nan(test_stripe, ncore=ncore)
        padsize = tomo.shape[2]//4
        test_stripe = tomopy.misc.morph.pad(test_stripe, axis=2, npad=padsize, mode='edge')
        COR = COR + padsize
        mask = theta_mask()
        test_slice = tomopy.recon(test_stripe[mask], recon_theta[mask], COR, algorithm='gridrec', filter_name='ramlak', sinogram_order=False)
        test_slice = test_slice[:,padsize:-padsize,padsize:-padsize]
        image.setData(test_slice[0])
    unblock_gui()
    ui.plot_index.setText(str(ui.plot_slider.value()))
#    ui.silx_plot.resetZoom()

def remove_outliers():
    '''remove outliers from the whole dataset
    '''
    global tomo, tori
    try:
        difference = abs(float(ui.outlier_dif_input.text()))
    except:
        ui.statusbar.showMessage("provide a valid intensity difference value")
        return
    block_gui('wait for outliers removal...', 0)
    chunksize = tomo.shape[1]/20
    if chunksize < 10:
        #do in a single go
        block_gui('wait for outliers removal...', 0)
        tomo = tomopy.misc.corr.remove_outlier(tomo, difference, ui.outlier_filter_size_spin.value(), ncore=ncore)
    else:
        #fixed_proj = np.zeros(shape = tomo.shape, dtype = 'float32')
        for i in range (0,20):
            block_gui('wait for outliers removal...', int(i*5))
            myslice = slice(int(np.round(i*chunksize)), int(np.round((i+1)*chunksize)))
            args['tomo'] = tomo[:,myslice,:]
            tomo[:,myslice,:] = tomopy.misc.corr.remove_outlier(tomo[:,myslice,:], difference, ui.outlier_filter_size_spin.value(), ncore=ncore)
    tori['correction_arguments']['outlier_removal'] = True
    tori['correction_arguments']['outlier_removal_kwargs']['dif'] = difference
    tori['correction_arguments']['outlier_removal_kwargs']['size'] = ui.outlier_filter_size_spin.value()
    unblock_gui()
    plot_slider_update()
    return

def cor_height_slider_update():
    ui.cor_tab_index.setText(str(ui.cor_tab_slider.value()))
    try:
        tomo.shape
    except:
        ui.statusbar.showMessage("No data loaded yet")
        return

def cor_height_index_udpate():
    try:
        ui.cor_tab_slider.setValue(int(ui.cor_tab_index.text()))
    except ValueError:
        #input is incorrect, read the current slider value
        pass
    cor_height_slider_update()

def switch_to_theta_from_file():
    global theta, theta_from_file
    print('would be using original source theta')
    ui.cor_start_theta_input.setEnabled(False)
    ui.cor_end_theta_input.setEnabled(False)
    theta = theta_from_file
    if debug:
        print(f'theta = {theta}')
    ui.cor_calc_ang_step_label.setText("(corresponds to step size of {} deg)".format(theta[1]-theta[0]))

def switch_to_calc_theta():
    global theta
    print('would be using calculated theta values')
    ui.cor_start_theta_input.setEnabled(True)
    ui.cor_end_theta_input.setEnabled(True)
    if ui.cor_start_theta_input.text() != '' and ui.cor_end_theta_input.text() != '':
        theta = np.linspace(float(ui.cor_start_theta_input.text()), float(ui.cor_end_theta_input.text()), num=tomo.shape[0])
        ui.cor_calc_ang_step_label.setText("(corresponds to step size of {} deg)".format(theta[1]-theta[0]))#print(theta)

def cor_algorithm_selected():
    if ui.cor_algorithm_select.currentText() == 'find_center':
        ui.cor_param1_label.setText('init')
        ui.cor_param2_label.setVisible(False)
        ui.cor_param2_input.setVisible(False)
        ui.cor_param3_label.setVisible(False)
        ui.cor_param3_input.setVisible(False)
        ui.cor_find_auto_mask_chk.setVisible(True)
        ui.cor_find_auto_mask_chk.setText('circular mask')
        ui.cor_find_auto_mask_ratio.setVisible(True)
        ui.cor_label_04.setText('tol')
    if ui.cor_algorithm_select.currentText() == 'find_center_vo':
        ui.cor_param1_label.setText('smin')
        ui.cor_param2_label.setText('srad')
        ui.cor_param2_label.setVisible(True)
        ui.cor_param2_input.setVisible(True)
        ui.cor_param3_label.setVisible(True)
        ui.cor_param3_label.setText('drop')
        ui.cor_param3_input.setVisible(True)
        ui.cor_find_auto_mask_chk.setVisible(True)
        ui.cor_find_auto_mask_chk.setText('FOV/object size ratio')
        ui.cor_find_auto_mask_ratio.setVisible(True)
        ui.cor_label_04.setText('step')
    if  ui.cor_algorithm_select.currentText() == 'find_center_pc':
        ui.cor_param1_label.setText('rotc_guess')
        ui.cor_param2_label.setVisible(False)
        ui.cor_param2_input.setVisible(False)
        ui.cor_param3_label.setVisible(False)
        ui.cor_param3_label.setVisible(False)
        ui.cor_param3_input.setVisible(False)
        ui.cor_find_auto_mask_chk.setVisible(False)
        ui.cor_find_auto_mask_ratio.setVisible(False)
        ui.cor_label_04.setText('tol')
    function = getattr(tomopy, ui.cor_algorithm_select.currentText())
    ui.cor_param1_input.setToolTip(get_description_for_param(function, ui.cor_param1_label.text()))
    ui.cor_param2_input.setToolTip(get_description_for_param(function, ui.cor_param2_label.text()))
    ui.cor_param3_input.setToolTip(get_description_for_param(function, ui.cor_param3_label.text()))
    ui.cor_find_auto_mask_ratio.setToolTip(get_description_for_param(function, ui.cor_find_auto_mask_chk.text()))
    ui.cor_find_auto_mask_chk.setToolTip(get_description_for_param(function, ui.cor_find_auto_mask_chk.text()))
    ui.autofind_COR_tolerance_select.setToolTip(get_description_for_param(function, ui.cor_label_04.text()))

def find_center_auto():
    if ui.cor_algorithm_select.currentText() == 'find_center':
        if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
            recon_theta = np.radians(theta)
        else:
            recon_theta = theta
        if ui.cor_param1_input.text() == '':
            init = None
        else:
            try:
                init = float(ui.cor_param1_input.text())
            except:
                ui.statusbar.showMessage("Provide a valid initial guess value or no value")
                return
        if ui.cor_find_auto_mask_chk.isChecked():
            try:
                ratio = float(ui.cor_find_auto_mask_ratio.text())
            except:
                ui.statusbar.showMessage("Provide a valid mask ratio value or no value")
                return
        block_gui('Finding center with the default method', 0)
        COR = tomopy.find_center(tomo, theta=recon_theta, ind=ui.cor_tab_slider.value(), init=init, tol=float(ui.autofind_COR_tolerance_select.currentText()), mask=ui.cor_find_auto_mask_chk.isChecked(), ratio=ratio, sinogram_order=False)[0]
    if ui.cor_algorithm_select.currentText() == 'find_center_vo':
        if ui.cor_param1_input.text() == '':
            smin = None
            smax = None
        else:
            try:
                sminmax = int(ui.cor_param1_input.text())
                smax = abs(sminmax)
                smin = -smax
            except:
                ui.statusbar.showMessage("Provide a valid smin/smax value or no value")
                return
        if ui.cor_param2_input.text() == '':
            srad = None
        else:
            try:
                srad = float(ui.cor_param2_input.text())
            except:
                ui.statusbar.showMessage("Provide a valid srad value or no value")
                return
        if ui.cor_param3_input.text() == '':
            drop = None
        else:
            try:
                drop = int(ui.cor_param3_input.text())
            except:
                ui.statusbar.showMessage("Provide a valid drop value or no value")
                return
        if ui.cor_find_auto_mask_chk.isChecked():
            try:
                ratio = float(ui.cor_find_auto_mask_ratio.text())
            except:
                ui.statusbar.showMessage("Provide a valid object size ratio value or no value")
                return
        else:
            ratio = None
        block_gui('Finding COR with the Vo method...', 0)
        COR = tomopy.find_center_vo(tomo, ind=ui.cor_tab_slider.value(), smin=smin, smax=smax,  srad=srad, drop=drop, step=float(ui.autofind_COR_tolerance_select.currentText()), ratio = ratio, ncore = ncore)
    if ui.cor_algorithm_select.currentText() == 'find_center_pc':
        #find two projections 180 deg apart
        idx_0 = np.abs(theta).argmin()
        idx_180 = np.abs(np.array(theta) - 180).argmin()
        print('zero degree theta is proj#{0}, 180 degrees theta is proj#{1}'.format(idx_0, idx_180))
        proj1 = tomo[idx_0]
        proj2 = tomo[idx_180]
        if ui.cor_param1_input.text() == '':
            rot_guess = None
        else:
            try:
                rot_guess = float(ui.cor_param1_input.text())
            except:
                ui.statusbar.showMessage("Provide a valid rotation guess value or no value")
                unblock_gui()
                return
        block_gui('finding COR with PC method...', 0)
        COR = tomopy.find_center_pc(proj1, proj2, float(ui.autofind_COR_tolerance_select.currentText()), rot_guess)
    COR = np.round(COR, decimals=2)
    print('COR found at {}'.format(COR))
    ui.cor_value.setText(str(COR))
    tori['reconstruction_arguments']['recon_kwargs']['center'] = COR
    unblock_gui()
    cor_single_slice_recon()

def eval_theta_ranges():
    thmin = ui.cor_start_theta_input.text()
    thmax = ui.cor_end_theta_input.text()
    try:
        thmin = float(thmin)
        thmax = float(thmax)
        switch_to_calc_theta()
    except:
        return

def cor_single_slice_recon():
    global tomo, theta, tori
    block_gui('Reconstructing single slice...', 0)
    padsize = 0
    single_slice=np.empty(shape=[tomo.shape[0],1,tomo.shape[2]])
    single_slice[:,0,:] = tomo[:,ui.cor_tab_slider.value(),:]
    try:
        tori['reconstruction_arguments']['recon_kwargs']['center'] = float(ui.cor_value.text())
    except:
        ui.cor_value.setText(str(tori['reconstruction_arguments']['recon_kwargs']['center']))
    COR = tori['reconstruction_arguments']['recon_kwargs']['center']
    if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
        recon_theta = np.radians(theta)
    else:
        recon_theta = theta
    if ui.cor_tab_log_chk.isChecked():
        single_slice = -np.log(single_slice)
        single_slice = tomopy.misc.corr.remove_nan(single_slice, ncore=ncore)
    if ui.cor_tab_pad_chk.isChecked():
        padsize = tomo.shape[2]//4
        single_slice = tomopy.misc.morph.pad(single_slice, axis=2, npad=padsize, mode='edge')
        COR = COR + padsize
        #print(single_slice.shape)
    mask = theta_mask()
    test_slice = tomopy.recon(tomo=single_slice[mask], theta=recon_theta[mask], center=COR, algorithm='gridrec', filter_name='ramlak', sinogram_order=False)
    if ui.cor_tab_pad_chk.isChecked():
        test_slice = test_slice[:,padsize:-padsize,padsize:-padsize]
    image.setData(test_slice[0])
    unblock_gui()
    ui.silx_plot.resetZoom()

def cor_variance_sliders_update():
    cor_selection.crop_changed()
    display_cor_crop()
    calc_cor_variance()

def display_cor_crop():
    try:
        cor_range_slices.shape
    except:
        print('calculate range first')
        return
    ui.silx_cor_image_crop.getImage(legend='image').setData(cor_range_slices[ui.cor_tab_value_slider.value(),ui.cor_variance_bottom_slider.value():ui.cor_variance_top_slider.value(),ui.cor_variance_left_slider.value():ui.cor_variance_right_slider.value()])
    ui.silx_cor_image_crop.resetZoom()

def calc_cor_variance():
    global cor_range_slices, cor_range_values, cor_range_TV
    try:
        cor_range_slices.shape
    except:
        print('calculate range first')
        return
    cor_range_TV = []
    for c in range (0,len(cor_range_values)):
        cor_range_TV.append(totvar(cor_range_slices[c,ui.cor_variance_bottom_slider.value():ui.cor_variance_top_slider.value(),ui.cor_variance_left_slider.value():ui.cor_variance_right_slider.value()]))
    #now update the variance plot
    ui.silx_cor_plot_variance.getCurve('local variance').setData(x=cor_range_values, y=cor_range_TV)
    ui.silx_cor_plot_variance.getCurve('current_position').setData(x=[cor_range_values[ui.cor_tab_value_slider.value()]], y=[cor_range_TV[ui.cor_tab_value_slider.value()]])
    ui.silx_cor_plot_variance.resetZoom()

def cor_calculate_range():
    global tomo, theta, cor_range_slices, cor_range_values, tori, cor_range_TV, cor_range_STD
    try:
        cor_start = float(ui.cor_recon_range_start_inp.text())
        cor_end = float(ui.cor_recon_range_end_inp.text())
        cor_step = float(ui.cor_step_select.currentText())
    except:
        print("can't get start/end values")
        return
    cor_range_values = np.linspace(cor_start, cor_end, int((cor_end-cor_start)/ cor_step)+1)
    #calculate a "single" stripe dataset
    padsize = 0
    single_slice=np.empty(shape=[tomo.shape[0],len(cor_range_values),tomo.shape[2]])
    block_gui('wait for reconstructions with different center of rotation values...', 0)
    for i in range (0, len(cor_range_values)):
        single_slice[:,i,:] = tomo[:,ui.cor_tab_slider.value(),:]
    if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
        recon_theta = np.radians(theta)
    else:
        recon_theta = theta
    if ui.cor_tab_log_chk.isChecked():
        single_slice = -np.log(single_slice)
        single_slice = tomopy.misc.corr.remove_nan(single_slice)
    if ui.cor_tab_pad_chk.isChecked():
        padsize = tomo.shape[2]//4
        single_slice = tomopy.misc.morph.pad(single_slice, axis=2, npad=padsize, mode='edge')
        #print(single_slice.shape)
    #cor_range_slices = np.zeros(shape=[len(cor_range_values),tomo.shape[2],tomo.shape[2]])
    cor_range_TV = np.zeros(shape = len(cor_range_values))
    cor_range_STD = np.zeros(shape = len(cor_range_values))
    #now do the reconstruction
    block_gui('wait for reconstructions with different center of rotation values...', 5)
    #for c in range (0,len(cor_range_values)):
    #    block_gui('wait for reconstions with different center of rotation values...', 100*c/len(cor_range_values))
    #    print('reconstructing for COR = {}...'.format(cor_range_values[c]))
    mask = theta_mask()
    cor_range_slices = tomopy.recon(single_slice[mask], recon_theta[mask], center=cor_range_values+padsize, algorithm='gridrec', filter_name='ramlak', sinogram_order=False, ncore=ncore)
    if ui.cor_tab_pad_chk.isChecked():
        cor_range_slices = cor_range_slices[:,padsize:-padsize,padsize:-padsize]
    for c in range (0, len(cor_range_slices)):
        cor_range_TV[c] = totvar(cor_range_slices[c])
        cor_range_STD[c] = np.std(cor_range_slices[c])
    #print(cor_range_slices.shape)
    unblock_gui()
    ui.cor_tab_value_slider.setMaximum(cor_range_slices.shape[0]-1)
    ui.cor_tab_value_slider.setValue(int(cor_range_slices.shape[0]/2))
    cor_value_slider()

def cor_value_slider():
    global cor_range_slices, cor_range_values, cor_range_TV
    ui.cor_tab_value_value.setText(str(cor_range_values[ui.cor_tab_value_slider.value()]))
    image.setData(cor_range_slices[ui.cor_tab_value_slider.value()])
    try:
        local_var_curve.setData(x=cor_range_values, y=cor_range_TV)
        local_var_cursor.setData(x=[cor_range_values[ui.cor_tab_value_slider.value()]], y=[cor_range_TV[ui.cor_tab_value_slider.value()]])
        local_std_curve.setData(x=cor_range_values, y=cor_range_STD)
        local_std_cursor.setData(x=[cor_range_values[ui.cor_tab_value_slider.value()]], y=[cor_range_STD[ui.cor_tab_value_slider.value()]])
        ui.silx_cor_plot_variance.resetZoom()
        ui.silx_cor_plot_std.resetZoom()
    except:
        pass

def use_this_cor():
    global tori
    if np.array(cor_range_values).size > 0:
        ui.cor_value.setText(str(cor_range_values[ui.cor_tab_value_slider.value()]))
    tori['reconstruction_arguments']['recon_kwargs']['center'] = float(ui.cor_value.text())

def cor_is_given():
    global tori, tomo
    try:
        tori['reconstruction_arguments']['recon_kwargs']['center'] = float(ui.cor_value.text())
    except:
        print(f'cannot convert input {ui.cor_value.text()} to float')
        if tori['reconstruction_arguments']['recon_kwargs']['center'] == None:
            ui.cor_value.setText(str(tomo.shape[2]/2))
        else:
            ui.cor_value.setText(str(tori['reconstruction_arguments']['recon_kwargs']['center']))


def recalc_TV_STD():
    if len(cor_range_slices) == 0:
        return
    bounds = image.getVisibleBounds()
    #print(f'visible bounds = {bounds}')
    for c in range (0,len(cor_range_slices)):
        block_gui('recalculating local variance and std...', 100*c/len(cor_range_values))
        cor_range_TV[c] = totvar(cor_range_slices[c,int(bounds[0]):int(bounds[1]),int(bounds[2]):int(bounds[3])])
        cor_range_STD[c] = np.std(cor_range_slices[c,int(bounds[0]):int(bounds[1]),int(bounds[2]):int(bounds[3])])
    unblock_gui()
    cor_value_slider()

def energy_entered():
    try:
        ui.phase_energy_inp.setText('{:.2f}'.format(abs(float(ui.phase_energy_inp.text()))))
    except:
        #not a good input
        ui.phase_energy_inp.setText('{:.2f}'.format(tori['correction_arguments']['phase_retrieval_kwargs']['energy']))
        return
    tori['correction_arguments']['phase_retrieval_kwargs']['energy'] = float(ui.phase_energy_inp.text())

def postp_crop_checked():
    if ui.postp_crop_chk.isChecked():
        postp_crop_selection.show()
    else:
        postp_crop_selection.hide()

def pixel_size_entered():
    #parse input
    try:
        float(ui.phase_pixel_size.text())
    except:
        pixel_unit_changed()
        #not a good input
        return
    if ui.phase_pixel_unit_select.currentText() == 'microns':
        tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = float(ui.phase_pixel_size.text())*1e-4 #in cm
    if ui.phase_pixel_unit_select.currentText() == 'mm':
        tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = float(ui.phase_pixel_size.text())*0.1 #in cm
    if ui.phase_pixel_unit_select.currentText() == 'cm':
        tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = float(ui.phase_pixel_size.text()) #in cm
    tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'] = np.round(tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'], decimals = 7) #precision down to 1 nm
    print(f"current pixel size cm): {tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size']}")
    pixel_unit_changed()
    scalebar.scale = tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size']*10
    #scalebar.default_scale()

def sdd_entered():
    try:
        float(ui.phase_sdd.text())
    except:
        sdd_unit_changed()
        return
    if ui.phase_sdd_unit_select.currentText() == 'mm':
        tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = float(ui.phase_sdd.text())*0.1 #in cm
    if ui.phase_sdd_unit_select.currentText() == 'cm':
        tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = float(ui.phase_sdd.text()) #in cm
    tori['correction_arguments']['phase_retrieval_kwargs']['dist'] = np.round(tori['correction_arguments']['phase_retrieval_kwargs']['dist'], decimals = 2) #precision down to 0.1 mm
    sdd_unit_changed()

def pixel_unit_changed():
    if ui.phase_pixel_unit_select.currentText() == 'microns':
        ui.phase_pixel_size.setText(str(np.round(tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size']*1e4, decimals = 3)))
    if ui.phase_pixel_unit_select.currentText() == 'mm':
        ui.phase_pixel_size.setText(str(np.round(tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size']*10, decimals = 6)))
    if ui.phase_pixel_unit_select.currentText() == 'cm':
        ui.phase_pixel_size.setText(str(tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size']))

def sdd_unit_changed():
    if ui.phase_sdd_unit_select.currentText() == 'mm':
        ui.phase_sdd.setText(str(tori['correction_arguments']['phase_retrieval_kwargs']['dist']*10))
    if ui.phase_sdd_unit_select.currentText() == 'cm':
        ui.phase_sdd.setText(str(tori['correction_arguments']['phase_retrieval_kwargs']['dist']))

def phase_slider_changed():
    ui.phase_index.setText(str(ui.phase_slider.value()))
    if ui.phase_autoupd_chk.isChecked():
        single_phase_retrieve()

def phase_index_changed():
    try:
        ui.phase_proj_slider.setValue(int(ui.phase_proj_index.text()))
        phase_proj_slider_changed()
    except:
        phase_proj_slider_changed()


def deltabeta_changed():
    try:
        float(ui.delta_beta_value.text())
    except:
        #
        deltabeta_switched()
        return
    if ui.phase_deltabeta_select.currentText() == 'delta/beta':
        tori['correction_arguments']['phase_retrieval_kwargs']['alpha'] = 1/float(ui.delta_beta_value.text())
    else:
        tori['correction_arguments']['phase_retrieval_kwargs']['alpha'] = float(ui.delta_beta_value.text())
    tori['correction_arguments']['phase_retrieval_kwargs']['alpha'] = np.round(tori['correction_arguments']['phase_retrieval_kwargs']['alpha'], decimals = 6)
    deltabeta_switched()
    if ui.phase_autoupd_chk.isChecked():
        single_phase_retrieve()

def deltabeta_switched():
    if ui.phase_deltabeta_select.currentText() == 'delta/beta':
        ui.delta_beta_value.setText('{:.2f}'.format(1/tori['correction_arguments']['phase_retrieval_kwargs']['alpha']))
    if ui.phase_deltabeta_select.currentText() == 'beta/delta':
        ui.delta_beta_value.setText('{:.6f}'.format(tori['correction_arguments']['phase_retrieval_kwargs']['alpha']))

def single_phase_retrieve():
    global tomo, tori
    try:
        tomo.shape
    except:
        ui.statusbar.showMessage("No data loaded yet")
        return
    if len(tomo) == 0:
        ui.statusbar.showMessage("No data loaded yet")
        return
    if ui.phase_proj_slice_select.currentText() == 'projection':#retrieve phase of a single projection
        block_gui('wait for single projection phase retrieval', 0)
        single_proj=np.empty(shape=[1,tomo.shape[1],tomo.shape[2]])
        single_proj[0,:,:] = tomo[ui.phase_slider.value(),:,:]
        retrieved_proj = retrieve_phase_of_proj_stack(single_proj, tori['correction_arguments']['phase_retrieval_kwargs']['alpha'], 1, "")
        image.setData(retrieved_proj[0])
    if ui.phase_proj_slice_select.currentText() == 'slice':#retrieve phase of a single projection
        block_gui('reconstructing single slice...', 0)
        phase_range_slices = tomo[:,ui.phase_slider.value()-5:ui.phase_slider.value()+5,:]
        phase_range_slices = retrieve_phase_of_proj_stack(phase_range_slices, tori['correction_arguments']['phase_retrieval_kwargs']['alpha'], 5, "")
        single_sino=np.empty(shape=[tomo.shape[0],1,tomo.shape[2]])
        single_sino[:,0,:] = phase_range_slices[:,4,:]
        if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
            recon_theta = np.radians(theta)
        else:
            recon_theta = theta
        if ui.phase_log_chk.isChecked():
            single_sino = -np.log(single_sino)
            single_sino = tomopy.misc.corr.remove_nan(single_sino, ncore=ncore)
        COR = tori['reconstruction_arguments']['recon_kwargs']['center']
        if ui.phase_pad_chk.isChecked():
            padsize = single_sino.shape[2]//4
            single_sino = tomopy.misc.morph.pad(single_sino, axis=2, npad=padsize, mode='edge')
            COR = COR + padsize
        mask = theta_mask()
        test_slice = tomopy.recon(single_sino[mask], recon_theta[mask], COR, algorithm='gridrec', filter_name='ramlak', sinogram_order=False)
        if ui.phase_pad_chk.isChecked():
            test_slice = test_slice[:,padsize:-padsize,padsize:-padsize]
        image.setData(test_slice[0])
    unblock_gui()
    ui.silx_plot.resetZoom()

def retrieve_phase_of_proj_stack(projs, alpha, nchunk=20, text=""):
    #all validity checks have been done
    global tori
    print(f'going to retrieve phase of a stack with dimensions: {projs.shape}')
    print(f'with alpha value = {alpha}')
    chunksize = projs.shape[0]/nchunk
    print(f'chunk size = {chunksize}')
    args = {}
    args = {'pixel_size': tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'],
            'dist': tori['correction_arguments']['phase_retrieval_kwargs']['dist'],
            'energy': tori['correction_arguments']['phase_retrieval_kwargs']['energy'],
            'alpha': alpha,
            'pad': ui.phase_pad_chk.isChecked(),
            'ncore': ncore}
    if chunksize < 10: #retrieve in one go
        block_gui(f'wait for phase retrieval {text}...', 0)
        retrieved_proj = tomopy.retrieve_phase(projs, **args)
    else: #retrieve in chunks
        retrieved_proj = np.zeros(shape = projs.shape, dtype = 'float32')
        for i in range (0,nchunk):
            block_gui(f'wait for phase retrieval {text}...', 100*int(i)/nchunk)
            my_phase_slice = slice(int(np.round(i*chunksize)), int(np.round((i+1)*chunksize)))
            currentchunk = copy.deepcopy(projs[int(np.round(i*chunksize)):int(np.round((i+1)*chunksize)),:,:])
            retrieved_proj[my_phase_slice,:,:] = tomopy.retrieve_phase(currentchunk, **args)
    return retrieved_proj

def retrieve_phase_range():
    block_gui('wait for retrieval for a range of alpha values', 0)
    global tomo, phase_range_values, phase_range_projections, phase_range_slices
    #check the ranges
    phase_range_values = np.array(np.linspace(float(ui.phase_range_from.text()), float(ui.phase_range_to.text()), (int(ui.phase_range_intervals.text())+1)))
    ui.phase_image_refraction_label.setText(ui.phase_deltabeta_select.currentText())
    phase_range_projections = np.zeros(shape=[phase_range_values.shape[0],tomo.shape[1],tomo.shape[2]])
    phase_range_slices = np.zeros(shape=[phase_range_values.shape[0],tomo.shape[2],tomo.shape[2]])
    single_proj=np.empty(shape=[1,tomo.shape[1],tomo.shape[2]])
    single_proj[0,:,:] = tomo[ui.phase_slider.value(),:,:]
    single_slice = tomo[:,max(ui.phase_slider.value()-10, 0):ui.phase_slider.value()+10,:]
    if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
        recon_theta = np.radians(theta)
    else:
        recon_theta = theta
    for p in range (0,len(phase_range_values)):
        if ui.phase_image_refraction_label.text() == 'delta/beta':
            alpha = 1/phase_range_values[p]
        if ui.phase_image_refraction_label.text() == 'beta/delta':
            alpha = phase_range_values[p]
        phase_range_projections[p] = tomopy.retrieve_phase(single_proj,
                                               pixel_size = tori['correction_arguments']['phase_retrieval_kwargs']['pixel_size'],
                                               dist = tori['correction_arguments']['phase_retrieval_kwargs']['dist'],
                                               energy = tori['correction_arguments']['phase_retrieval_kwargs']['energy'],
                                               alpha = alpha, pad=ui.phase_pad_chk.isChecked(),
                                               ncore=ncore)
        vol_for_slices = retrieve_phase_of_proj_stack(single_slice, alpha, 5, f"phase value {p+1}/{len(phase_range_values)}")
        single_sino=np.empty(shape=[tomo.shape[0],1,tomo.shape[2]])
        single_sino[:,0,:] = vol_for_slices[:,10 + min(0, ui.phase_slider.value()-10),:]
        COR = tori['reconstruction_arguments']['recon_kwargs']['center']
        if ui.phase_pad_chk.isChecked():
            padsize = single_sino.shape[2]//4
            COR = COR+padsize
            single_sino = tomopy.misc.morph.pad(single_sino, axis=2, npad=padsize, mode='edge')
        if ui.phase_log_chk.isChecked():
            single_sino = -np.log(single_sino)
            single_sino = tomopy.misc.corr.remove_nan(single_sino, ncore=ncore)
        mask = theta_mask()
        test_slice = tomopy.recon(single_sino[mask], recon_theta[mask], COR, algorithm='gridrec', filter_name='ramlak', sinogram_order=False)
        if ui.phase_pad_chk.isChecked():
            test_slice = test_slice[:,padsize:-padsize,padsize:-padsize]
        phase_range_slices[p] = test_slice[0]
    ui.phase_image_refraction_slider.setMaximum(phase_range_values.shape[0]-1)
    ui.phase_image_refraction_slider.setValue(phase_range_values.shape[0]//2)
    phase_range_slider_moved()
    unblock_gui()

def phase_range_slider_moved():
    if phase_range_projections.shape[0] == 0:
        return
    slider_index = ui.phase_image_refraction_slider.value()
    if ui.phase_proj_slice_select.currentText() == 'projection':
        image.setData(phase_range_projections[slider_index])
    if ui.phase_proj_slice_select.currentText() == 'slice':
        image.setData(phase_range_slices[slider_index])
    ui.phase_image_refraction_value.setText(str(phase_range_values[slider_index]))

def use_this_phase():
    '''sets a current phase parameter as the one to use
    '''
    ui.delta_beta_value.setText(str(phase_range_values[ui.phase_image_refraction_slider.value()]))
    deltabeta_changed()
        
def full_data_phase_retrieval():
    global tomo
    tori['correction_arguments']['phase_retrieval'] = True
    tori['correction_arguments']['phase_retrieval_kwargs']['pad'] = ui.phase_pad_chk.isChecked()
    block_gui('Wait for full data phase retrieval...', 0)
    tomo = retrieve_phase_of_proj_stack(tomo, tori['correction_arguments']['phase_retrieval_kwargs']['alpha'], 20, "")
    unblock_gui()
    ui.phase_retrieve_btn.setEnabled(False) #don't try to retrieve phase twice! need to reload data

def reconstruct_all():
    global tomo, theta, recon, tori
    if ui.recon_filter_select.isVisible():
        tori['reconstruction_arguments']['recon_kwargs']['algorithm'] = ui.recon_algorithm_select.currentText()
        tori['reconstruction_arguments']['recon_kwargs']['filter_name'] = ui.recon_filter_select.currentText()
        if ui.recon_filter_par_list.text() != '':
            tori['reconstruction_arguments']['recon_kwargs']['filter_par'] = [float(i) for i in list(ui.recon_filter_par_list.text().split(','))]
    else:
        try:
            tori['reconstruction_arguments']['recon_kwargs'].pop('filter_name')
            tori['reconstruction_arguments']['recon_kwargs'].pop('filter_par')
        except:
            pass
    tori['reconstruction_arguments']['pad'] = ui.recon_pad_chk.isChecked()
    block_gui('Wait for data reconstion...', 0)
    if np.max(theta) - np.min(theta) > 10: #probably theta is in degrees
        recon_theta = np.radians(theta)
    else:
        recon_theta = theta
    #make an empty recon array with the unpadded shape
    recon = np.empty(shape = (0, tomo.shape[2], tomo.shape[2]), dtype = 'float32')
    recon_slice = slice(ui.recon_first_slice_slider.value(),ui.recon_last_slice_slider.value()+1)
    if debug:
        print(f'recon slice = {recon_slice}')
    nchunk = 10
    chunksize = (ui.recon_last_slice_slider.value()+1-ui.recon_first_slice_slider.value())/nchunk
    if chunksize < 10: #less then 100 slices - do in a single go without progerss bar updates
        nchunk = 1
        chunksize = ui.recon_last_slice_slider.value()+1-ui.recon_first_slice_slider.value()
    print(f'performing {nchunk} chunks of {chunksize} slices')
    mask = theta_mask()
    #tomopy.recon arguments as a dictionary
    recargs = {'tomo': None,
            'theta': recon_theta[mask],
            'sinogram_order': False,
            'algorithm': tori['reconstruction_arguments']['recon_kwargs']['algorithm'],
            'ncore': ncore
            }
    for i in range(0,7):
        if ui.recon_param_labels[i].isVisible():
            if ui.recon_param_inputs[i].text() != '':
                if 'int,' in ui.recon_param_inputs[i].toolTip():
                    myvalue = int(ui.recon_param_inputs[i].text())
                elif 'float,' in ui.recon_param_inputs[i].toolTip():
                    myvalue = float(ui.recon_param_inputs[i].text())
                else:
                    myvalue = [float(i) for i in list(ui.recon_param_inputs[i].text().split(','))]
                recargs[ui.recon_param_labels[i].text()] = myvalue
    if ui.recon_filter_select.isVisible():
        recargs['filter_name'] = ui.recon_filter_select.currentText()
        if ui.recon_filter_par_list.text() != '':
            recargs['filter_par'] = [float(i) for i in list(ui.recon_filter_par_list.text().split(','))]
    for i in range (0,nchunk):
        block_gui('wait for reconstruction...', 100*int(i)/nchunk)
        padsize = 0
        myVslice = slice(int(np.round(i*chunksize)+ui.recon_first_slice_slider.value()), int(np.round((i+1)*chunksize))+ui.recon_first_slice_slider.value())
        print(f'my V slice for recon = {myVslice}')
        #add padding if needed: add padding to the whole tomo set
        if ui.recon_pad_chk.isChecked():
            padsize = tomo.shape[2]//4
            partial_tomo = tomopy.misc.morph.pad(tomo[:,myVslice,:], axis=2, npad=padsize, mode='edge')
            recargs['tomo'] = partial_tomo[mask]
        else:
            partial_tomo = tomo[:,myVslice,:]
            padsize = 0
        recargs['tomo'] = partial_tomo[mask]
        recargs['center'] = tori['reconstruction_arguments']['recon_kwargs']['center']+padsize
        if debug:
            print(f'reconstruction arguments: {recargs}')
        partial_recon = tomopy.recon(**recargs)
        if ui.recon_pad_chk.isChecked(): #unpad results
            partial_recon = partial_recon[:,padsize:-padsize,padsize:-padsize]
        recon = np.append(recon, partial_recon, axis = 0)
    ui.plot_selector.setCurrentIndex(0)
    image_source_changed()
    ui.silx_plot.resetZoom()
    unblock_gui()

def draw_a_circular_mask():
    x = []
    y = []
    ui.silx_plot.getCurve('circle').setData(x=[], y=[])
    ui.silx_plot.getCurve('line1').setData(x=[], y=[])
    ui.silx_plot.getCurve('line2').setData(x=[], y=[])
    if ui.postp_circ_mask_chk.isChecked() == False:
        return
    try:
        radius = 0.5*recon.shape[1]*ui.postp_circ_mask_spin.value()
    except:
        print('cant create a circular mask')
        return
    if ui.plot_selector.currentText() == 'XY':
        for i in range(0, 361):
            x.append(np.sin(np.radians(i))*radius + (recon.shape[1]/2))
            y.append(np.cos(np.radians(i))*radius + (recon.shape[1]/2))
        ui.silx_plot.getCurve('circle').setData(x=x, y=y)
    else:
        try:
            dist_from_cen = np.sqrt(np.square(radius) - np.square(ui.plot_slider.value() - (recon.shape[1]/2)))
        except Exception as error:
            print(error)
            return
        ui.silx_plot.getCurve('line1').setData(x=[(recon.shape[1]/2-dist_from_cen),(recon.shape[1]/2-dist_from_cen)], y=[0, recon.shape[0]])
        ui.silx_plot.getCurve('line2').setData(x=[(recon.shape[1]/2+dist_from_cen),(recon.shape[1]/2+dist_from_cen)], y=[0, recon.shape[0]])

def recon_display_select_changed():
    if ui.recon_display_select.currentText() == 'XY':
        ui.recon_display_slide.setMaximum(recon.shape[0]-1)
        ui.recon_display_slide.setValue(recon.shape[0]//2)
    if ui.recon_display_select.currentText() == 'XZ' or ui.recon_display_select.currentText() == 'YZ':
        ui.recon_display_slide.setMaximum(recon.shape[1]-1)
        ui.recon_display_slide.setValue(recon.shape[1]//2)
    recon_display_slide_changed()
    draw_a_circular_mask()

def recon_display_slide_changed():
    global recon
    ui.recon_display_index.setText(str(ui.recon_display_slide.value()))
    if ui.recon_display_select.currentText() == 'XY':
        ui.silx_recon.getImage(legend='image').setData(recon[ui.recon_display_slide.value(),:,:])
    if ui.recon_display_select.currentText() == 'XZ':
        ui.silx_recon.getImage(legend='image').setData(recon[:,ui.recon_display_slide.value(),:])
    if ui.recon_display_select.currentText() == 'YZ':
        ui.silx_recon.getImage(legend='image').setData(recon[:,:,ui.recon_display_slide.value()])
    draw_a_circular_mask()

def preview_circ_mask():
    global recon
    try:
        recon.shape
    except:
        print('do the reconstruction first')
        return
    if ui.recon_display_select.currentText() == 'XY':
        testvol = np.empty(shape=(2,recon.shape[1],recon.shape[1]))
        testvol[0,:,:] = recon[ui.recon_display_slide.value(),:,:]
        testvol[1,:,:] = recon[ui.recon_display_slide.value(),:,:]
        testvol = tomopy.circ_mask(testvol, axis=0, ratio=float(ui.recon_circ_mask_value.text()))
        ui.silx_recon.getImage(legend='image').setData(testvol[0,:,:])
    if ui.recon_display_select.currentText() == 'XZ':
        testimage = recon[:,ui.recon_display_slide.value(),:]
        try:
            radius = 0.5*recon.shape[1]*float(ui.recon_circ_mask_value.text())
            dist_from_cen = np.sqrt(np.square(radius) - np.square(ui.recon_display_slide.value() - (recon.shape[1]/2)))
        except:
            return
        testimage[:,0:int((recon.shape[1]/2-dist_from_cen))] = 0
        testimage[:,int((recon.shape[1]/2+dist_from_cen)):-1] = 0
        ui.silx_recon.getImage(legend='image').setData(testimage)
    if ui.recon_display_select.currentText() == 'YZ':
        testimage = recon[:,:,ui.recon_display_slide.value()]
        try:
            radius = 0.5*recon.shape[1]*float(ui.recon_circ_mask_value.text())
            dist_from_cen = np.sqrt(np.square(radius) - np.square(ui.recon_display_slide.value() - (recon.shape[1]/2)))
        except:
            return
        testimage[:,0:int((recon.shape[1]/2-dist_from_cen))] = 0
        testimage[:,int((recon.shape[1]/2+dist_from_cen)):-1] = 0
        ui.silx_recon.getImage(legend='image').setData(testimage)
        

def recon_display_index_changed():
    try:
        ui.recon_display_slide.setValue(int(ui.recon_display_index.text()))
    except ValueError:
        pass
    recon_display_select_changed()

def next_tab():
    ui.controls_tabs.setCurrentIndex(ui.controls_tabs.currentIndex()+1)

def apply_minus_log(autoload):
    '''-log() convertion of the tomo data, could be applied once
    '''
    global tomo, tori
    block_gui('wait for log conversion...', 0)
    tomo = -np.log(tomo)
    block_gui('removing NaNs...', 0)
    tomo = tomopy.misc.corr.remove_nan(tomo, ncore=ncore)
    ui.apply_minus_log_btn.setEnabled(False)
    ui.apply_minus_log_btn2.setEnabled(False)
    ui.plot_tab_displaylog_chk.setChecked(False)
    ui.plot_tab_displaylog_chk.setEnabled(False)
    ui.cor_tab_log_chk.setChecked(False)
    ui.cor_tab_log_chk.setEnabled(False)
    ui.phase_log_chk.setChecked(False)
    ui.phase_log_chk.setEnabled(False)
    if not autoload:
        if tori['correction_arguments']['phase_retrieval']:
            print('phase is allready done')
            tori['correction_arguments']['minus_log_before_phase'] = False
            tori['correction_arguments']['minus_log_after_phase'] = True
        else:#phase was retrieved before
            print('no phase (yet?)')
            tori['correction_arguments']['minus_log_before_phase'] = True
            tori['correction_arguments']['minus_log_after_phase'] = False
        #we assume minus log is applied only once
    unblock_gui()

def recon_first_slice_slider_changed():
    if ui.recon_first_slice_slider.value() >= (ui.recon_last_slice_slider.value()):
        ui.recon_first_slice_slider.setValue(ui.recon_last_slice_slider.value()) 
    ui.recon_first_slice_index.setText(str(ui.recon_first_slice_slider.value()))
    
    
def recon_last_slice_slider_changed():
    if ui.recon_last_slice_slider.value() <= (ui.recon_first_slice_slider.value()):
        ui.recon_last_slice_slider.setValue(ui.recon_first_slice_slider.value())
    ui.recon_last_slice_index.setText(str(ui.recon_last_slice_slider.value()))

def recon_first_slice_index_changed():
    try:
        ui.recon_first_slice_slider.setValue(int(ui.recon_first_slice_index.text()))
    except:
        recon_first_slice_slider_changed() 

def recon_last_slice_index_changed():
    try:
        ui.recon_last_slice_slider.setValue(int(ui.recon_last_slice_index.text()))
    except:
        recon_last_slice_slider_changed() 

def recon_algorithm_selected():
#    global ark
    try:
        aakw = ark[ui.recon_algorithm_select.currentText()].copy()
    except:
        print('no allowed recon kwargs found for the selected algorithm...')
        return 
    fn = 'filter_name' in aakw
    ui.recon_label_04.setVisible(fn)
    ui.recon_filter_select.setVisible(fn)
    ui.recon_label_06.setVisible(fn)
    ui.recon_filter_par_list.setVisible(fn)
    ui.recon_try_filters_btn.setVisible(fn)
    if fn:
        aakw.remove('filter_name')
        aakw.remove('filter_par')
    if len(aakw) < 2:
        ui.controls_tabs.setMaximumHeight(266)
    else:
        ui.controls_tabs.setMaximumHeight(266+29*(len(aakw)-1))
    for i in range (0,7):
        if i < len(aakw):
            ui.recon_param_labels[i].setVisible(True)
            ui.recon_param_labels[i].setText(aakw[i])
            ui.recon_param_inputs[i].setVisible(True)
            ui.recon_param_inputs[i].setToolTip(get_description_for_param(tomopy.recon, ui.recon_param_labels[i].text()))
        else:
            ui.recon_param_labels[i].setVisible(False)
            ui.recon_param_inputs[i].setVisible(False)
    return

def apply_circ_mask():
    global recon, tori
    if recon.shape == 0:
        print('do the reconstruction first')
        return
    try:
        circm_value = float(ui.postp_circ_mask_value.text())
    except:
        ui.statusbar.showMessage('provide a value to fill outside the circular mask', 5000)
        return
    try:
        block_gui('applying circular mask - d={ui.postp_circ_mask_spin.value()}, v={circm_value}...', 0)
        recon=tomopy.circ_mask(recon, axis=0, ratio=ui.postp_circ_mask_spin.value(), val = circm_value)
        tori['post_recon_process_arguments']['circ_mask'] = True
        tori['post_recon_process_arguments']['circ_mask_kwargs']['ratio'] = ui.postp_circ_mask_spin.value()
        tori['post_recon_process_arguments']['circ_mask_kwargs']['val'] = circm_value
        unblock_gui()
    except Exception as error:
        tori['post_recon_process_arguments']['circ_mask'] = False
        print(error)
        unblock_gui()
        return
    ui.postp_circ_mask_chk.setChecked(False)
    plot_slider_update()

def save_results():
    #ui.postp_binning_chk.isChecked()
    #ui.postp_binning_spin.value() - bining factor
    block_gui('Wait for saving data...', 0)
    global tori, recon
    save_file = ui.postp_save_path.text()
    save_folder = os.path.dirname(save_file)
    if not os.path.isdir(save_folder):
        print(f'making directories to: {save_folder}')
        os.makedirs(save_folder)
    try:
        hf = h5py.File(save_file, 'w')
        hf.create_dataset(ui.postp_dataset_name.text(), data=recon)
    except Exception as error:
        print('something went wrong with saving the file')
        print(error)
        unblock_gui()
        return
    if ui.postp_binning_chk.isChecked():#do downsampling - set the right tori entries
        tori['post_recon_process_arguments']['downscale'] = True
        tori['post_recon_process_arguments']['downscale_kwargs']['down_scale_factors'] = [ui.postp_binning_spin.value()]
    savedict = copy.deepcopy(tori)
    while any(isinstance(savedict[x], dict) for x in savedict):
        savedict = flatten_one_level(savedict)
    for key in savedict.keys():
        if savedict[key] != None:
            hf.create_dataset(f'tori/{key}', data = savedict[key])
    hf.close()
    print('...full-res file saved')
    if ui.postp_binning_chk.isChecked():#do downsampling
        block_gui('downsampling data...', 0)
        import cv2
        factor = ui.postp_binning_spin.value()
        downscaled_cv = []
        for i in recon:
            downscaled_i = cv2.resize(i, (0,0), fx=(1/factor), fy=(1/factor))
            downscaled_cv.append(downscaled_i)
        downscaled_cv = np.array(downscaled_cv)
        binned_dataset = np.empty(shape = [downscaled_cv.shape[0]//factor, downscaled_cv.shape[1], downscaled_cv.shape[2]])
        block_gui('downsampling data in z...', 0)
        for i in range(0, downscaled_cv.shape[1]):
            binned_dataset[:,i,:] = cv2.resize(downscaled_cv[:,i,:], (0,0), fy=(1/factor), fx=1)
        block_gui('saving binned dataset...', 0)
        hf = h5py.File(save_file.replace('.h5', f'bin_{factor}x{factor}x{factor}.h5'), 'w')
        hf.create_dataset(f'{ui.postp_dataset_name.text()}_{factor}x{factor}x{factor}',data=binned_dataset)
        for key in savedict.keys():
            if savedict[key] != None:
                hf.create_dataset(f'tori/{key}', data = savedict[key])
        hf.close()
        print('binned file saved')
    unblock_gui()

#filter dialog window staff:
def make_silx_plot():
    dui.silx = PlotWindow(FilterDialog, mask=False, roi=False, logScale=False, autoScale=False, grid=False, curveStyle=False, position=True)
    dui.silx.setObjectName("silx")
    dui.silx.setKeepDataAspectRatio()
    dui.silx.setAxesDisplayed(False)
    dui.silx.addImage(np.zeros(shape=[500,500]), legend='image')
    dui.silx_layout.addWidget(dui.silx)
    dui.silx.addCurve(x=[], y=[], color='red', legend='crop', selectable=False)

def new_dataset():
    global tori
    try:
        ds = h5py.File(tori['file_definitions']['data_file'], 'r')[dui.dataset_combo.currentText()]
    except:
        return

    dui.proj_range_0_spin.setMaximum(ds.shape[0]-1)
    dui.proj_range_1_spin.setMaximum(ds.shape[0])

    dui.proj_range_0_spin.setValue(tori['file_definitions']['slice_proj'][0])
    if tori['file_definitions']['slice_proj'][1] == -1:
        dui.proj_range_1_spin.setValue(ds.shape[0])
    else:
        dui.proj_range_1_spin.setValue(np.min([tori['file_definitions']['slice_proj'][1],ds.shape[0]]))
    dui.proj_downsample_spin.setValue(tori['file_definitions']['slice_proj'][2])

    dui.ver_range_0_spin.setMaximum(ds.shape[1]-1)
    dui.ver_range_1_spin.setMaximum(ds.shape[1])

    dui.ver_range_0_spin.setValue(tori['file_definitions']['slice_ver'][0])
    if tori['file_definitions']['slice_ver'][1] == -1:
        dui.ver_range_1_spin.setValue(ds.shape[1])
    else:
        dui.ver_range_1_spin.setValue(np.min([tori['file_definitions']['slice_ver'][1],ds.shape[1]]))
    dui.ver_downsample_spin.setValue(tori['file_definitions']['slice_ver'][2])

    dui.hor_range_0_spin.setMaximum(ds.shape[2]-1)
    dui.hor_range_1_spin.setMaximum(ds.shape[2])

    dui.hor_range_0_spin.setValue(tori['file_definitions']['slice_hor'][0])
    if tori['file_definitions']['slice_hor'][1] == -1:
        dui.hor_range_1_spin.setValue(ds.shape[2])
    else:
        dui.hor_range_1_spin.setValue(np.min([tori['file_definitions']['slice_hor'][1],ds.shape[2]]))
    dui.hor_downsample_spin.setValue(tori['file_definitions']['slice_hor'][2])

    single_image = ds[0]
    dui.silx.getImage(legend='image').setData(single_image)
    dui.silx.resetZoom()
    recalc()

def recalc():
    '''Recalculates data size estimation in the pre-load filter dialog window
    '''
    global tori
    try:
        ds = h5py.File(tori['file_definitions']['data_file'], 'r')[dui.dataset_combo.currentText()]
    except:
        return
    dui.nproj_label.setText(str(ds.shape[0]))
    dui.ver_label.setText(str(ds.shape[1]))
    dui.horizontal_label.setText(str(ds.shape[2]))
    dui.full_ds_label.setText('Full dataset: {0}x{1}x{2} = {3:.1f} Gb   (you have {4:.1f} Gb available)'.format(ds.shape[0], ds.shape[1], ds.shape[2], ds.nbytes*1e-9, psutil.virtual_memory()[1]*1e-9))
    a = int((dui.proj_range_1_spin.value()-dui.proj_range_0_spin.value())/dui.proj_downsample_spin.value())
    b = int((dui.ver_range_1_spin.value()-dui.ver_range_0_spin.value())/dui.ver_downsample_spin.value())
    c = int((dui.hor_range_1_spin.value()-dui.hor_range_0_spin.value())/dui.hor_downsample_spin.value())
    d = a*b*c*int(ds.nbytes/ds.size)*1e-9
    dui.red_ds_label.setText('Filtered dataset: {0}x{1}x{2} = {3:.1f} Gb'.format(a,b,c,d))
    dui.silx.getCurve('crop').setData(x=[dui.hor_range_0_spin.value(),
                                         dui.hor_range_1_spin.value(),
                                         dui.hor_range_1_spin.value(),
                                         dui.hor_range_0_spin.value(),
                                         dui.hor_range_0_spin.value()], y=[dui.ver_range_0_spin.value(),
                                                                           dui.ver_range_0_spin.value(),
                                                                           dui.ver_range_1_spin.value(),
                                                                           dui.ver_range_1_spin.value(),
                                                                           dui.ver_range_0_spin.value(),])

def filter_center_10(): #sets 10 cental vertical slices in the pre-load filter dialog window
    global tori
    ds = h5py.File(tori['file_definitions']['data_file'], 'r')[dui.dataset_combo.currentText()]
    dui.ver_range_0_spin.setValue(ds.shape[1]//2 - 5)
    dui.ver_range_1_spin.setValue(ds.shape[1]//2 + 5)
    recalc()

def filter_center_50(): #sets 50 cental vertical slices in the pre-load filter dialog window
    global tori
    ds = h5py.File(tori['file_definitions']['data_file'], 'r')[dui.dataset_combo.currentText()]
    dui.ver_range_0_spin.setValue(ds.shape[1]//2 - 25)
    dui.ver_range_1_spin.setValue(ds.shape[1]//2 + 25)
    recalc()

def filter_center_100(): #sets 100 cental vertical slices in the pre-load filter dialog window
    global tori
    ds = h5py.File(tori['file_definitions']['data_file'], 'r')[dui.dataset_combo.currentText()]
    dui.ver_range_0_spin.setValue(ds.shape[1]//2 - 50)
    dui.ver_range_1_spin.setValue(ds.shape[1]//2 + 50)
    recalc()

def filter_center_all(): #sets full vertical range in the pre-load filter dialog window
    global tori
    ds = h5py.File(tori['file_definitions']['data_file'], 'r')[dui.dataset_combo.currentText()]
    dui.ver_range_0_spin.setValue(0)
    dui.ver_range_1_spin.setValue(ds.shape[1])
    recalc()

def remove_nans():
    global tomo, ncore
    block_gui('removing NaNs from the projections...', 0)
    tomo = tomopy.misc.corr.remove_nan(tomo, ncore=ncore)
    unblock_gui()

def conv_360_180():
    global tomo, theta, tori
    CoR = float(ui.cor_value.text())
    block_gui('converting 360 deg off-axis scan to 180 deg...', 0)
    if CoR > tomo[0].shape[1]//2:
        direction = 'right'
        overlap = 2*int(tomo[0].shape[1] - CoR)
    else:
        direction = 'left'
        overlap = 2*int(CoR)
    print('direction = {}, overlap = {} px'.format(direction, overlap))
    tomo = tomopy.misc.morph.sino_360_to_180(tomo,overlap=overlap,rotation=direction)
    reset_interface()
    theta = tomopy.angles(tomo.shape[0]+1)[:-1]
    ui.cor_start_theta_input.setText("0")
    ui.cor_end_theta_input.setText("{}".format(180-(180/tomo.shape[0])))
    eval_theta_ranges()
    tori['reconstruction_arguments']['sino_360_to_180_old_center'] = CoR
    ui.cor_value.setText('{}'.format(tomo.shape[2]/2))
    ui.convert_360_180_btn.setEnabled(False)
    tori['reconstruction_arguments']['sino_360_to_180'] = True
    unblock_gui()

def save_cur_proj():
    global last_saved_path, tomo
    filename = QtWidgets.QFileDialog.getSaveFileName(MainWindow, 'Save projections as', last_saved_path, "h5 files (*.h5)")
    if filename[0]=='':
        print('no file specified')
        return
    savefilename = filename[0]
    if savefilename[-3:] != '.h5':
        savefilename = savefilename + '.h5'
    block_gui('Saving projections as {}'.format(savefilename), 0)
    print(savefilename)
    hf = h5py.File(savefilename, 'w')
    hf.create_dataset('exchange/data',data=tomo)
    hf.close()
    unblock_gui()

def save_tori(fname):
    global last_saved_path, tori
    torpath = last_saved_path.split('raw')[0]+'process/recon/tori/'
    if not os.path.isdir(torpath):
        os.makedirs(torpath)
    if not fname:
        print('no name given')
        filename = QtWidgets.QFileDialog.getSaveFileName(MainWindow, 'Save tori parameters file as', torpath, "tori files (*.tori)")
        if filename[0]=='':
            print('no file specified')
            return
        savefilename = filename[0]
        if savefilename[-5:] != '.tori':
            savefilename = savefilename + '.tori'
    if fname:
        savefilename = fname
    block_gui('Saving config as {}'.format(savefilename), 0)
    print(savefilename)
    with open(savefilename, 'w', encoding='utf-8') as f:
        json.dump(tori, f, indent=4)
    unblock_gui()

def load_tori():
    '''
    loads (and optionally executes) an existing tori file
    '''
    global last_saved_path, tori, stripe_arguments
    torpath = last_saved_path.split('raw')[0]+'process/tori/'
    filename = QtWidgets.QFileDialog.getOpenFileName(MainWindow, 'Open tori parameter file', torpath, "tori files (*.tori)")
    if filename[0]=='':
        print('no file specified')
        return
    loadfilename = filename[0]
    with open(loadfilename, 'r') as file:
        try:
            tori = json.load(file)
        except:
            print('error loading json file: wrong format or corrupted file')
    #fill the GUI entry fields
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Question)
    msg.setText("Do you want to load also the data?")
    # setting Message box window title
    msg.setWindowTitle("Load data?")
    msg.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes)
    retval = msg.exec_()
    if retval == 65536:
        print('not loading data')
    if retval == 16384:
        print('loading data...')
        select_data_file(True)
        if tori['file_definitions']['dark_file'] != tori['file_definitions']['data_file']:
            print('loading custom dark...')
            ui.custom_dark_filename_display.setText(tori['file_definitions']['dark_file'])
            ui.custom_dark_path_edit.setText(tori['file_definitions']['dark_path'])
            myslices = [slice(0,None,1),
                        slice(tori['file_definitions']['slice_ver'][0], tori['file_definitions']['slice_ver'][1], tori['file_definitions']['slice_ver'][2]),
                        slice(tori['file_definitions']['slice_hor'][0], tori['file_definitions']['slice_hor'][1], tori['file_definitions']['slice_hor'][2])]
            try:
                block_gui('Wait for dark data to load...', 0)
                dark = load_3d_data(tori['file_definitions']['dark_file'], tori['file_definitions']['dark_path'], myslices)
            except:
                ui.statusbar.showMessage('something went wrong...')
                unblock_gui()
        if tori['file_definitions']['flat_file'] != tori['file_definitions']['data_file']:
            print('loading custom flat...')
            ui.custom_flat_filename_display.setText(tori['file_definitions']['flat_file'])
            ui.custom_flat_path_edit.setText(tori['file_definitions']['flat_path'])
            myslices = [slice(0,None,1),
                        slice(tori['file_definitions']['slice_ver'][0], tori['file_definitions']['slice_ver'][1], tori['file_definitions']['slice_ver'][2]),
                        slice(tori['file_definitions']['slice_hor'][0], tori['file_definitions']['slice_hor'][1], tori['file_definitions']['slice_hor'][2])]
            try:
                block_gui('Wait for flat data to load...', 0)
                flat = load_3d_data(tori['file_definitions']['flat_file'], tori['file_definitions']['flat_path'], myslices)
            except:
                ui.statusbar.showMessage('something went wrong...')
                unblock_gui()
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Question)
    msg.setText("Do you want to process the data according to the tori file parameters?")
    # setting Message box window title
    msg.setWindowTitle("Process data?")
    msg.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes)
    retval = msg.exec_()
    if retval == 65536:
        print('not processing data')
    if retval == 16384:
        print('processing data...')
        ui.dark_correction_check.setChecked(tori['correction_arguments']['normalize_dark'])
        ui.flat_correction_check.setChecked(tori['correction_arguments']['normalize_flat'])
        block_gui('normalizing...', 0)
        normalize()
        if tori['correction_arguments']['normalize_i0']:
            i0_correction()
        if tori['correction_arguments']['normalize_to_unity']:
            normalize_to_unity()
        if tori['correction_arguments']['flatten_images']:
            ui.flatten_filter_size_input.setText(str(tori['correction_arguments']['flatten_images_kwargs']['flatten_size']))
            flatten_all()
        if tori['correction_arguments']['crop']:
            ui.crop_bottom_slider.setValue(tori['correction_arguments']['crop_kwargs']['crop_ranges'][0])
            ui.crop_top_slider.setValue(tori['correction_arguments']['crop_kwargs']['crop_ranges'][1]-1)
            ui.crop_left_slider.setValue(tori['correction_arguments']['crop_kwargs']['crop_ranges'][2])
            ui.crop_right_slider.setValue(tori['correction_arguments']['crop_kwargs']['crop_ranges'][3]-1)
            crop_data()
        if tori['correction_arguments']['outlier_removal']:
            ui.outlier_dif_input.setText(str(tori['correction_arguments']['outlier_removal_kwargs']['dif']))
            ui.outlier_filter_size_spin.setValue(int(tori['correction_arguments']['outlier_removal_kwargs']['size']))
            remove_outliers()
        if tori['correction_arguments']['stripe_removal']:
            ui.stripe_algorithm_select.setCurrentIndex(ui.stripe_algorithm_select.findText(tori['correction_arguments']['stripe_function']))
            stripe_algorithm()
            myargs = tori['correction_arguments']['stripe_removal_kwargs']
            for a in myargs.keys():
                if a == 'pad':
                    ui.stripe_pad_chk.setChecked(myargs['pad'])
                if a == 'norm':
                    ui.stripe_norm_chk.setChecked(myargs['norm'])
                if a == ui.stripe_param1_label.Text():
                    ui.stripe_param1_input.setText(str(myargs[a]))
                if a == ui.stripe_param2_label.Text():
                    ui.stripe_param2_input.setText(str(myargs[a]))
                if a == ui.stripe_param3_label.Text():
                    ui.stripe_param3_input.setText(str(myargs[a]))
                if a == ui.stripe_param4_label.Text():
                    ui.stripe_param5_input.setText(str(myargs[a]))
            remove_stripes()
        if tori['correction_arguments']['minus_log_before_phase']:
            apply_minus_log(True)
        if tori['correction_arguments']['phase_retrieval']:
            ui.phase_pad_chk.setChecked(tori['correction_arguments']['phase_retrieval_kwargs']['pad'])
            ui.phase_deltabeta_select.setCurrentIndex(1)
            ui.delta_beta_value.setText(str(tori['correction_arguments']['phase_retrieval_kwargs']['alpha']))
            full_data_phase_retrieval()
        if tori['correction_arguments']['minus_log_after_phase']:
            apply_minus_log(True)
        if tori['reconstruction_arguments']['sino_360_to_180']:
            ui.cor_value.setText(str(tori['reconstruction_arguments']['sino_360_to_180_old_center']))
            conv_360_180()
        #reconstrution here
        ui.recon_algorithm_select.setCurrentIndex(ui.recon_algorithm_select.findText(tori['reconstruction_arguments']['recon_kwargs']['algorithm']))
        ui.recon_filter_select.setCurrentIndex(ui.recon_filter_select.findText(tori['reconstruction_arguments']['recon_kwargs']['filter_name']))
        if 'filter_par' in tori['reconstruction_arguments']['recon_kwargs'].keys():
            ui.recon_filter_par_list.setText(str(tori['reconstruction_arguments']['recon_kwargs']['filter_par'].replace('[','').replace(']','')))
        ui.recon_pad_chk.setChecked(tori['reconstruction_arguments']['pad'])
        reconstruct_all()
        #post-processing
        if tori['post_recon_process_arguments']['circ_mask']:
            ui.postp_circ_mask_chk.setChecked(True)
            ui.postp_circ_mask_spin.setValue(tori['post_recon_process_arguments']['circ_mask_kwargs']['ratio'])
            ui.postp_circ_mask_value.setText(str(tori['post_recon_process_arguments']['circ_mask_kwargs']['val']))
            apply_circ_mask()
        if tori['post_recon_process_arguments']['ring_removal']:
            if 'rwidth' in tori['post_recon_process_arguments']['ring_removal_kwargs'].keys():
                ui.postp_ring_size.setText(str(tori['post_recon_process_arguments']['ring_removal_kwargs']['rwidth']))
            if 'theta_min' in tori['post_recon_process_arguments']['ring_removal_kwargs'].keys():
                ui.postp_ring_thetamin.setText(str(tori['post_recon_process_arguments']['ring_removal_kwargs']['theta_min']))
            if 'thresh' in tori['post_recon_process_arguments']['ring_removal_kwargs'].keys():
                ui.postp_ring_thresh.setText(str(tori['post_recon_process_arguments']['ring_removal_kwargs']['thresh']))
            remove_ring_full()
        if tori['post_recon_process_arguments']['convert']:
            if tori['post_recon_process_arguments']['convert_kwargs']['mode'] == 'minmax':
                ui.postp_convert_bit_scale_combo.setCurrentIndex(0)
            if tori['post_recon_process_arguments']['convert_kwargs']['mode'] == 'std':
                ui.postp_convert_bit_scale_combo.setCurrentIndex(1)
            if tori['post_recon_process_arguments']['convert_kwargs']['mode'] == 'manual':
                ui.postp_convert_bit_scale_combo.setCurrentIndex(2)
            if tori['post_recon_process_arguments']['convert_kwargs']['dtype'] == 'float16':
                ui.postp_convert_bit_type_combo.setCurrentIndex(0)
            if tori['post_recon_process_arguments']['convert_kwargs']['dtype'] == 'uint16':
                ui.postp_convert_bit_type_combo.setCurrentIndex(1)
            if tori['post_recon_process_arguments']['convert_kwargs']['dtype'] == 'uint8':
                ui.postp_convert_bit_type_combo.setCurrentIndex(2)
            reduce_bit_depth([tori['post_recon_process_arguments']['convert_kwargs']['min'], tori['post_recon_process_arguments']['convert_kwargs']['max']])
        if tori['post_recon_process_arguments']['downscale']:
            ui.postp_binning_chk.setChecked(True)
            ui.postp_binning_spin.setValue(int(tori['post_recon_process_arguments']['downscale_kwargs']['down_scale_factors'][0]))
    unblock_gui()


def create_task():
    global loadfilename, tori_config
    taskname = loadfilename.split('raw')[0]+'/process/tasks/'+loadfilename.split('/')[-1].replace('.h5','.task')
    taskjson = {}
    taskjson['tori'] = tori_config
    taskjson['file'] = loadfilename
    with open(taskname, 'w', encoding='utf-8') as f:
        json.dump(taskjson, f, indent=4)
    ui.statusbar.showMessage('task file saved', 5000)

def show_scale(show):
    if show:
        scalebar.show()
    else:
        scalebar.hide()

def toggle_log(islog):
    colormap = ui.silx_plot.getColorBarWidget().getColormap()
    if islog:
        colormap.setNormalization('log')
    else:
        colormap.setNormalization('linear')

def plot_limits_changed():
    '''recalculate local variance and std if nessesarily!'''
    if ui.controls_tabs.currentIndex() != 3: #not a COR tab
        return
    if len(cor_range_slices) == 0:
        return
    if ui.cor_autorecalc_chk.isChecked(): #auto recalc is selected
        recalc_TV_STD()

def remove_ring_single():
    '''preview ring removal algorithm on a single slice
    '''
    global recon, tori
    if ui.plot_selector.currentText != 'XY': #force switch to XY stripe_function_name
        ui.plot_selector.setCurrentIndex(0)
    single_slice = np.empty(shape = (1, recon.shape[1], recon.shape[2]), dtype = 'float32')
    single_slice[0,:,:] = recon[ui.plot_slider.value(),:,:]
    kwargs = {}
    kwargs['rec'] = single_slice
    if ui.postp_ring_size.text() != '':
        try:
            rwidth = int(ui.postp_ring_size.text())
            kwargs['rwidth'] = rwidth
        except:
            ui.statusbar.showMessage('provide a valid size (rwidth) value or no value', 5000)
            return
    if ui.postp_ring_thetamin.text() != '':
        try:
            theta_min = float(ui.postp_ring_thetamin.text())
            kwargs['theta_min'] = theta_min
        except:
            ui.statusbar.showMessage('provide a valid theta min value or no value', 5000)
            return
    if ui.postp_ring_thresh.text() != '':
        try:
            thresh = float(ui.postp_ring_thresh.text())
            kwargs['thresh'] = thresh
        except:
            ui.statusbar.showMessage('provide a valid threshhold (offset) value or no value', 5000)
            return
    block_gui('removing ring from a single slice...', 0)
    try:
        #print(f'ring remove args = {kwargs}')
        single_slice = tomopy.remove_ring(**kwargs)
        image.setData(single_slice[0])
        unblock_gui()
    except:
        unblock_gui()
        ui.statusbar.showMessage('something went wrong', 5000)

def remove_ring_full():
    '''full ring removal
    '''
    global recon, tori
    kwargs = {}
    if ui.postp_ring_size.text() != '':
        try:
            rwidth = int(ui.postp_ring_size.text())
            kwargs['rwidth'] = rwidth
            tori['post_recon_process_arguments']['ring_removal_kwargs']['rwidth'] = rwidth
        except:
            ui.statusbar.showMessage('provide a valid size (rwidth) value or no value', 5000)
            return
    if ui.postp_ring_thetamin.text() != '':
        try:
            theta_min = float(ui.postp_ring_thetamin.text())
            kwargs['theta_min'] = theta_min
            tori['post_recon_process_arguments']['ring_removal_kwargs']['theta_min'] = theta_min
        except:
            ui.statusbar.showMessage('provide a valid theta min value or no value', 5000)
            return
    if ui.postp_ring_thresh.text() != '':
        try:
            thresh = float(ui.postp_ring_thresh.text())
            kwargs['thresh'] = thresh
            tori['post_recon_process_arguments']['ring_removal_kwargs']['thresh'] = thresh
        except:
            ui.statusbar.showMessage('provide a valid threshhold (offset) value or no value', 5000)
            return
    nchunk = 20
    chunksize = recon.shape[0]/nchunk
    if chunksize < 5:
        nchunk = 1
        chunksize = recon.shape[0]
    for i in range (0,nchunk):
        block_gui('wait for rings removal...', int(i*100/nchunk))
        kwargs['rec'] = recon[int(np.round(i*chunksize)):int(np.round((i+1)*chunksize))]
        recon[int(np.round(i*chunksize)):int(np.round((i+1)*chunksize))] = tomopy.remove_ring(**kwargs)
    tori['post_recon_process_arguments']['ring_removal'] = True
    unblock_gui()
    plot_slider_update()

def getVscale(plot):
    '''Function to get current min and max intensity values from the silx plot
    plot: a silx plot
    returns Vmin, Vmax
    '''
    colormap = plot.getColorBarWidget().getColormap()
    Vmin = colormap.getVMin()
    if not Vmin: #means an autoscale is selected, have to find the value myself
        curimage = image.getData()
        if colormap.getAutoscaleMode() == 'stddev3':
            Vmin = np.mean(curimage) - 3*np.std(curimage)
        if colormap.getAutoscaleMode() == 'minmax':
            Vmin = np.min(curimage)
    Vmax = colormap.getVMax()
    if not Vmax: #means an autoscale is selected, have to find the value myself
        curimage = image.getData()
        if colormap.getAutoscaleMode() == 'stddev3':
            Vmax = np.mean(curimage) + 3*np.std(curimage)
        if colormap.getAutoscaleMode() == 'minmax':
            Vmax = np.max(curimage)
    return(Vmin, Vmax)


def reduce_bit_depth(prov):
    '''convert reconstructed data to a smaller bit depth
    '''
    global recon, tori
    curim = image.getData()
    if ui.postp_convert_bit_scale_combo.currentText() == 'min/max':
        Vmin = np.min(curim)
        Vmax = np.max(curim)
        tori['post_recon_process_arguments']['convert_kwargs']['mode'] = 'minmax'
    if ui.postp_convert_bit_scale_combo.currentText() == 'mean+-3std':
        Vmin = np.mean(curim) - 3*np.std(curim)
        Vmax = np.mean(curim) + 3*np.std(curim)
        tori['post_recon_process_arguments']['convert_kwargs']['mode'] = 'std'
    if ui.postp_convert_bit_scale_combo.currentText() == 'current graph view':
        Vmin, Vmax = getVscale(ui.silx_plot)
        tori['post_recon_process_arguments']['convert_kwargs']['mode'] = "manual"
    tori['post_recon_process_arguments']['convert'] = True
    if prov:
        Vmin = prov[0]
        Vmax = prov[1]
    tori['post_recon_process_arguments']['convert_kwargs']['min'] = Vmin
    tori['post_recon_process_arguments']['convert_kwargs']['max'] = Vmax
    if ui.postp_convert_bit_type_combo.currentText() == 'float 16-bit':
        block_gui('converting results to float16...',0)
        recon = recon.astype('float16')
        tori['post_recon_process_arguments']['convert_kwargs']['dtype'] = 'float16'
    if ui.postp_convert_bit_type_combo.currentText() == 'uint 16-bit':
        block_gui('converting results to uint16...',0)
        recon=65535*(recon-Vmin)/(Vmax-Vmin)
        recon=recon.clip(min=0, max=65535).astype('uint16')
        tori['post_recon_process_arguments']['convert_kwargs']['dtype'] = 'uint16'
    if ui.postp_convert_bit_type_combo.currentText() == 'uint 8-bit':
        block_gui('converting results to uint8...',0)
        recon=255*(recon-Vmin)/(Vmax-Vmin)
        recon=recon.clip(min=0, max=255).astype('uint8')
        tori['post_recon_process_arguments']['convert_kwargs']['dtype'] = 'uint8'
    unblock_gui()
    plot_slider_update()


def select_save_file():
    '''selects a custom save file name
    '''
    filename = QtWidgets.QFileDialog.getSaveFileName(MainWindow, 'Save results as...', last_saved_path, "h5 files (*.h5)")
    if filename[0]=='':
        print('no file specified')
        return
    ui.postp_save_path.setText(filename[0])

def flatten_one_level(inputdict):
    resdict = {}
    for key in inputdict.keys():
        if isinstance(inputdict[key], dict):
            for dkey in inputdict[key].keys():
                resdict[f'{key}/{dkey}'] = inputdict[key][dkey]
        else:
            resdict[key] = inputdict[key]
    return resdict

def generate_python_script(tori):
    #global tori
    script=[]
    script.append('# This python script is supposed to recreate the same result')
    script.append('# as the reconstructor')
    script.append('')
    script.append(f'# made with python version = {sys.version.split("|")[0]}')
    script.append('')
    script.append(f'import numpy as np         # version = {np.__version__}')
    script.append(f'import tomopy              # version = {tomopy.__version__}')
    script.append(f'import h5py                # version = {h5py.__version__}')
    script.append(f'from scipy import ndimage  # version = {scipy.__version__}')
    script.append('')
    script.append('# importing data')
    script.append(f'with h5py.File("{tori["file_definitions"]["data_file"]}", "r") as fh:')
    script.append(f'    tomo = fh.get("{tori["file_definitions"]["data_path"]}")[{tori["file_definitions"]["slice_proj"][0]}:{tori["file_definitions"]["slice_proj"][1]}:{tori["file_definitions"]["slice_proj"][2]}, {tori["file_definitions"]["slice_ver"][0]}:{tori["file_definitions"]["slice_ver"][1]}:{tori["file_definitions"]["slice_ver"][2]}, {tori["file_definitions"]["slice_hor"][0]}:{tori["file_definitions"]["slice_hor"][1]}:{tori["file_definitions"]["slice_hor"][2]}]')
    if 'dark_file' in tori["file_definitions"].keys() and tori["file_definitions"]["data_file"] == tori["file_definitions"]["dark_file"]:
        script.append(f'    dark = fh.get("{tori["file_definitions"]["dark_path"]}")[:, {tori["file_definitions"]["slice_ver"][0]}:{tori["file_definitions"]["slice_ver"][1]}:{tori["file_definitions"]["slice_ver"][2]}, {tori["file_definitions"]["slice_hor"][0]}:{tori["file_definitions"]["slice_hor"][1]}:{tori["file_definitions"]["slice_hor"][2]}]')
        script.append('    if len(dark.shape) == 3:')
        script.append('        dark = np.mean(dark, axis = 0)')
    if 'flat_file' in tori["file_definitions"].keys() and tori["file_definitions"]["data_file"] == tori["file_definitions"]["flat_file"]:
        script.append(f'    flat = fh.get("{tori["file_definitions"]["flat_path"]}")[:, {tori["file_definitions"]["slice_ver"][0]}:{tori["file_definitions"]["slice_ver"][1]}:{tori["file_definitions"]["slice_ver"][2]}, {tori["file_definitions"]["slice_hor"][0]}:{tori["file_definitions"]["slice_hor"][1]}:{tori["file_definitions"]["slice_hor"][2]}]')
        script.append('    if len(flat.shape) == 3:')
        script.append('        if flat.shape[0] != tomo.shape[0]:')
        script.append('            flat = np.mean(flat, axis = 0)')
    if tori["file_definitions"]["theta_source"] == 'exchange':
        script.append(f"    theta = fh.get('exchange/theta')[{tori['file_definitions']['slice_proj'][0]}:{tori['file_definitions']['slice_proj'][1]}:{tori['file_definitions']['slice_proj'][2]}]")
    if tori["correction_arguments"]["normalize_i0"]:
        script.append(f"    i0_data = fh.get('instrument/beam_monitor/data')[{tori['file_definitions']['slice_proj'][0]}:{tori['file_definitions']['slice_proj'][1]}:{tori['file_definitions']['slice_proj'][2]}]")
        script.append("    i0_data = i0_data/i0_data[0]")
    #in case dark or flat are from another sources
    if 'dark_file' in tori["file_definitions"].keys() and tori["file_definitions"]["data_file"] != tori["file_definitions"]["dark_file"]:
        script.append(f'dark = h5py.File("{tori["file_definitions"]["data_file"]}", "r").get("{tori["file_definitions"]["dark_path"]}")[:,{tori["file_definitions"]["slice_ver"][0]}:{tori["file_definitions"]["slice_ver"][1]}:{tori["file_definitions"]["slice_ver"][2]}, {tori["file_definitions"]["slice_hor"][0]}:{tori["file_definitions"]["slice_hor"][1]}:{tori["file_definitions"]["slice_hor"][2]}]')
        script.append('if len(dark.shape) == 3:')
        script.append('    dark = np.mean(dark, axis = 0)')
    if 'flat_file' in tori["file_definitions"].keys() and tori["file_definitions"]["data_file"] != tori["file_definitions"]["flat_file"]:
        script.append(f'flat = h5py.File("{tori["file_definitions"]["flat_file"]}", "r").get("{tori["file_definitions"]["flat_path"]}")[:, {tori["file_definitions"]["slice_ver"][0]}:{tori["file_definitions"]["slice_ver"][1]}:{tori["file_definitions"]["slice_ver"][2]}, {tori["file_definitions"]["slice_hor"][0]}:{tori["file_definitions"]["slice_hor"][1]}:{tori["file_definitions"]["slice_hor"][2]}]')
        script.append('if len(flat.shape) == 3:')
        script.append('    if flat.shape[0] != tomo.shape[0]:')
        script.append('        flat = np.mean(flat, axis = 0)')
    if tori["file_definitions"]["theta_source"] == 'calc':
        script.append(f'theta  = np.linspace({tori["file_definitions"]["theta_start"]},{tori["file_definitions"]["theta_end"]},{(tori["file_definitions"]["slice_proj"][1]-tori["file_definitions"]["slice_proj"][0])/tori["file_definitions"]["slice_proj"][2]})')
    script.append('theta = np.radians(theta)')
    script.append('')
    script.append('# normalization (if any)')
    if tori["correction_arguments"]["normalize_dark"] and not tori["correction_arguments"]["normalize_flat"]:#only dark correction
        script.append('tomo = tomo - dark')
    if not tori["correction_arguments"]["normalize_dark"] and tori["correction_arguments"]["normalize_flat"]:#only flat correction
        script.append('tomo = tomo/flat')
    if tori["correction_arguments"]["normalize_dark"] and tori["correction_arguments"]["normalize_flat"]:#all correction
        script.append('tomo = tomopy.prep.normalize.normalize(tomo, flat, dark)')
    script.append('# correction (if any)')
    if tori["correction_arguments"]["crop"]:
        script.append('# cropping after loading:')
        script.append(f'tomo = tomo[:,{tori["correction_arguments"]["crop_kwargs"]["crop_ranges"][0]}:{tori["correction_arguments"]["crop_kwargs"]["crop_ranges"][1]},{tori["correction_arguments"]["crop_kwargs"]["crop_ranges"][2]}:{tori["correction_arguments"]["crop_kwargs"]["crop_ranges"][3]}]')
    if tori["correction_arguments"]["normalize_i0"]:
        script.append('# normalizing to I0 scalars')
        script.append('for i in range (0, tomo.shape[0]):')
        script.append('    tomo[i] = tomo[i]/i0_data[i]')
    if tori["correction_arguments"]["normalize_to_unity"]:
        script.append('# normalizing each projection to unity')
        script.append('for i in range(0,tomo.shape[0]):')
        script.append('    tomo[i] = tomo[i]/np.mean(tomo[i])')
    if tori["correction_arguments"]["outlier_removal"]:
        script.append('# remove outliers')
        script.append(f'tomo = tomopy.misc.corr.remove_outlier(tomo, {tori["correction_arguments"]["outlier_removal_kwargs"]["dif"]}, {tori["correction_arguments"]["outlier_removal_kwargs"]["size"]})')
    if tori["correction_arguments"]["flatten_images"]:
        script.append('# flatten projections')
        script.append('for i in range (0, tomo.shape[0]):')
        script.append(f'    tomo[i] = ndimage.uniform_filter(tomo[i], size = {tori["correction_arguments"]["flatten_images_kwargs"]["flatten_size"]})')
    if tori["correction_arguments"]["stripe_removal"]:
        script.append('# remove stripes')
        param_string = ' ,'
        for i in tori["correction_arguments"]["stripe_removal_kwargs"]:
            if type(tori["correction_arguments"]["stripe_removal_kwargs"][i]) == str:
                param_string = f'{param_string}, {i} = "{tori["correction_arguments"]["stripe_removal_kwargs"][i]}"'
            else:
                param_string = f'{param_string}, {i} = {tori["correction_arguments"]["stripe_removal_kwargs"][i]}'
        script.append(f'tomo = tomopy.{tori["correction_arguments"]["stripe_function"]}(tomo = tomo{param_string})')
    width = (tori["file_definitions"][ "slice_hor"][1] - tori["file_definitions"][ "slice_hor"][0])/tori["file_definitions"][ "slice_hor"][2]
    if tori["correction_arguments"]["crop"]:
        width = tori["correction_arguments"]["crop_kwargs"]["crop_ranges"][3] - tori["correction_arguments"]["crop_kwargs"]["crop_ranges"][2]
    if tori["reconstruction_arguments"]["sino_360_to_180"]:
        if tori['reconstruction_arguments']['sino_360_to_180_old_center'] > width//2:
            direction = 'right'
            overlap = 2*int(width - tori['reconstruction_arguments']['sino_360_to_180_old_center'])
        else:
            direction = 'left'
            overlap = 2*int(tori['reconstruction_arguments']['sino_360_to_180_old_center'])
        width = int(2*width - overlap)
        script.append(f'tomo = tomopy.misc.morph.sino_360_to_180(tomo, overlap = {overlap}, rotation = "{direction}")')
        script.append('theta = tomopy.angles(tomo.shape[0]+1)[:-1]')
    if tori["correction_arguments"]["minus_log_before_phase"]:
        script.append('tomo = -np.log(tomo)')
        script.append('tomo = tomopy.misc.corr.remove_nan(tomo)')
    if tori["correction_arguments"]["phase_retrieval"]:
        script.append('#phase retrieval')
        script.append(f'tomo = tomopy.retrieve_phase(tomo, pixel_size = {tori["correction_arguments"]["phase_retrieval_kwargs"]["pixel_size"]}, dist = {tori["correction_arguments"]["phase_retrieval_kwargs"]["dist"]}, energy = {tori["correction_arguments"]["phase_retrieval_kwargs"]["energy"]}, alpha = {tori["correction_arguments"]["phase_retrieval_kwargs"]["alpha"]}, pad = {tori["correction_arguments"]["phase_retrieval_kwargs"]["pad"]})')
    if tori["correction_arguments"]["minus_log_after_phase"]:
        script.append('tomo = -np.log(tomo)')
        script.append('tomo = tomopy.misc.corr.remove_nan(tomo)')
    padsize = 0
    if tori['reconstruction_arguments']['pad']:
        padsize = width//4
        script.append(f'tomo = tomopy.misc.morph.pad(tomo, axis=2, npad={int(padsize)}, mode="edge")')
    script.append("#reconstruct")
    param_string = 'tomo = tomo'
    param_string = f'{param_string}, algorithm = "{tori["reconstruction_arguments"]["recon_kwargs"]["algorithm"]}"'
    if 'filter_name' in tori['reconstruction_arguments']['recon_kwargs'].keys():
        param_string = f'{param_string}, filter_name = "{tori["reconstruction_arguments"]["recon_kwargs"]["filter_name"]}"'
    if 'filter_par' in tori['reconstruction_arguments']['recon_kwargs'].keys():
        param_string = f"{param_string}, filter_par = ["
        for i in tori['reconstruction_arguments']['recon_kwargs']['filter_par']:
            param_string = f'{param_string},{i}'
        param_string = f"{param_string}]"
    for i in tori['reconstruction_arguments']['recon_kwargs']:
        if i == 'algorithm':
            pass
        elif i == 'filter_name':
            pass
        elif i == 'filter_par':
            pass
        elif i == 'ncore':
            pass
        elif i == 'nchunk':
            pass
        elif i == 'center':
            param_string = f'{param_string}, center = {tori["reconstruction_arguments"]["recon_kwargs"]["center"] + padsize}'
        else:
            param_string = f'{param_string}, {i} = {tori["reconstruction_arguments"]["recon_kwargs"][i]}'
    script.append(f'recon = tomopy.recon({param_string}, sinogram_order = False, theta = theta)')
    return script

    #for h in savehistory:
    #        fulltext=fulltext+h+"\n"
    #    ui.param_text_edit.setPlainText(fulltext)

if __name__ == "__main__":
    ui.plot_next_btn.clicked.connect(next_tab)
    ui.controls_tabs.currentChanged.connect(tab_selected)
    ui.select_data_btn.clicked.connect(select_data_file)
    ui.custom_dark_btn.clicked.connect(load_custom_dark)
    ui.custom_flat_btn.clicked.connect(load_custom_flat)
    ui.normalize_btn.clicked.connect(normalize)
    ui.normalize_btn.setEnabled(False)
    ui.unity_normalize_btn.clicked.connect(normalize_to_unity)
    ui.crop_top_slider.valueChanged.connect(crop_selection.crop_changed)
    ui.crop_top_index.returnPressed.connect(crop_top_index_changed)
    ui.crop_bottom_index.returnPressed.connect(crop_bottom_index_changed)
    ui.crop_left_index.returnPressed.connect(crop_left_index_changed)
    ui.crop_right_index.returnPressed.connect(crop_right_index_changed)
    ui.crop_bottom_slider.valueChanged.connect(crop_selection.crop_changed)
    ui.crop_left_slider.valueChanged.connect(crop_selection.crop_changed)
    ui.crop_right_slider.valueChanged.connect(crop_selection.crop_changed)
    ui.crop_btn.clicked.connect(crop_data)
    ui.stripe_algorithm_select.currentIndexChanged.connect(stripe_algorithm)
    ui.stripe_preview_btn.clicked.connect(stripe_one_slice)
    ui.remove_stripes_btn.clicked.connect(remove_stripes)
    ui.cor_tab_slider.valueChanged.connect(cor_height_slider_update)
    ui.cor_tab_index.returnPressed.connect(cor_height_index_udpate)
    ui.cor_use_theta_from_file_rbtn.clicked.connect(switch_to_theta_from_file)
    ui.cor_calculate_theta_rbtn.clicked.connect(switch_to_calc_theta)
    ui.cor_algorithm_select.currentIndexChanged.connect(cor_algorithm_selected)
    ui.cor_find_auto_btn.clicked.connect(find_center_auto)
    ui.cor_start_theta_input.returnPressed.connect(eval_theta_ranges)
    ui.cor_end_theta_input.returnPressed.connect(eval_theta_ranges)
    ui.cor_preview_btn.clicked.connect(cor_single_slice_recon)
    ui.cor_recon_in_range_btn.clicked.connect(cor_calculate_range)
    ui.cor_tab_value_slider.valueChanged.connect(cor_value_slider)
    ui.cor_tab_usethis_btn.clicked.connect(use_this_cor)
    ui.cor_value.returnPressed.connect(cor_is_given)
    ui.phase_energy_inp.returnPressed.connect(energy_entered)
    ui.phase_pixel_size.returnPressed.connect(pixel_size_entered)
    ui.phase_sdd.returnPressed.connect(sdd_entered)
    ui.phase_pixel_unit_select.currentIndexChanged.connect(pixel_unit_changed)
    ui.phase_sdd_unit_select.currentIndexChanged.connect(sdd_unit_changed)
    ui.phase_slider.valueChanged.connect(phase_slider_changed)
    ui.phase_index.returnPressed.connect(phase_index_changed)
    ui.delta_beta_value.returnPressed.connect(deltabeta_changed)
    ui.phase_retrieve_single_btn.clicked.connect(single_phase_retrieve)
    ui.phase_deltabeta_select.currentIndexChanged.connect(deltabeta_switched)
    ui.phase_range_retrieve_btn.clicked.connect(retrieve_phase_range)
    ui.phase_image_refraction_usethis_btn.clicked.connect(use_this_phase)
    ui.phase_image_refraction_slider.valueChanged.connect(phase_range_slider_moved)
    ui.phase_retrieve_btn.clicked.connect(full_data_phase_retrieval)
    ui.phase_proj_slice_select.currentIndexChanged.connect(phase_proj_slice_switched)
    ui.recon_run_recon_btn.clicked.connect(reconstruct_all)
    ui.recon_try_filters_btn.clicked.connect(try_all_filters)
    ui.recon_first_slice_slider.valueChanged.connect(recon_first_slice_slider_changed)
    ui.recon_last_slice_slider.valueChanged.connect(recon_last_slice_slider_changed)
    ui.recon_first_slice_index.returnPressed.connect(recon_first_slice_index_changed)
    ui.recon_last_slice_index.returnPressed.connect(recon_last_slice_index_changed)
    ui.apply_minus_log_btn.clicked.connect(apply_minus_log)
    ui.apply_minus_log_btn2.clicked.connect(apply_minus_log)
    ui.recon_algorithm_select.currentIndexChanged.connect(recon_algorithm_selected)
    ui.remove_nan_btn.clicked.connect(remove_nans)
    ui.convert_360_180_btn.clicked.connect(conv_360_180)
    ui.actionSave_current_projections_as.triggered.connect(save_cur_proj)
    ui.actionSave_recon_parameters_file.triggered.connect(save_tori)
    ui.actionLoad_recon_parameters_file.triggered.connect(load_tori)
    FilterDialog = QtWidgets.QDialog()
    dui = Ui_filter_dialog()
    dui.setupUi(FilterDialog)
    dui.dataset_combo.currentIndexChanged.connect(new_dataset)
    dui.proj_downsample_spin.valueChanged.connect(recalc)
    dui.hor_downsample_spin.valueChanged.connect(recalc)
    dui.ver_downsample_spin.valueChanged.connect(recalc)
    dui.proj_range_0_spin.valueChanged.connect(recalc)
    dui.proj_range_1_spin.valueChanged.connect(recalc)
    dui.ver_range_0_spin.valueChanged.connect(recalc)
    dui.ver_range_1_spin.valueChanged.connect(recalc)
    dui.hor_range_0_spin.valueChanged.connect(recalc)
    dui.hor_range_1_spin.valueChanged.connect(recalc)
    dui.central_10_btn.clicked.connect(filter_center_10)
    dui.central_50_btn.clicked.connect(filter_center_50)
    dui.central_100_btn.clicked.connect(filter_center_100)
    dui.central_all_btn.clicked.connect(filter_center_all)
    DatasetDialog = QtWidgets.QDialog()
    dsui = Ui_Dialog()
    dsui.setupUi(DatasetDialog)
    AboutDialog = QtWidgets.QDialog()
    abui = Ui_About()
    abui.setupUi(AboutDialog)
    ui.actionAbout.triggered.connect(AboutDialog.exec)
    ui.submit_task_btn.clicked.connect(create_task)
    ui.plot_tab_scale_chk.toggled.connect(show_scale)
    ui.plot_tab_displaylog_chk.toggled.connect(toggle_log)
    ui.plot_slider.valueChanged.connect(plot_slider_update)
    ui.plot_selector.currentIndexChanged.connect(image_source_changed)
    ui.plot_index.returnPressed.connect(plot_index_update)
    ui.i0_normalize_btn.clicked.connect(i0_correction)
    ui.flatten_preview_btn.clicked.connect(flatten_preview)
    ui.flatten_projections_btn.clicked.connect(flatten_all)
    ui.outlier_preview_btn.clicked.connect(outliers_one_slice)
    ui.remove_outliers_btn.clicked.connect(remove_outliers)
    ui.cor_recalc_btn.clicked.connect(recalc_TV_STD)
    ui.silx_plot.getXAxis().sigLimitsChanged.connect(plot_limits_changed)
    ui.postp_crop_chk.toggled.connect(postp_crop_checked)
    ui.postp_top_slider.valueChanged.connect(postp_crop_selection.crop_changed)
    ui.postp_top_index.returnPressed.connect(postp_crop_top_index_changed)
    ui.postp_bottom_index.returnPressed.connect(postp_crop_bottom_index_changed)
    ui.postp_left_index.returnPressed.connect(postp_crop_left_index_changed)
    ui.postp_right_index.returnPressed.connect(postp_crop_right_index_changed)
    ui.postp_bottom_slider.valueChanged.connect(postp_crop_selection.crop_changed)
    ui.postp_left_slider.valueChanged.connect(postp_crop_selection.crop_changed)
    ui.postp_right_slider.valueChanged.connect(postp_crop_selection.crop_changed)
    ui.postp_crop_btn.clicked.connect(crop_recon)
    ui.postp_circ_mask_chk.toggled.connect(draw_a_circular_mask)
    ui.postp_circ_mask_spin.valueChanged.connect(draw_a_circular_mask)
    ui.postp_circ_mask_apply.clicked.connect(apply_circ_mask)
    ui.postp_ring_prev_btn.clicked.connect(remove_ring_single)
    ui.postp_ring_all_btn.clicked.connect(remove_ring_full)
    ui.postp_convert_bit_btn.clicked.connect(reduce_bit_depth)
    ui.postp_save_btn.clicked.connect(save_results)
    ui.postp_select_output_btn.clicked.connect(select_save_file)
    ui.param_tori_radio.toggled.connect(param_type_changed)
    ui.param_python_radio.toggled.connect(param_type_changed)
    for c in excl_reg_chk:
        c.toggled.connect(excl_regions_changed)
    for s in excl_reg_to_spin:
        s.valueChanged.connect(excl_regions_changed)
    for s in excl_reg_from_spin:
        s.valueChanged.connect(excl_regions_changed)
    make_silx_plot()
    tori = fresh_tori()
    tab_selected()
    MainWindow.show()
    sys.exit(app.exec_())

