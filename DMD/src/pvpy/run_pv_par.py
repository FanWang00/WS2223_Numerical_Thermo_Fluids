# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:02:55 2020

@author: AdminF
"""

# trace generated using paraview version 5.7.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *

import argparse
parser = argparse.ArgumentParser()

## input variables
# input_file = ['/media/fan/Seagate Expansion Drive/output/L5/aerobreakup_l5_40/DMD_Recons/data_index.xdmf',
#               '/media/fan/Seagate Expansion Drive/PostProcess/interp/L5/data_index.xdmf'] 
# input_item = 'velocityU'
# input_CellArr = ['pressure', 'velocityU']
# input_save_path = '/media/fan/Seagate Expansion Drive/test/test_file.png'
# input_fram = [0, 767]
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="input file path", nargs=2)
parser.add_argument("input_CellArr", help="input CellArr")
parser.add_argument("input_item", help="input item")
parser.add_argument("input_save_path", help="save path")
parser.add_argument("input_fram", help="fram of pngs to save", nargs=2, type=int)
args = parser.parse_args()

input_file = args.input_file
input_item = args.input_item
input_CellArr =args.input_CellArr
input_save_path = args.input_save_path
input_fram = args.input_fram
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XDMF Reader'
data_indexxdmf = XDMFReader(FileNames=input_file[0])
data_indexxdmf.CellArrayStatus = input_CellArr

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# set active source
SetActiveSource(data_indexxdmf)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1075, 591]

# show data in view
data_indexxdmfDisplay = Show(data_indexxdmf, renderView1)

# get color transfer function/color map for 'velocityU'
velocityULUT = GetColorTransferFunction(input_item)

# get opacity transfer function/opacity map for 'velocityU'
velocityUPWF = GetOpacityTransferFunction(input_item)

# trace defaults for the display properties.
data_indexxdmfDisplay.Representation = 'Surface'
data_indexxdmfDisplay.ColorArrayName = ['CELLS', input_item]
data_indexxdmfDisplay.LookupTable = velocityULUT
data_indexxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
data_indexxdmfDisplay.SelectOrientationVectors = 'None'
data_indexxdmfDisplay.ScaleFactor = 0.01878906250000002
data_indexxdmfDisplay.SelectScaleArray = input_item
data_indexxdmfDisplay.GlyphType = 'Arrow'
data_indexxdmfDisplay.GlyphTableIndexArray = input_item
data_indexxdmfDisplay.GaussianRadius = 0.0009394531250000008
data_indexxdmfDisplay.SetScaleArray = [None, '']
data_indexxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
data_indexxdmfDisplay.OpacityArray = [None, '']
data_indexxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
data_indexxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
data_indexxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
data_indexxdmfDisplay.ScalarOpacityFunction = velocityUPWF
data_indexxdmfDisplay.ScalarOpacityUnitDistance = 0.0043283601635017155

# show color bar/color legend
data_indexxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# reset view to fit data
renderView1.ResetCamera()

# get layout
layout1 = GetLayout()

# split cell
layout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(None)

# create a new 'XDMF Reader'
data_indexxdmf_1 = XDMFReader(FileNames=input_file[1])
data_indexxdmf_1.CellArrayStatus = input_CellArr

# set active source
SetActiveSource(data_indexxdmf_1)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraFocalDisk = 1.0
renderView2.Background = [0.32, 0.34, 0.43]
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1
# uncomment following to set a specific view size
# renderView2.ViewSize = [400, 400]

# show data in view
data_indexxdmf_1Display = Show(data_indexxdmf_1, renderView2)

# trace defaults for the display properties.
data_indexxdmf_1Display.Representation = 'Surface'
data_indexxdmf_1Display.ColorArrayName = ['CELLS', input_item]
data_indexxdmf_1Display.LookupTable = velocityULUT
data_indexxdmf_1Display.OSPRayScaleFunction = 'PiecewiseFunction'
data_indexxdmf_1Display.SelectOrientationVectors = 'None'
data_indexxdmf_1Display.ScaleFactor = 0.01878906250000002
data_indexxdmf_1Display.SelectScaleArray = input_item
data_indexxdmf_1Display.GlyphType = 'Arrow'
data_indexxdmf_1Display.GlyphTableIndexArray = input_item
data_indexxdmf_1Display.GaussianRadius = 0.0009394531250000008
data_indexxdmf_1Display.SetScaleArray = [None, '']
data_indexxdmf_1Display.ScaleTransferFunction = 'PiecewiseFunction'
data_indexxdmf_1Display.OpacityArray = [None, '']
data_indexxdmf_1Display.OpacityTransferFunction = 'PiecewiseFunction'
data_indexxdmf_1Display.DataAxesGrid = 'GridAxesRepresentation'
data_indexxdmf_1Display.PolarAxes = 'PolarAxesRepresentation'
data_indexxdmf_1Display.ScalarOpacityFunction = velocityUPWF
data_indexxdmf_1Display.ScalarOpacityUnitDistance = 0.0043283601635017155

# add view to a layout so it's visible in UI
AssignViewToLayout(view=renderView2, layout=layout1, hint=2)

# show color bar/color legend
data_indexxdmf_1Display.SetScalarBarVisibility(renderView2, True)

# reset view to fit data
renderView2.ResetCamera()

# hide data in view
Hide(data_indexxdmf_1, renderView2)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2)

# show data in view
data_indexxdmf_1Display = Show(data_indexxdmf_1, renderView2)

# show color bar/color legend
data_indexxdmf_1Display.SetScalarBarVisibility(renderView2, True)

# reset view to fit data
renderView2.ResetCamera()

# reset view to fit data
renderView2.ResetCamera()

# current camera placement for renderView1
renderView1.CameraPosition = [0.10000000000000007, 0.10000000000000007, 0.5133267337658716]
renderView1.CameraFocalPoint = [0.10000000000000007, 0.10000000000000007, 0.0]
renderView1.CameraViewUp = [-1.0, 0.0, 0.0]
renderView1.CameraParallelScale = 0.13285873505887877

# current camera placement for renderView2
renderView2.CameraPosition = [0.10000000000000007, 0.10000000000000007, -0.5657150735620529]
renderView2.CameraFocalPoint = [0.10000000000000007, 0.10000000000000007, 0.0]
renderView2.CameraViewUp = [1.0, 0.0, 0.0]
renderView2.CameraParallelScale = 0.1473433339938693

# save animation
SaveAnimation(input_save_path, layout1, SaveAllViews=1,
    ImageResolution=[1066, 591],
    # FrameWindow=[0, 767], 
    FrameWindow = input_fram,
    # PNG options
    CompressionLevel='9')

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [0.10000000000000007, 0.10000000000000007, 0.5133267337658716]
renderView1.CameraFocalPoint = [0.10000000000000007, 0.10000000000000007, 0.0]
renderView1.CameraViewUp = [-1.0, 0.0, 0.0]
renderView1.CameraParallelScale = 0.13285873505887877

# current camera placement for renderView2
renderView2.CameraPosition = [0.10000000000000007, 0.10000000000000007, -0.5657150735620529]
renderView2.CameraFocalPoint = [0.10000000000000007, 0.10000000000000007, 0.0]
renderView2.CameraViewUp = [1.0, 0.0, 0.0]
renderView2.CameraParallelScale = 0.1473433339938693

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).

