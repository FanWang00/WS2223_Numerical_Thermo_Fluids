#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:15:13 2020

@author: fan
"""

# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
import argparse

## input variables
# input_file = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/DataSet/L3/WithShock/aerobreakup_data_index.xdmf'
# input_file = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/PostProcess/L3/dmd/DMDrecons_test/aerobreakup_data.xdmf'
# actual_dis_item = 'pressure'
# input_CellArr = ['pressure', 'velocityU']
# input_save_path = '/media/fan/SeagateExpansionDrive/ALPACA_CWD/test/test.png'
# input_frame = [0, 5]


# default_dis_item = ['velocityU', 'density']
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="input file path")
# parser.add_argument("input_CellArr", help="input CellArr")
parser.add_argument("actual_dis_item", help="input item to png")
parser.add_argument("input_save_path", help="save path")
parser.add_argument("input_frame", help="frame to ouput", type=int, nargs=2)
args = parser.parse_args()

input_file = args.input_file
actual_dis_item = args.actual_dis_item
input_save_path = args.input_save_path
input_frame =args.input_frame

print(f"input file: {input_file}")
print(f"output item: {actual_dis_item}")
print(f"save_path: {input_save_path}")
print(f"frame: {input_frame}")
## default
default_dis_item = 'velocityU'
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XDMF Reader'
aerobreakup_data_indexxdmf = XDMFReader(FileNames=input_file)
aerobreakup_data_indexxdmf.CellArrayStatus = ['density', 'interface_velocity', 
                                              'levelset', 'partition', 
                                              'pressure', 'velocity', 'velocityU']

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# set active source
SetActiveSource(aerobreakup_data_indexxdmf)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1100, 598]

# get layout
layout1 = GetLayout()

# show data in view
aerobreakup_data_indexxdmfDisplay = Show(aerobreakup_data_indexxdmf, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'velocityU'
velocityULUT = GetColorTransferFunction('velocityU')

# get opacity transfer function/opacity map for 'velocityU'
velocityUPWF = GetOpacityTransferFunction('velocityU')

# trace defaults for the display properties.
aerobreakup_data_indexxdmfDisplay.Representation = 'Surface'
aerobreakup_data_indexxdmfDisplay.ColorArrayName = ['CELLS', default_dis_item]
aerobreakup_data_indexxdmfDisplay.LookupTable = velocityULUT
aerobreakup_data_indexxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
aerobreakup_data_indexxdmfDisplay.SelectOrientationVectors = 'None'
aerobreakup_data_indexxdmfDisplay.ScaleFactor = 0.005707499999999999
aerobreakup_data_indexxdmfDisplay.SelectScaleArray = default_dis_item
aerobreakup_data_indexxdmfDisplay.GlyphType = 'Arrow'
aerobreakup_data_indexxdmfDisplay.GlyphTableIndexArray = default_dis_item
aerobreakup_data_indexxdmfDisplay.GaussianRadius = 0.00028537499999999993
aerobreakup_data_indexxdmfDisplay.SetScaleArray = [None, '']
aerobreakup_data_indexxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
aerobreakup_data_indexxdmfDisplay.OpacityArray = [None, '']
aerobreakup_data_indexxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
aerobreakup_data_indexxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
aerobreakup_data_indexxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
aerobreakup_data_indexxdmfDisplay.ScalarOpacityFunction = velocityUPWF
aerobreakup_data_indexxdmfDisplay.ScalarOpacityUnitDistance = 0.0009657443458293173


# show color bar/color legend
aerobreakup_data_indexxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# reset view to fit data
renderView1.ResetCamera()

# set scalar coloring
ColorBy(aerobreakup_data_indexxdmfDisplay, ('CELLS', actual_dis_item, 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityULUT, renderView1)

# rescale color and/or opacity maps used to include current data range
aerobreakup_data_indexxdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
aerobreakup_data_indexxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction(actual_dis_item)

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction(actual_dis_item)

#change interaction mode for render view
renderView1.InteractionMode = '2D'

# get the material library
materialLibrary1 = GetMaterialLibrary()

# #change interaction mode for render view
# renderView1.InteractionMode = '3D'

# current camera placement for renderView1
renderView1.CameraPosition = [0.0288, 0.0144, 0.12456019650222538]
renderView1.CameraFocalPoint = [0.0288, 0.0144, 0.00015]
renderView1.CameraParallelScale = 0.03219972825972294

# save animation
SaveAnimation(input_save_path, renderView1, ImageResolution=[1560, 867],
    FrameWindow=input_frame, 
    # PNG options
    CompressionLevel='4')

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [0.0288, 0.0144, 0.12456019650222538]
renderView1.CameraFocalPoint = [0.0288, 0.0144, 0.00015]
renderView1.CameraParallelScale = 0.03219972825972294

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).