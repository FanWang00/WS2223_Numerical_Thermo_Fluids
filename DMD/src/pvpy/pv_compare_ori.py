# trace generated using paraview version 5.8.0-RC2
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XDMF Reader'
aerobreakup_data_indexxdmf = XDMFReader(FileNames=['/media/overflow/Volume/ALPACA_CWD/SimulationData/L3/WithoutShock/aerobreakup_data_index.xdmf'])
aerobreakup_data_indexxdmf.CellArrayStatus = ['density', 'interface_velocity', 'levelset', 'partition', 'pressure', 'velocity']

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
# renderView1.ViewSize = [1068, 504]

# get layout
layout1 = GetLayout()

# show data in view
aerobreakup_data_indexxdmfDisplay = Show(aerobreakup_data_indexxdmf, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')

# trace defaults for the display properties.
aerobreakup_data_indexxdmfDisplay.Representation = 'Surface'
aerobreakup_data_indexxdmfDisplay.ColorArrayName = ['CELLS', 'density']
aerobreakup_data_indexxdmfDisplay.LookupTable = densityLUT
aerobreakup_data_indexxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
aerobreakup_data_indexxdmfDisplay.SelectOrientationVectors = 'velocity'
aerobreakup_data_indexxdmfDisplay.ScaleFactor = 0.00576
aerobreakup_data_indexxdmfDisplay.SelectScaleArray = 'density'
aerobreakup_data_indexxdmfDisplay.GlyphType = 'Arrow'
aerobreakup_data_indexxdmfDisplay.GlyphTableIndexArray = 'density'
aerobreakup_data_indexxdmfDisplay.GaussianRadius = 0.000288
aerobreakup_data_indexxdmfDisplay.SetScaleArray = [None, '']
aerobreakup_data_indexxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
aerobreakup_data_indexxdmfDisplay.OpacityArray = [None, '']
aerobreakup_data_indexxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
aerobreakup_data_indexxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
aerobreakup_data_indexxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
aerobreakup_data_indexxdmfDisplay.ScalarOpacityFunction = densityPWF
aerobreakup_data_indexxdmfDisplay.ScalarOpacityUnitDistance = 0.001884056898673629

# show color bar/color legend
aerobreakup_data_indexxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# reset view to fit data
renderView1.ResetCamera()

# set scalar coloring
ColorBy(aerobreakup_data_indexxdmfDisplay, ('CELLS', 'velocity', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(densityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
aerobreakup_data_indexxdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
aerobreakup_data_indexxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'velocity'
velocityLUT = GetColorTransferFunction('velocity')

# get opacity transfer function/opacity map for 'velocity'
velocityPWF = GetOpacityTransferFunction('velocity')

# split cell
layout1.SplitVertical(0, 0.5)

# set active view
SetActiveView(None)

# create a new 'XDMF Reader'
berobreakupxdmf = XDMFReader(FileNames=['/media/overflow/Volume/ALPACA_CWD/PreProcess/interp/L3/WithoutShock/berobreakup.xdmf'])
berobreakupxdmf.CellArrayStatus = ['pressure', 'velocityU']

# set active source
SetActiveSource(berobreakupxdmf)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraFocalDisk = 1.0
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1
# uncomment following to set a specific view size
# renderView2.ViewSize = [400, 400]

# show data in view
berobreakupxdmfDisplay = Show(berobreakupxdmf, renderView2, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'velocityU'
velocityULUT = GetColorTransferFunction('velocityU')

# get opacity transfer function/opacity map for 'velocityU'
velocityUPWF = GetOpacityTransferFunction('velocityU')

# trace defaults for the display properties.
berobreakupxdmfDisplay.Representation = 'Surface'
berobreakupxdmfDisplay.ColorArrayName = ['CELLS', 'velocityU']
berobreakupxdmfDisplay.LookupTable = velocityULUT
berobreakupxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
berobreakupxdmfDisplay.SelectOrientationVectors = 'None'
berobreakupxdmfDisplay.ScaleFactor = 0.005707499999999999
berobreakupxdmfDisplay.SelectScaleArray = 'velocityU'
berobreakupxdmfDisplay.GlyphType = 'Arrow'
berobreakupxdmfDisplay.GlyphTableIndexArray = 'velocityU'
berobreakupxdmfDisplay.GaussianRadius = 0.00028537499999999993
berobreakupxdmfDisplay.SetScaleArray = [None, '']
berobreakupxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
berobreakupxdmfDisplay.OpacityArray = [None, '']
berobreakupxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
berobreakupxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
berobreakupxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
berobreakupxdmfDisplay.ScalarOpacityFunction = velocityUPWF
berobreakupxdmfDisplay.ScalarOpacityUnitDistance = 0.0009657443458293173

# add view to a layout so it's visible in UI
AssignViewToLayout(view=renderView2, layout=layout1, hint=2)

# show color bar/color legend
berobreakupxdmfDisplay.SetScalarBarVisibility(renderView2, True)

# reset view to fit data
renderView2.ResetCamera()

# set scalar coloring
ColorBy(berobreakupxdmfDisplay, ('CELLS', 'pressure'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(velocityULUT, renderView2)

# rescale color and/or opacity maps used to include current data range
berobreakupxdmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
berobreakupxdmfDisplay.SetScalarBarVisibility(renderView2, True)

# get color transfer function/color map for 'pressure'
pressureLUT = GetColorTransferFunction('pressure')

# get opacity transfer function/opacity map for 'pressure'
pressurePWF = GetOpacityTransferFunction('pressure')

# current camera placement for renderView1
renderView1.CameraPosition = [0.0288, 0.014399999999999998, 0.05818827484163124]
renderView1.CameraFocalPoint = [0.0288, 0.014399999999999998, 0.00015]
renderView1.CameraParallelScale = 0.032199728259722935

# current camera placement for renderView2
renderView2.CameraPosition = [0.028799999999999992, 0.014399999999999996, 0.05740324833831881]
renderView2.CameraFocalPoint = [0.028799999999999992, 0.014399999999999996, 0.0]
renderView2.CameraParallelScale = 0.031847414533993174

# save animation
SaveAnimation('/home/overflow/Desktop/test.png', layout1, SaveAllViews=1,
    ImageResolution=[1068, 469],
    FrameWindow=[0, 2])

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [0.0288, 0.014399999999999998, 0.05818827484163124]
renderView1.CameraFocalPoint = [0.0288, 0.014399999999999998, 0.00015]
renderView1.CameraParallelScale = 0.032199728259722935

# current camera placement for renderView2
renderView2.CameraPosition = [0.028799999999999992, 0.014399999999999996, 0.05740324833831881]
renderView2.CameraFocalPoint = [0.028799999999999992, 0.014399999999999996, 0.0]
renderView2.CameraParallelScale = 0.031847414533993174

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).