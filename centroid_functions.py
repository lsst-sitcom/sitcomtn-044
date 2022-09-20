import yaml
import os
import numpy as np
from scipy.ndimage import rotate
import centroid_functions as func 

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
from matplotlib import rcParams 

from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.cwfs.Algorithm import Algorithm
from lsst.ts.wep.cwfs.CompensableImage import CompensableImage
from lsst.ts.wep.Utility import (
    getConfigDir,
    DonutTemplateType,
    DefocalType,
    CamType,
    getCamType,
    getDefocalDisInMm,
    CentroidFindType
)

from lsst.ts.wep.task.DonutStamps import DonutStamp, DonutStamps
from lsst.ts.wep.task.EstimateZernikesCwfsTask import (
    EstimateZernikesCwfsTask,
    EstimateZernikesCwfsTaskConfig,
)

from lsst.daf import butler as dafButler
import lsst.afw.cameraGeom as cameraGeom
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, FIELD_ANGLE
from lsst.geom import Point2D

from astropy.visualization import ZScaleInterval


def get_butler_stamps(repo_dir, instrument='LSSTComCam', iterN=0, detector="R22_S01",
                     dataset_type = 'donutStampsExtra', collection=''):
    ''' Get from the butler gen3 repo donut stamps and donut catalog-
    the data products originating from the AOS simulation.
    
    Parameters:
    -----------
    repo_dir: str, path to a directory containing the butler.yaml file.
    instrument: str, a name of the instrument accepted by the butler,
        for instance 'LSSTCam', 'LATISS', or 'LSSTComCam' (default).
    iterN: int, iteration number for which to gather the donuts
        (by default 0). 
    detector: str, name of the detector for which donuts were simulated
        (by default "R22_S01", which is one of the nine ComCam rafts)
    dataset_type: str, type of dataset accepted by the butler. By 
        default, it is for extra-focal stamps, i.e. 'donutStampsExtra'
    collection: str, if not provided then the name is constructed 
        given the iterN, as `ts_phosim_90060{iterN}`, which is the
        default collection naming in the imgCloseLoop.py
                     
    
    Returns:
    --------
    donutStamps: lsst.afw.image.maskedImage.MaskedImage, 
        collection of postage stamps with additional metadata.
    donutCatalog: pandas.core.frame.DataFrame, 
        an array of detected donuts, containing
        coord_ra, coord_dec, centroid_x, centroid_y, source_flux
    
    
    '''

    butler = dafButler.Butler(repo_dir)
    registry = butler.registry
    if collection == '':
        collection=f'ts_phosim_90060{iterN}1'
    dataId0 = dict(instrument=instrument)
    dataset = next(iter(butler.registry.queryDatasets(
                            datasetType='postISRCCD',
                            dataId=dataId0, 
                            collections=[collection]  )
                       ))
    expN = dataset.dataId["exposure"]
    # construct a dataId for zernikes and donut catalog:
    # switch exposure to visit 
    
    dataId = {'detector':detector, 'instrument':instrument,
              'visit':expN}

    donutStamps = butler.get(dataset_type, 
                              dataId=dataId, 
                              collections=[collection])  
    
    donutCatalog = butler.get('donutCatalog', 
                              dataId=dataId, 
                              collections=[collection]) 
    return donutStamps, donutCatalog


def get_butler_image(repo_dir,instrument='LSSTComCam', iterN=0, detector="R22_S01",
                     collection=''):
    ''' Get from the butler gen3 repo the postISR image
    from the AOS simulation.
    
    Parameters:
    -----------
    repo_dir: str, path to a directory containing the butler.yaml file.
    instrument: str, a name of the instrument accepted by the butler,
        for instance 'LSSTCam', 'LATISS', or 'LSSTComCam' (default).
    iterN: int, iteration number for which to gather the donuts
        (by default 0). 
    detector: str, name of the detector for which donuts were simulated
        (by default "R22_S01", which is one of the nine ComCam rafts)
   
    collection: str, if not provided then the name is constructed 
        given the iterN, as `ts_phosim_90060{iterN}`, which is the
        default collection naming in the imgCloseLoop.py
                     
    
    Returns:
    --------
    postIsr: lsst.afw.image.Exposure
        Exposure with the donut image.
    
    '''
    butler = dafButler.Butler(repo_dir)
    registry = butler.registry
    if collection == '':
        collection=f'ts_phosim_90060{iterN}1'
    dataId0 = dict(instrument=instrument)
    dataset = next(iter(butler.registry.queryDatasets(
                            datasetType='postISRCCD',
                            dataId=dataId0, 
                            collections=[collection]  )
                       ))
    expN = dataset.dataId["exposure"]
    dataId = {'detector':detector, 'instrument':instrument,
          'exposure':expN}
    postIsr = butler.get('postISRCCD',dataId=dataId, 
                          collections=[collection])
    return postIsr

def offset_centroid_square_grid(repo_dir ,
                                instrument = 'LSSTCam',
                                collection='ts_phosim_9006070',
                                sensors=['R00'],
                                index_increase='both', 
                                experiment_index = 1,
                                out_dir = 'DM-33104'):
    ''' Steps to vary centroid position in a rectangular grid,
    given the corner sensors simulation data.
    All output is stored as .npy files, hence no direct output
    is returned.


    Parameters:
    ------------
    repo_dir: str, path to a directory containing the butler.yaml file. 
    instrument: str, a name of the instrument accepted by the butler,
        for instance 'LSSTCam' (default).
    collection: str, if not provided then the name is constructed 
        given the iterN, as `ts_phosim_90060{iterN}`, which is the
        default collection naming in the imgCloseLoop.py 'ts_phosim_9006070'
    sensors: list of sensor names, only first half needed, eg. ['R00'] (default). 
    index_increase: str, accepted values are
        - 'both' increase the index of both extra and intra-focal donuts,  
           i.e. fit donutIntra[i] with donutExtra[i],
        - 'intra': increase only the index of intra-focal donuts, 
           keeping the extra-focal fixed, eg. pair donutIntra[i] 
           with donutExtra[0]
    experiment_index: int, an integer used to distinguish different types of 
        experiment (default is 1).
    out_dir: str, name of the output directory, by default 'DM-33104'.
                                
    Returns:
    --------
    None


    '''

    #################################
    ### STEP 1 :  Load the stamps
    #################################

    # iterate over all sensors
    for sensor in sensors: 

        print(f'Fitting {sensor}')
        donutStampsExtra, extraFocalCatalog = get_butler_stamps(repo_dir,
                                              instrument=instrument, 
                                              iterN=0, detector=f"{sensor}_SW0",
                                              dataset_type = 'donutStampsExtra', 
                                              collection=collection)

        donutStampsIntra, intraFocalCatalog = get_butler_stamps(repo_dir,
                                              instrument=instrument, 
                                              iterN=0, detector=f"{sensor}_SW1",
                                              dataset_type = 'donutStampsIntra', 
                                              collection=collection)

        extraImage = get_butler_image(repo_dir,instrument=instrument, 
                                      iterN=0, detector=f"{sensor}_SW0",
                                      collection=collection)
        # get the pixel scale from exposure to convert from pixels 
        # to arcsec to degrees
        pixelScale = extraImage.getWcs().getPixelScale().asArcseconds()


        configDir = getConfigDir()
        instDir = os.path.join(configDir, "cwfs", "instData")
        algoDir = os.path.join(configDir, "cwfs", "algo")

        # now I follow parts of 
        # wfEsti = WfEstimator(instDir, algoDir)
        inst = Instrument(instDir)
        algo = Algorithm(algoDir)

        # now I get the camera type and defocal distance
        # as inside estimateZernikes()
        # instName gets specified in the config,
        # and does not get overridden for CWFS
        instName='lsst'    
        camType = getCamType(instName)
        defocalDisInMm = getDefocalDisInMm(instName)

        # now I follow parts of 
        # wfEsti.config
        opticalModel = 'offAxis'
        sizeInPix = 160
        inst.config(camType,sizeInPix, announcedDefocalDisInMm=defocalDisInMm)

        # choose the solver for the algorithm 
        solver = 'exp' # or fft 
        debugLevel=0  # 1 to 3 
        algo.config(solver, inst, debugLevel=debugLevel)

        centroidFindType = CentroidFindType.RandomWalk
        imgIntra = CompensableImage(centroidFindType=centroidFindType)
        imgExtra = CompensableImage(centroidFindType=centroidFindType)

        #######################################
        ### STEP 2 :  Loop over donut pairs 
        #######################################

        if index_increase == 'both':
            irange = min(len(donutStampsExtra),len(donutStampsIntra))
        elif index_increase == 'intra':
            irange = len(donutStampsIntra)

        for i in range(irange): 
            if index_increase == 'both':
                i_in = i
                i_ex = i
                donutExtra = donutStampsExtra[i_ex]
                donutIntra = donutStampsIntra[i_in]
            elif index_increase == 'intra':
                i_in = i
                i_ex = 0
                donutExtra = donutStampsExtra[i_ex]
                donutIntra = donutStampsIntra[i_in]

            print(f'\n Fitting sensor {sensor}, donutIntra {i_in} with donutExtra {i_ex} ')
            
            # obtain the original fieldXY position attached to each donut stamp 
            fieldXYExtra = donutExtra.calcFieldXY()
            fieldXYIntra = donutIntra.calcFieldXY()

            #######################################
            ### STEP 2 A : fit baseline (no offset)
            #######################################

            camera = donutExtra.getCamera()
            detectorExtra = camera.get(donutExtra.detector_name)
            detectorIntra = camera.get(donutIntra.detector_name)

            # Rotate any sensors that are not lined up with the focal plane.
            eulerZExtra = -detectorExtra.getOrientation().getYaw().asDegrees()
            eulerZIntra = -detectorIntra.getOrientation().getYaw().asDegrees()

            # Now parts of wfEsti.setImg, which inherits .setImg 
            # method from CompensableImage
            imgExtra.setImg(fieldXYExtra,
                            DefocalType.Extra,
                            image=rotate(donutExtra.stamp_im.getImage().getArray(), 
                                eulerZExtra).T)

            imgIntra.setImg(fieldXYIntra,
                            DefocalType.Intra,
                            image=rotate(donutIntra.stamp_im.getImage().getArray(), 
                                eulerZIntra).T)

            # wfEsti.reset() inherits the .reset method of Algorithm.py: 
            algo.reset()

            # wfEsti.calWfsErr() after checking for image size 
            # (both need to be squares)
            # calls the Algorithm.py
            tol = 1e-3 # explicitly set the tolerance level
            algo.runIt(imgIntra,imgExtra, opticalModel, tol=tol)
            zk_no_offset = algo.getZer4UpInNm()
            
            # store the compensable image, as well as the zernikes
            baseline = {}
            baseline['imgIntra'] = imgIntra
            baseline['imgExtra'] = imgExtra
            baseline['zks'] = zk_no_offset
            
            fname = f'exp-{experiment_index}_{sensor}_baseline_extra_{i_ex}_intra_{i_in}.npy'
            fpath = os.path.join(out_dir, fname)
            np.save(fpath, baseline)
            print(f'\nSaved the baseline fit as {fpath}')
            

            #######################################
            ### STEP 2 B : fit grid of dx, dy offset 
            #######################################

            ## First applying change to the extra-focal mask, and then to the intra-focal mask
            for defocal in ['intra','extra']: # or extra 
                print(f'Fitting {sensor} donuts {i_in} {i_ex} shifting {defocal} centroid ')
                zk0 = zk_no_offset

                dxInPixels = np.linspace(-100,100,20)
                dyInPixels = dxInPixels

                dxInDegrees = (dxInPixels* pixelScale) / 3600
                dyInDegrees = (dyInPixels* pixelScale) / 3600

                results = {}
                j =0 
                
                for dxDeg in dxInDegrees:
                    for dyDeg in dyInDegrees:
                        
                        # calculate radial offset
                        radialOffsetDegrees = np.sqrt(dxDeg**2.0+dyDeg**2.0)

                        # convert the x,y offset of the mask from degrees to pixels via pixelScale
                        radialOffsetPixels = ( radialOffsetDegrees * 3600 )/pixelScale

                        dxPx = (dxDeg * 3600) / pixelScale
                        dyPx = (dyDeg * 3600) / pixelScale
                        
                        results[j] = {'dxDeg':dxDeg, 'dyDeg':dyDeg, 
                                      'dxPx':dxPx, 'dyPx':dyPx, 
                                      'drDeg':radialOffsetDegrees,
                                      'drPx':radialOffsetPixels, 
                                      }

                         # do the fit 
                        if defocal == 'extra':
                            fieldXYExtraUpd = (fieldXYExtra[0]+dxDeg, fieldXYExtra[1]+dyDeg)
                            fieldXYIntraUpd = fieldXYIntra

                        elif defocal =='intra':
                            fieldXYExtraUpd = fieldXYExtra
                            fieldXYIntraUpd = (fieldXYIntra[0]+dxDeg, fieldXYIntra[1]+dyDeg)

                        imgExtra.setImg(fieldXYExtraUpd,
                                        DefocalType.Extra,
                                        image=rotate(donutExtra.stamp_im.getImage().getArray(), eulerZExtra).T)

                        imgIntra.setImg(fieldXYIntraUpd,
                                        DefocalType.Intra,
                                        image=rotate(donutIntra.stamp_im.getImage().getArray(), eulerZIntra).T)

                        # right after we set the image, the compensable image mask is empty - 
                        # it is only calculated in Algorithm.py, L694
                        maskScalingFactorLocal = 1
                        boundaryT = algo.getBoundaryThickness()
                        imgIntra.makeMask(inst, opticalModel, boundaryT, maskScalingFactorLocal)
                        imgExtra.makeMask(inst, opticalModel, boundaryT, maskScalingFactorLocal)

                        # now wfEsti.reset() calls the .reset() method of Algorithm.py : 
                        algo.reset()

                        # wfEsti.calWfsErr() after checking for image size (both need to be squares)
                        # simply calls algo:
                        tol = 1e-3 # explicitly set the tolerance level ( this is default )
                        algo.runIt(imgIntra, imgExtra, opticalModel, tol=tol)
                        zk = algo.getZer4UpInNm()
                        
                        # calculate diffMax, diffRms
                        diffMax = np.max(np.abs(zk - zk0))
                        diffRms = np.sqrt(np.sum(np.abs(zk - zk0) ** 2) / len(zk))

                        # store the results 
                        results[j]['diffMax'] = diffMax
                        results[j]['diffRms'] = diffRms
                        results[j]['zk'] = zk
                        results[j]['imgExtraFieldXY'] = imgExtra.getFieldXY()
                        results[j]['imgIntraFieldXY'] = imgIntra.getFieldXY()

                        # increase the grid counter 
                        j += 1 
                        print(j, dxPx, dyPx, diffRms, )
                fname = f'exp-{experiment_index}_{sensor}_square_grid_extra_{i_ex}_intra_{i_in}.npy'
                fpath = os.path.join(out_dir, fname)
                np.save(fpath, results)
                print(f"saved {fname}")

                
def offset_centroid_circle(repo_dir ,
                           instrument = 'LSSTCam',
                           collection='ts_phosim_9006070',
                           sensors=['R00'],
                           index_increase='both', 
                           experiment_index = 2,
                           out_dir = 'DM-33104'
                           ):
    ''' Steps to vary centroid position radially until  a diffRms 
        threshold is reached, given the corner sensors simulation data.
    All output is stored as .npy files in a designated output directory.


    Parameters:
    ------------
    repo_dir: str, path to a directory containing the butler.yaml file. 
    instrument: str, a name of the instrument accepted by the butler,
        for instance 'LSSTCam' (default).
    collection: str, if not provided then the name is constructed 
        given the iterN, as `ts_phosim_90060{iterN}`, which is the
        default collection naming in the imgCloseLoop.py 'ts_phosim_9006070'
    sensors: list of sensor names, only first half needed, eg. ['R00'] (default). 
    index_increase: str, accepted values are
        - 'both' increase the index of both extra and intra-focal donuts,  
           i.e. fit donutIntra[i] with donutExtra[i],
        - 'intra': increase only the index of intra-focal donuts, 
           keeping the extra-focal fixed, eg. pair donutIntra[i] 
           with donutExtra[0]
    experiment_index: int, an integer used to distinguish different types of 
        experiment (default is 2).
    out_dir: str, name of the output directory, by default 'DM-33104'.
                                
    Returns:
    --------
    None


    '''

    #################################
    ### STEP 1 :  Load the stamps
    #################################

    # iterate over all sensors
    for sensor in sensors: 

        print(f'Fitting {sensor}')
        donutStampsExtra, extraFocalCatalog = get_butler_stamps(repo_dir,
                                              instrument=instrument, 
                                              iterN=0, detector=f"{sensor}_SW0",
                                              dataset_type = 'donutStampsExtra', 
                                              collection=collection)

        donutStampsIntra, intraFocalCatalog = get_butler_stamps(repo_dir,
                                              instrument=instrument, 
                                              iterN=0, detector=f"{sensor}_SW1",
                                              dataset_type = 'donutStampsIntra', 
                                              collection=collection)

        extraImage = get_butler_image(repo_dir,instrument=instrument, 
                                      iterN=0, detector=f"{sensor}_SW0",
                                      collection=collection)
        # get the pixel scale from exposure to convert from pixels 
        # to arcsec to degrees
        pixelScale = extraImage.getWcs().getPixelScale().asArcseconds()


        configDir = getConfigDir()
        instDir = os.path.join(configDir, "cwfs", "instData")
        algoDir = os.path.join(configDir, "cwfs", "algo")

        # now I follow parts of 
        # wfEsti = WfEstimator(instDir, algoDir)
        inst = Instrument(instDir)
        algo = Algorithm(algoDir)

        # now I get the camera type and defocal distance
        # as inside estimateZernikes()
        # instName gets specified in the config,
        # and does not get overridden for CWFS
        instName='lsst'    
        camType = getCamType(instName)
        defocalDisInMm = getDefocalDisInMm(instName)

        # now I follow parts of 
        # wfEsti.config
        opticalModel = 'offAxis'
        sizeInPix = 160
        inst.config(camType,sizeInPix, announcedDefocalDisInMm=defocalDisInMm)

        # choose the solver for the algorithm 
        solver = 'exp' # or fft 
        debugLevel=0  # 1 to 3 
        algo.config(solver, inst, debugLevel=debugLevel)

        centroidFindType = CentroidFindType.RandomWalk
        imgIntra = CompensableImage(centroidFindType=centroidFindType)
        imgExtra = CompensableImage(centroidFindType=centroidFindType)

        #######################################
        ### STEP 2 :  Loop over donut pairs 
        #######################################

        if index_increase == 'both':
            irange = min(len(donutStampsExtra),len(donutStampsIntra))
        elif index_increase == 'intra':
            irange = len(donutStampsIntra)

        for i in range(irange): 
            if index_increase == 'both':
                i_in = i
                i_ex = i
                donutExtra = donutStampsExtra[i_ex]
                donutIntra = donutStampsIntra[i_in]
            elif index_increase == 'intra':
                i_in = i
                i_ex = 0
                donutExtra = donutStampsExtra[i_ex]
                donutIntra = donutStampsIntra[i_in]

            print(f'\n Fitting sensor {sensor}, donutIntra {i_in} with donutExtra {i_ex} ')
            
            # obtain the original fieldXY position attached to each donut stamp 
            fieldXYExtra = donutExtra.calcFieldXY()
            fieldXYIntra = donutIntra.calcFieldXY()

            #######################################
            ### STEP 2 A : fit baseline (no offset)
            #######################################

            camera = donutExtra.getCamera()
            detectorExtra = camera.get(donutExtra.detector_name)
            detectorIntra = camera.get(donutIntra.detector_name)

            # Rotate any sensors that are not lined up with the focal plane.
            eulerZExtra = -detectorExtra.getOrientation().getYaw().asDegrees()
            eulerZIntra = -detectorIntra.getOrientation().getYaw().asDegrees()

            # Now parts of wfEsti.setImg, which inherits .setImg 
            # method from CompensableImage
            imgExtra.setImg(fieldXYExtra,
                            DefocalType.Extra,
                            image=rotate(donutExtra.stamp_im.getImage().getArray(), 
                                eulerZExtra).T)

            imgIntra.setImg(fieldXYIntra,
                            DefocalType.Intra,
                            image=rotate(donutIntra.stamp_im.getImage().getArray(), 
                                eulerZIntra).T)

            # wfEsti.reset() inherits the .reset method of Algorithm.py: 
            algo.reset()

            # wfEsti.calWfsErr() after checking for image size 
            # (both need to be squares)
            # calls the Algorithm.py
            tol = 1e-3 # explicitly set the tolerance level
            algo.runIt(imgIntra,imgExtra, opticalModel, tol=tol)
            zk_no_offset = algo.getZer4UpInNm()
            
            # store the compensable image, as well as the zernikes
            baseline = {}
            baseline['imgIntra'] = imgIntra
            baseline['imgExtra'] = imgExtra
            baseline['zks'] = zk_no_offset
            
            fname = f'exp-{experiment_index}_{sensor}_baseline_extra_{i_ex}_intra_{i_in}.npy'
            fpath = os.path.join(out_dir, fname)
            np.save(fpath, baseline)
            print(f'\nSaved the baseline fit as {fname}')
            

            #######################################
            ### STEP 2 B : fit grid of dx, dy offset 
            #######################################

            ## First applying change to the extra-focal mask, and then to the intra-focal mask
            for defocal in ['intra','extra']: # or extra 
                print(f'Fitting {sensor} donuts {i_in} {i_ex} shifting {defocal} centroid ')
                zk0 = zk_no_offset

                # starting boundary conditions 
                thetaRad = 0 # radians
                dthetaRad = 0.2 # radians ~ 11.45 deg 
                drPxStep = 10 # pixels 
                diffRmsThresh = 1.0 # nm 

                results = {}

                j=0
                # iterate over a range of angles 
                for thetaRad in np.arange(0,2*np.pi, dthetaRad):

                    # always start from 0 radius ... 
                    drPx = 0 # px 
                    diffRms = 0 # nm 

                    # increase radius until threshold is reached 
                    while diffRms < diffRmsThresh:

                        dxPx = drPx*np.cos(thetaRad)
                        dyPx = drPx*np.sin(thetaRad)

                        dxDeg = (dxPx * pixelScale) / 3600.
                        dyDeg = (dyPx * pixelScale) / 3600.

                        results[j]={'dxPx':dxPx, 'dyPx':dyPx, 'dxDeg':dxDeg, 'dyDeg':dyDeg,
                                    'drPx':drPx, }

                         # do the fit 
                        if defocal == 'extra':
                            fieldXYExtraUpd = (fieldXYExtra[0]+dxDeg, 
                                               fieldXYExtra[1]+dyDeg)
                            fieldXYIntraUpd = fieldXYIntra

                        elif defocal =='intra':
                            fieldXYExtraUpd = fieldXYExtra
                            fieldXYIntraUpd = (fieldXYIntra[0]+dxDeg, 
                                               fieldXYIntra[1]+dyDeg)

                        imgExtra.setImg(fieldXYExtraUpd,
                                        DefocalType.Extra,
                                        image=rotate(donutExtra.stamp_im.getImage().getArray(), 
                                                     eulerZExtra).T)

                        imgIntra.setImg(fieldXYIntraUpd,
                                        DefocalType.Intra,
                                        image=rotate(donutIntra.stamp_im.getImage().getArray(), 
                                                     eulerZIntra).T)

                        # right after we set the image, the compensable image mask is empty - 
                        # it is only calculated in Algorithm.py, L694
                        maskScalingFactorLocal = 1
                        boundaryT = algo.getBoundaryThickness()
                        imgIntra.makeMask(inst, opticalModel, boundaryT, 
                                          maskScalingFactorLocal)
                        imgExtra.makeMask(inst, opticalModel, boundaryT, 
                                          maskScalingFactorLocal)

                        # now wfEsti.reset() calls the .reset() method of Algorithm.py : 
                        algo.reset()

                        # wfEsti.calWfsErr() after checking for image size (both need to be squares)
                        # simply calls algo:
                        tol = 1e-3 # explicitly set the tolerance level ( this is default )
                        algo.runIt(imgIntra, imgExtra, opticalModel, tol=tol)
                        zk = algo.getZer4UpInNm()
                        
                        # calculate diffMax, diffRms
                        diffMax = np.max(np.abs(zk - zk0))
                        diffRms = np.sqrt(np.sum(np.abs(zk - zk0) ** 2) / len(zk))

                        # store the results 
                        results[j]['diffMax'] = diffMax
                        results[j]['diffRms'] = diffRms
                        results[j]['zk'] = zk
                        results[j]['imgExtraFieldXY'] = imgExtra.getFieldXY()
                        results[j]['imgIntraFieldXY'] = imgIntra.getFieldXY()
                        
                        # increase the grid counter 
                        j += 1 
                    print(j-1, drPx, thetaRad, diffRms) 
                    print('Threshold reached at radius ', drPx)  
                # save the results
                fname = f'exp-{experiment_index}_{sensor}_circle_extra_{i_ex}_intra_{i_in}.npy'
                fpath = os.path.join(out_dir, fname)
                print(f"saved {fpath}")
                np.save(fpath, results)
            
def get_nx_ny(x,y,dr):
    ''' Obtain new x, new y coordinate, 
    shifted by dr along radial direction.
    '''
    tan = x / y # y cannot be 0 
    dx = dr * (1 + tan**2.0)**(-0.5)
    dy = dr * (tan**2.0 / (1 + tan**2.0))**(0.5)

    # this ensures we shift away or toward the focal plane,
    # depending on dr and which quadrant we are in 
    nx = x - np.sign(x)*dx
    ny = y - np.sign(y)*dy
    
    return nx, ny 

def offset_centroid_radially(repo_dir,
                             instrument='LSSTCam', 
                             collection='ts_phosim_9006000',
                             sensors=['R00'],
                             experiment_index=5,
                             drDegMin=-0.3,
                             drDegMax=0.3,
                             drPxMin=-50,
                             drPxMax=50,
                             nGrid=1000,
                             out_dir = 'DM-33104'
                             ):
    ''' Offset centroid only in radial direction,
    given the corner sensors simulation data. 

    Parameters:
    ------------
    repo_dir: str, path to a directory containing the butler.yaml file. 
    instrument: str, a name of the instrument accepted by the butler,
        for instance 'LSSTCam' (default).
    collection: str, if not provided then the name is constructed 
        given the iterN, as `ts_phosim_90060{iterN}`, which is the
        default collection naming in the imgCloseLoop.py 'ts_phosim_9006070'
    sensors: list of sensor names, only first half needed, eg. ['R00'] (default). 
    index_increase: str, accepted values are
        - 'both' increase the index of both extra and intra-focal donuts,  
           i.e. fit donutIntra[i] with donutExtra[i],
        - 'intra': increase only the index of intra-focal donuts, 
           keeping the extra-focal fixed, eg. pair donutIntra[i] 
           with donutExtra[0]
    experiment_index: int, an integer used to distinguish different types of 
        experiment (default is 5).
    drDegMin: float, radial offset away from the donut 
            centroid in degrees.  It is assumed to be smaller than zero, 
            since the shift axis points towards the focal plane. 
            If None, then using the value in pixels (default: -0.3 deg).
    drDegMax: float, radial offset away from the donut centroid in degrees.
            It is assumed to be larger than zero, 
            since the shift axis points towards the focal plane. 
            If None, then using the value in pixels (default: 0.3 deg).
    drPxMin: float, radial offset away from the donut 
            in pixels. It is assumed to be smaller than zero, 
            since the shift axis points towards the focal plane. 
            It is converted to degrees given the plate scale
            (default: -50 px).
    drPxMax: float, radial offset away from the donut 
            in pixels. It is assumed to be larger than zero, 
            since the shift axis points towards the focal plane. 
            It is converted to degrees given the plate scale
            (default: +50 px).
    nGrid: int, size of the computational space spanned by 
            (drDegMin, drDegMax), with regular intervals
    out_dir: str, name of the output directory, by default 'DM-33104'.
    
    Returns:
    --------
    None

    '''

    #################################
    ### STEP 1 :  Load the stamps
    #################################
    for sensor in sensors:
        print(f'Fitting {sensor}')
        donutStampsExtra, extraFocalCatalog = get_butler_stamps(repo_dir,instrument=instrument, 
                                              iterN=0, detector=f"{sensor}_SW0",
                                              dataset_type = 'donutStampsExtra', 
                                              collection=collection)

        donutStampsIntra, intraFocalCatalog = get_butler_stamps(repo_dir,instrument=instrument, 
                                              iterN=0, detector=f"{sensor}_SW1",
                                              dataset_type = 'donutStampsIntra', 
                                              collection=collection)

        extraImage = get_butler_image(repo_dir,instrument=instrument, 
                              iterN=0, detector=f"{sensor}_SW0",
                              collection=collection)
        # get the pixel scale from exposure to convert from pixels to arcsec to degrees
        pixelScale = extraImage.getWcs().getPixelScale().asArcseconds()


        # Given pixel scale, convert drPxMin, drPxMax to degrees
        if drDegMin is None:
            drDegMin = (drPxMin * pixelScale) / 3600. 
            print('Converting px to deg using pixel scale')
            print(f'{drPxMin} px is {drDegMin} degrees')
        if drDegMax is None:
            drDegMax = (drPxMax * pixelScale) / 3600. 
            print('Converting px to deg using pixel scale')
            print(f'{drPxMax} px is {drDegMax} degrees')
        
        configDir = getConfigDir()
        instDir = os.path.join(configDir, "cwfs", "instData")
        algoDir = os.path.join(configDir, "cwfs", "algo")

        # now I follow parts of 
        # wfEsti = WfEstimator(instDir, algoDir)
        inst = Instrument(instDir)
        algo = Algorithm(algoDir)

        # now I get the camera type and defocal distance
        # as inside estimateZernikes()
        # instName gets specified in the config,
        # and does not get overridden for CWFS
        instName='lsst'    
        camType = getCamType(instName)
        defocalDisInMm = getDefocalDisInMm(instName)

        # now I follow parts of 
        # wfEsti.config 
        opticalModel = 'offAxis'
        sizeInPix = 160
        inst.config(camType,sizeInPix, announcedDefocalDisInMm=defocalDisInMm)

        # choose the solver for the algorithm 
        solver = 'exp' # or fft
        debugLevel = 0 # 1 to 3 
        algo.config(solver, inst, debugLevel=debugLevel)

        centroidFindType = CentroidFindType.RandomWalk
        imgIntra = CompensableImage(centroidFindType=centroidFindType)
        imgExtra = CompensableImage(centroidFindType=centroidFindType)


        #######################################
        ### STEP 2 :  Loop over donut pairs 
        #######################################

        for i in range(len(donutStampsIntra)):
            donutExtra = donutStampsExtra[0]
            donutIntra = donutStampsIntra[i]
            print(f'\n Fitting sensor {sensor}, donutIntra {i}, donutExtra 0 ')

            fieldXYExtra = donutExtra.calcFieldXY()
            fieldXYIntra = donutIntra.calcFieldXY()

            #######################################
            ### STEP 2 A : fit baseline (no offset)
            #######################################

            camera = donutExtra.getCamera()
            detectorExtra = camera.get(donutExtra.detector_name)
            detectorIntra = camera.get(donutIntra.detector_name)

            # Rotate any sensors that are not lined up with the focal plane.
            eulerZExtra = -detectorExtra.getOrientation().getYaw().asDegrees()
            eulerZIntra = -detectorIntra.getOrientation().getYaw().asDegrees()

            # Now parts of wfEsti.setImg, which inherits .setImg 
            # method from CompensableImage
            imgExtra.setImg(fieldXYExtra,
                            DefocalType.Extra,
                            image=rotate(donutExtra.stamp_im.getImage().getArray(), eulerZExtra).T)

            imgIntra.setImg(fieldXYIntra,
                            DefocalType.Intra,
                            image=rotate(donutIntra.stamp_im.getImage().getArray(), eulerZIntra).T)

            # wfEsti.reset() inherits the .reset method of Algorithm.py: 
            algo.reset()

            # wfEsti.calWfsErr() after checking for image size 
            # (both need to be squares)
            # calls the Algorithm.py
            tol = 1e-3
            algo.runIt(imgIntra,imgExtra, opticalModel, tol=tol)
            zk_no_offset = algo.getZer4UpInNm()

            # store the compensable image, as well as the zernikes
            baseline = {}
            baseline['imgIntra'] = imgIntra
            baseline['imgExtra'] = imgExtra
            baseline['zks'] = zk_no_offset

            fname = f'exp-{experiment_index}_{sensor}_donut_{i}_no_offset_zk_radial.npy'
            fpath = os.path.join(out_dir, fname)
            np.save(fpath, baseline)
            print(f'\nSaved the baseline fit (as well as imgIntra, imgExtra) as {fpath}')


            #########################################################
            ### STEP 2 B : fit over a range of radial distances 
            #########################################################


            ## First applying change to the extra-focal mask, and then to the intra-focal mask

            for defocal in ['intra', 'extra']: # or extra 
                print(f'Shifting the centroid for {defocal}-donut in that pair ')
                
                fname = f'exp-{experiment_index}_{sensor}_donut_0_{i}_move_{defocal}_results_radialN.npy'
                fpath = os.path.join(out_dir, fname)
                if os.path.exists(fpath):
                    print(f'Skipping {fpath}')
                    pass
                else:
                    zk0 = zk_no_offset
                    results = {}
                    j =0 
                    for drDeg in np.linspace(drDegMin,drDegMax, nGrid):

                        if defocal == 'intra':
                            xDeg, yDeg = fieldXYIntra

                        elif defocal == 'extra':
                            xDeg, yDeg = fieldXYIntra

                        xDegNew, yDegNew = get_nx_ny(xDeg, yDeg, drDeg )

                        dxDeg = xDegNew - xDeg # +ve in I, IV,  -ve in II, III
                        dyDeg = yDegNew - yDeg # +ve in I, II, -ve in III, IV

                        dxPx = dxDeg*3600 / pixelScale
                        dyPx = dyDeg*3600 / pixelScale 

                        drPx = drDeg*3600 / pixelScale 


                        results[j] = {'dxDeg': dxDeg,  'dxPx' : dxPx, 
                                          'dyDeg': dyDeg,  'dyPx' : dyPx, 
                                          'drDeg': drDeg,  'drPx' : drPx, 
                                          }
                        # do the fit 
                        if defocal == 'extra':
                            fieldXYExtraUpd = (fieldXYExtra[0]+dxDeg, fieldXYExtra[1]+dyDeg)
                            fieldXYIntraUpd = fieldXYIntra

                        elif defocal =='intra':
                            fieldXYExtraUpd = fieldXYExtra
                            fieldXYIntraUpd = (fieldXYIntra[0]+dxDeg, fieldXYIntra[1]+dyDeg)

                        imgExtra.setImg(fieldXYExtraUpd,
                                        DefocalType.Extra,
                                        image=rotate(donutExtra.stamp_im.getImage().getArray(), eulerZExtra).T)

                        imgIntra.setImg(fieldXYIntraUpd,
                                        DefocalType.Intra,
                                        image=rotate(donutIntra.stamp_im.getImage().getArray(), eulerZIntra).T)


                        # right after we set the image, the compensable image mask is empty - 
                        # it is only calculated in Algorithm.py, L694
                        maskScalingFactorLocal = 1
                        boundaryT = algo.getBoundaryThickness()
                        imgIntra.makeMask(inst, opticalModel, boundaryT, maskScalingFactorLocal)
                        imgExtra.makeMask(inst, opticalModel, boundaryT, maskScalingFactorLocal)

                        # store the imgIntra, imgExtra before running the algorithm
                        results[j]['imgExtra0mp'] = imgExtra.getNonPaddedMask()
                        results[j]['imgIntra0mp'] = imgIntra.getNonPaddedMask()

                        results[j]['imgExtra0mc'] = imgExtra.getPaddedMask()
                        results[j]['imgIntra0mc'] = imgIntra.getPaddedMask()

                        results[j]['imgExtra0img'] = imgExtra.getImg()
                        results[j]['imgIntra0img'] = imgIntra.getImg()

                        # now wfEsti.reset() simply calls the same method in `Algorithm.py` : 
                        algo.reset()

                        # now wfEsti.calWfsErr() after checking for image size (both need to be squares)
                        # simply calls algo:
                        tol = 1e-3 # explicitly set the tolerance level ( this is default )
                        algo.runIt(imgIntra, imgExtra, opticalModel, tol=tol)
                        zk = algo.getZer4UpInNm()

                        # store imgIntra, imgExtra after running the algorithm
                        results[j]['imgExtra1mp'] = imgExtra.getNonPaddedMask()
                        results[j]['imgIntra1mp'] = imgIntra.getNonPaddedMask()

                        results[j]['imgExtra1mc'] = imgExtra.getPaddedMask()
                        results[j]['imgIntra1mc'] = imgIntra.getPaddedMask()

                        results[j]['imgExtra1img'] = imgExtra.getImg()
                        results[j]['imgIntra1img'] = imgIntra.getImg()

                        # calculate diffMax, diffRms
                        diffMax = np.max(np.abs(zk - zk0))
                        diffRms = np.sqrt(np.sum(np.abs(zk - zk0) ** 2) / len(zk))

                        # store the results 
                        results[j]['diffMax'] = diffMax
                        results[j]['diffRms'] = diffRms
                        results[j]['zk'] = zk
                        results[j]['imgExtraFieldXY'] = imgExtra.getFieldXY()
                        results[j]['imgIntraFieldXY'] = imgIntra.getFieldXY()

                        # increase the grid counter 
                        j += 1 
                        print(j, dxPx, dyPx, drPx,  diffRms, )
                    print(f"saved {fpath}")
                    np.save(fpath, results)
                