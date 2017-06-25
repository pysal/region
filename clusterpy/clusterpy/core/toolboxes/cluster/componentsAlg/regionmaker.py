# encoding: latin2
"""Algorithm utilities
G{packagetree core}
"""
__author__ = "Juan C. Duque"
__credits__ = "Copyright (c) 2009-11 Juan C. Duque"
__license__ = "New BSD License"
__version__ = "1.0.0"
__maintainer__ = "RiSE Group"
__email__ = "contacto@rise-group.org"

from copy import deepcopy
import numpy as np
from .objFunctions import makeObjDict, objectiveFunctionTypeDispatcher
from .selectionTypeFunctions import selectionTypeDispatcher
from warnings import warn
from time import time

from .memory import ExtendedMemory as ExtMem
from .areacl import AreaCl
from .helperfunctions import sortedKeys
from os import getpid

class RegionMaker:
    """
    This class deals with a large amount of methods required during both the
    construction and local search phases. This class takes the area instances and
    coordinate them during the solution process. It also send information to
    Memory when needed.
    """
    def __init__(self, am, pRegions=2, initialSolution=[],
                 seedSelection = "kmeans",
                 distanceType = "EuclideanSquared",
                 distanceStat = "Centroid",
                 selectionType = "Minimum",
                 alpha = 0.2,
                 numRegionsType = "Exogenous",
                 objectiveFunctionType = "SS",
                 threshold = 0.0,
                 weightsDistanceStat = [],
                 weightsObjectiveFunctionType = [],
                 indexDataStat = [],
                 indexDataOF = []):
        """
        @type am: AreaManager
        @param am: Area manager object.

        @type pRegions: integer
        @keyword pRegions: Number of regions in scheme

        @type seeds: list
        @keyword seeds: List of area IDs for initial seeds.

        @type distanceType: string
        @keyword distanceType: Type of distance to be used, by default "EuclideanSquared"

        @type distanceStat: string
        @keyword distanceStat: Type of conversion used for summarizing distance, by defaults "Average"

        @type selectionType: string
        @keyword selectionType: Type of selection criterion for construction phase, by defaults "Minimum"

        @type alpha: float.
        @keyword alpha: float equal or between the interval [0,1]; for GRASP selection only.

        @type numRegionsType: string
        @keyword numRegionsType: Type of constructive method (Exogenous, EndogenousThreshold,
        EndogenousRange), by default "Exogenous"

        @type objectiveFunctionType: string
        @keyword objectiveFunctionType: Method to calculate the objective function, by default "Total"

        @type threshold: float
        @keyword threshold: Minimum population threshold to be satisfied for each region

        @type weightsDistanceStat: list
        @keyword weightsDistanceStat:

        @type weightsObjectiveFunctionStat: list
        @keyword weightsObjectiveFunctionStat:

        @type indexDataStat = list
        @keyword indexDataStat:

        @type indexDataOf = list
        @keyword indexDataOf:
        """
        self.am = am
        self.areas = deepcopy(am.areas)
        self.distanceType = distanceType
        self.distanceStat = distanceStat
        self.weightsDistanceStat = weightsDistanceStat
        self.indexDataStat = indexDataStat
        self.weightsObjectiveFunctionType = weightsObjectiveFunctionType
        self.indexDataOF = indexDataOF
        self.selectionType = selectionType
        self.objectiveFunctionType = objectiveFunctionType
        self.n = len(self.areas)
        self.unassignedAreas = list(self.areas.keys())
        self.assignedAreas = []
        self.area2Region = {}
        self.region2Area = {}
        self.potentialRegions4Area = {}
        self.intraBorderingAreas = {}
        self.candidateInfo = {}
        self.externalNeighs = set()
        self.alpha = alpha
        self.numRegionsType = numRegionsType
        self.neighSolutions = {(0,0): 9999}
        self.regionMoves = set()
        self.distances = {}
        self.NRegion = []
        self.N = 0
        self.data = {}
        self.objInfo = -1
        self.assignAreasNoNeighs()

        #  PREDEFINED NUMBER OF REGIONS

        seeds = []
        regions2createKeys = []
        emptyList = []
        c = 0
        lenUnassAreas = len(self.unassignedAreas)
        s = 0
        i = 0
        lseeds = 0
        if numRegionsType == "Exogenous":
            if not initialSolution:
                self.pRegions = pRegions
                seeds = self.kmeansInit()
                self.setSeeds(seeds)
                c = 0
                while lenUnassAreas > 0:
                    self.constructRegions()
                    lenUnassAreas = len(self.unassignedAreas)
                    c += 1
                self.objInfo = self.getObj()
            else:
                uniqueInitSolution = set(initialSolution)
                self.pRegions = len(uniqueInitSolution)
                seeds = []
                for s in uniqueInitSolution:
                    seeds.append(initialSolution.index(s))
                self.setSeeds(seeds)
                regions2create = {}
                c = 0

                for i in initialSolution:
                    regions2create.setdefault(i, []).append(c)
                    c += 1
                c = 0
                regions2createKeys = list(regions2create.keys())
                for i in regions2createKeys:
                    self.unassignedAreas = regions2create[i][1:]
                    lenUnassAreas = len(self.unassignedAreas)
                    while lenUnassAreas > 0:
                        self.constructRegions(filteredCandidates=self.unassignedAreas,
                                              filteredReg=i)
                        lenUnassAreas = len(self.unassignedAreas)
                        c += 1
                self.objInfo = self.getObj()
        #  NUMBER OF REGIONS IS ENDOGENOUS WITH A THRESHOLD VALUE

        if self.numRegionsType == "EndogenousThreshold":
            self.constructionStage = "growing"
            try:
                self.areas[list(self.areas.keys())[0]].thresholdVar
            except:
                self.extractThresholdVar()
            self.regionalThreshold = threshold
            c = 0
            self.feasibleRegions = {}
            self.regionValue = {}
            seeds = []
            for aID in self.areas:
                if self.areas[aID].thresholdVar >= self.regionalThreshold:
                    seed = aID
                    seeds = seeds + [seed]
                    self.regionValue[c] = self.areas[seed].thresholdVar
                    self.feasibleRegions[c] = [seed]
                    self.removeRegionAsCandidate()
                    c += 1
            self.setSeeds(seeds)
            while len(self.unassignedAreas) != 0:
                np.random.shuffle(self.unassignedAreas)
                vals = []
                for index in self.unassignedAreas:
                    vals += [self.areas[index].thresholdVar]
                seed = self.unassignedAreas[0]
                self.setSeeds([seed], c)
                self.regionValue[c] = self.areas[seed].thresholdVar
                if self.regionValue[c] >= self.regionalThreshold:
                    self.feasibleRegions[c] = [seed]
                    self.removeRegionAsCandidate()
                    c += 1
                else:
                    feasibleThreshold = 1
                    while self.regionValue[c] < self.regionalThreshold:
                        self.addedArea = -1
                        try:
                            self.constructRegions()
                            self.regionValue[c] += self.areas[self.addedArea].thresholdVar
                        except:
                            feasibleThreshold = 0
                            break
                    if feasibleThreshold == 1:
                        self.feasibleRegions[c] = self.region2Area[c]
                        self.removeRegionAsCandidate()
                    c += 1

        #  NUMBER OF REGIONS IS ENDOGENOUS WITH A RANGE VALUE

        if self.numRegionsType == "EndogenousRange":
            self.constructionStage = "growing"  #  there are two values for constructionStage: "growing" and "enclaves"
            try:
                self.areas[list(self.areas.keys())[0]].thresholdVar
            except:
                self.extractThresholdVar()
            self.regionalThreshold = threshold
            c = 0
            self.feasibleRegions = {}
            while len(self.unassignedAreas) != 0:

                #  select seed

                np.random.shuffle(self.unassignedAreas)
                seed = self.unassignedAreas[0]
                self.setSeeds([seed],c)

                #  regionRange contains the current range per region
                #  regionalThreshold is the predefined threshold value

                self.regionRange = {}
                maxValue = self.areas[seed].thresholdVar
                minValue = self.areas[seed].thresholdVar
                currentRange = maxValue - minValue
                self.regionRange[c] = currentRange

                # grow region if possible

                stop = 0
                while stop == 0:
                    upplim = maxValue + self.regionalThreshold - currentRange
                    lowlim = minValue - self.regionalThreshold + currentRange
                    feasibleNeigh = 0
                    toRemove = []
                    for ext in self.externalNeighs:
                        if self.areas[ext].thresholdVar <= upplim and self.areas[ext].thresholdVar >= lowlim:
                            feasibleNeigh = 1
                        if self.areas[ext].thresholdVar > upplim or self.areas[ext].thresholdVar < lowlim:
                            toRemove.append(ext)
                    self.toRemove = toRemove
                    if feasibleNeigh == 0:
                        stop = 1
                    if feasibleNeigh == 1:
                        try:
                            self.constructRegions()
                            if self.areas[self.addedArea].thresholdVar > maxValue:
                                maxValue = self.areas[self.addedArea].thresholdVar
                            if self.areas[self.addedArea].thresholdVar < minValue:
                                minValue = self.areas[self.addedArea].thresholdVar
                            currentRange = maxValue - minValue
                            self.regionRange[c] = currentRange
                        except:
                            stop = 1
                self.feasibleRegions[c] = self.region2Area[c]
                self.removeRegionAsCandidate()
                c += 1
        self.getIntraBorderingAreas()

    def kmeansInit(self):
        cachedDistances = {}
        y = self.am.y
        n = len(y)
        distances = np.ones(n)
        total = sum(distances)
        probabilities = [x / float(total) for x in distances]
        seeds = []
        localDistanceType = self.distanceType
        returnDistance2Area = AreaCl.returnDistance2Area
        np.random.seed(int(time() * getpid()) % 4294967295)
        for k in range(self.pRegions):
            random = np.random.uniform(0, 1)
            find = False
            acum = 0
            cont = 0
            while not find:
                inf = acum
                sup = acum + probabilities[cont]
                if inf <= random <= sup:
                    find = True
                    seeds += [cont]
                    selfAmAreas = self.am.areas

                    for area in selfAmAreas:
                        currentArea = selfAmAreas[area]
                        tempMap = []
                        for x in seeds:
                            if x < area:
                                k = (x, area)
                            elif x > area:
                                k = (area, x)
                            else:
                                k = (0,0)
                            cached = cachedDistances.get(k, -1)
                            if cached < 0:
                                newDist = returnDistance2Area(currentArea,
                                                               selfAmAreas[x],
                                                               distanceType = localDistanceType)
                                tempMap.append(newDist)
                                cachedDistances[k] = newDist

                            else:
                                tempMap.append(cached)
                        distancei = min(tempMap)
                        distances[area] = distancei
                    total = sum(distances)
                    probabilities = [x / float(total) for x in distances]
                else:
                    cont += 1
                    acum = sup
        del cachedDistances
        return seeds

    def extractThresholdVar(self):
        """
        Separate aggregation variables (data) from the variable selected
        to satisfy a threshold value (thresholdVar)
        """
        self.totalThresholdVar = 0.0
        for areaId in list(self.areas.keys()):
            self.areas[areaId].thresholdVar = self.areas[areaId].data[-1]
            self.areas[areaId].data = self.areas[areaId].data[0: -1]
            self.totalThresholdVar += self.areas[areaId].thresholdVar

    def removeRegionAsCandidate(self):
        """
        Remove a region from candidates
        """
        for i in list(self.candidateInfo.keys()):
            a, r = i
            if r in self.feasibleRegions:
                self.candidateInfo.pop(i)

    def returnRegions(self):
        """
        Return regions created
        """
        areasId = list(self.area2Region.keys())
        areasId = np.sort(areasId).tolist()
        return [self.area2Region[area] for area in areasId]

    def resetNow(self):
        """
        Reset all variables
        """
        self.unassignedAreas = list(self.areas.keys())
        self.assignedAreas = []
        self.area2Region = {}
        self.region2Area = {}
        self.potentialRegions4Area = {}
        self.intraBorderingAreas = {}
        self.candidateInfo = {}
        self.externalNeighs = set([])
        self.neighsMinusAssigned = set([])

    def setSeeds(self, seeds, c=0):
        """
        Sets the initial seeds for clustering
        """
        if self.numRegionsType == "Exogenous" and len(seeds) <= self.pRegions:
            idx = list(range(self.n))
            didx = list((set(idx) - set(seeds)) - self.am.noNeighs)
            np.random.shuffle(didx)
            self.seeds = seeds + didx[0:(self.pRegions - len(seeds))]
        else:
            self.seeds = seeds
        for seed in self.seeds:
            self.NRegion += [0]
            self.assignSeeds(seed, c)
            c += 1

    def assignAreaStep1(self, areaID, regionID):
        """
        Assign an area to a region
        """
        a = self.areas[areaID]
        neighs = a.neighs
        try:
            self.region2Area[regionID].append(areaID)
            if self.objectiveFunctionType == "GWalt":
                try:
                    self.NRegion[regionID] += a.data[0]
                    for index in range(1,len(a.data)):
                        self.data[regionID][index - 1] += a.data[index] * a.data[0]
                except:
                    self.NRegion[regionID] = a.data[0]
                    for index in range(1, len(a.data)):
                        self.data[regionID][index - 1] = a.data[index] * a.data[0]
                self.N += a.data[0]
        except:
            self.region2Area[regionID] = [areaID]
            if self.objectiveFunctionType == "GWalt":
                self.NRegion[regionID] = a.data[0]
                for index in range(1, len(a.data)):
                    if index == 1:
                        self.data[regionID] = [a.data[index] * a.data[0]]
                    else:
                        self.data[regionID] += [a.data[index] * a.data[0]]
                self.N += a.data[0]
        self.area2Region[areaID] = regionID
        try:
            aid = self.unassignedAreas.remove(areaID)
        except:
            pass
        self.assignedAreas.append(areaID)
        setNeighs = set(neighs)
        setAssigned = set(self.assignedAreas)
        self.oldExternal = self.externalNeighs
        self.externalNeighs.update(setNeighs)
        self.externalNeighs.difference_update(setAssigned)
        self.newExternal = self.externalNeighs - self.oldExternal
        self.neighsMinusAssigned = setNeighs - setAssigned

    def assignSeeds(self, areaID, regionID):
        """
        Assign an area to a region and updates potential regions for the neighs
        Parameters
        """
        self.assignAreaStep1(areaID, regionID)
        for neigh in self.neighsMinusAssigned:
            self.potentialRegions4Area.setdefault(neigh, set()).add(regionID)

        try:
            self.potentialRegions4Area.pop(areaID)
        except:
            pass
        self.changedRegion = 'null'
        self.newExternal = list(self.potentialRegions4Area.keys())

    def assignAreasNoNeighs(self):
        """
        Assign to the region "-1" for the areas without neighbours
        """
        noNeighs = list(self.am.noNeighs)
        nr = -1
        for areaID in noNeighs:
            self.area2Region[areaID] = nr
            try:
                aid = self.unassignedAreas.remove(areaID)
            except:
                pass
            self.assignedAreas.append(areaID)
            setAssigned = set(self.assignedAreas)
        nr = nr - 1

    def assignArea(self, areaID, regionID):
        """
        Assign an area to a region and updates potential regions for neighs
        """
        self.changedRegion = regionID
        self.addedArea = areaID
        self.assignAreaStep1(areaID, regionID)
        for neigh in self.neighsMinusAssigned:
            self.potentialRegions4Area.setdefault(neigh, set()).add(regionID)

        try:
            self.potentialRegions4Area.pop(areaID)
        except:
            pass

    def returnBorderingAreas(self, regionID):
        """
        Returns bordering areas of a region
        """
        areas2Eval = self.region2Area[regionID]
        borderingAreas = set()
        for area in areas2Eval:
            try:
                if len(self.intraBorderingAreas[area]) > 0:
                    borderingAreas.add(area)
            except:
                pass
        return borderingAreas

    def getIntraBorderingAreas(self):
        """
        Gets the intrabordering areas
        """
        self.intraBorderingAreas = {}
        if self.numRegionsType == "Exogenous":
            nr = list(range(self.pRegions))
        else:
            nr = self.feasibleRegions
        for regionID in nr:
            setNeighsNoRegion = set()
            try:
                areas2Eval = self.region2Area[regionID]
            except:
                areas2Eval = []

            for area in areas2Eval:
                setNeighsNoRegion.update(self.areas[area].neighs)
            setNeighsNoRegion.difference_update(areas2Eval)

            for neigh in setNeighsNoRegion:
                self.intraBorderingAreas.setdefault(neigh, set()).add(regionID)

    def constructRegions(self, filteredCandidates=-99, filteredReg=-99):
        """
        Construct potential regions per area
        """
        _d_stat = self.distanceStat
        _wd_stat = self.weightsDistanceStat
        _ida_stat = self.indexDataStat
        _fun_am_d2r = self.am.getDistance2Region

        lastRegion = 0
        for areaID in list(self.potentialRegions4Area.keys()):
            a = self.areas[areaID]
            regionIDs = list(self.potentialRegions4Area[areaID])
            for region in regionIDs:
                if (self.numRegionsType != "Exogenous" and
                    self.constructionStage == "growing"
                    and region in self.feasibleRegions):
                    #  Once a region reaches the threshold, the grow is
                    #  rejected until the assignation of enclaves
                    continue
                else:
                    if filteredCandidates == -99:
                        if (areaID not in self.newExternal and
                            region != self.changedRegion):
                            lastRegion = region
                            pass
                        else:
                            _reg_dist = 0.0
                            if self.selectionType != "FullRandom":
                                _reg_dist = _fun_am_d2r(self.areas[areaID],
                                                        self.region2Area[region],
                                                        distanceStat = _d_stat,
                                                        weights = _wd_stat,
                                                        indexData = _ida_stat)
                            self.candidateInfo[(areaID, region)] = _reg_dist

                    elif (filteredCandidates != -99 and
                          areaID in filteredCandidates and
                          region == filteredReg):
                        _reg_dist = _fun_am_d2r(self.areas[areaID],
                                                self.region2Area[region],
                                                distanceStat = _d_stat,
                                                weights = _wd_stat,
                                                indexData = _ida_stat)
                        self.candidateInfo[(areaID, region)] = _reg_dist

        if len(self.candidateInfo) == 0:
            self.changedRegion = lastRegion
        if self.numRegionsType == "EndogenousRange":
            self.filterCandidate(self.toRemove)
        selectionTypeDispatcher[self.selectionType](self)

    def filterCandidate(self, removeCandidate=[]):
        """
        Filter candidates
        """
        if len(removeCandidate) > 0:
            toRemove = []
            for _id in removeCandidate:
                for cand, reg in list(self.candidateInfo.keys()):
                    if cand == _id:
                        toRemove.append((cand, reg))
            for remov in toRemove:
                self.candidateInfo.pop(remov)

    def graspList(self, xList, alpha=0.0):
        """
        Return random index of values with specified range.
        """
        maxX = max(xList)
        minX = min(xList)
        xRangeMax = minX + ((maxX - minX) * alpha)
        candidates = [i <= xRangeMax for i in xList]
        indices = indexMultiple(candidates, 1)
        nCandidates = len(indices)
        idx = list(range(nCandidates))
        np.random.shuffle(idx)
        random = idx[0]
        index4Grasp = indices[random]
        return index4Grasp

    def getObjective(self, region2AreaDict):
        """
        Return the value of the objective function from regions2area dictionary

        This function acts as a proxy function since the idea behind the
        getObjective and getObjectiveFast is the same. When the non-fast
        approach is needed, this function will call getObjectiveFast with the
        extra parameter as None. This way the fast function will execute as the
        non-fast would have.
        """
        return self.getObjectiveFast(region2AreaDict, modifiedRegions=None)

    def getObjectiveFast(self, region2AreaDict, modifiedRegions=[]):
        """
        Return the value of the objective function from regions2area dictionary

        When this function gets called, the objectiveFunctionType property
        could be either a String representing the type of the function (the
        common case), or could be a list of function types, in which case it's
        necessary to iterate over all the functions.
        """
        distance = 0.0
        _objFunType = self.objectiveFunctionType

        if isinstance(_objFunType, "".__class__):
            if len(self.indexDataOF) == 0:
                indexData = list(range(len(self.areas[0].data)))
            else:
                indexData = self.indexDataOF
            _fun = None
            if modifiedRegions == None:
                _fun = objectiveFunctionTypeDispatcher[_objFunType]
                distance = _fun(self, region2AreaDict, indexData)
            else:
                _fun = objectiveFunctionTypeDispatcher[_objFunType+'f']
                distance=_fun(self, region2AreaDict, modifiedRegions, indexData)

        else:
            i = 0
            for oFT in _objFunType:
                if len(self.indexDataOF) == 0:
                    indexData = list(range(len(self.areas[0].data)))
                else:
                    indexData = self.indexDataOF[i]

                if len(self.weightsObjectiveFunctionType) > 0:
                    _fun = objectiveFunctionTypeDispatcher[oFT]
                    distance += (self.weightsObjectiveFunctionType[i] *
                                 _fun(self, region2AreaDict, indexData))
                    i += 1
                else:
                    _fun = objectiveFunctionTypeDispatcher[oFT]
                    distance += _fun(self, region2AreaDict, indexData)

        return distance

    def getLambda(self):
        L = np.matrix(np.identity(self.pRegions))
        for r in range(self.pRegions):
            L[r, r] = 1.0 * self.NRegion[r] / self.N
        return L

    def getB(self):
        """
        Return matrix of parameters of all regions
        """
        B = np.matrix(np.zeros(len(self.data[0]) * self.pRegions)).T
        index = 0
        for r in range(self.pRegions):
            for i in range(len(self.data[0])):
                B[index, 0] = self.data[r][i] / self.NRegion[r]
                index += 1
        return B

    def getY(self):
        """
        Return matrix of the average variance-covariance of all regions
        """
        Y = np.matrix(np.identity(len(self.data[0])))
        centroids = {}
        for r in range(self.pRegions):
            centroids[r] = calculateCentroid([self.areas[aID] for aID in self.region2Area[r]])
        for r in range(self.pRegions):
            Y += centroids[r].var * np.power(self.NRegion[r] / self.N, 2)
        return Y

    def getH(self):
        """
        Return composite matrix
        """
        E = np.matrix(np.ones((1, self.pRegions, self.pRegions)))
        L = self.getLambda()
        H = L - L * E * L
        return H

    def getObj(self):
        """
        Return the value of the objective function
        """
        if self.objInfo < 0:
            self.calcObj()
        return self.objInfo

    def calcObj(self):
        """
        Calculate the value of the objective function
        """
        self.objInfo = self.getObjective(self.region2Area)

    def recalcObj(self, region2AreaDict, modifiedRegions=[]):
        """
        Re-calculate the value of the objective function
        """
        if hasattr(self, 'objDict'):
            obj = self.getObjectiveFast(region2AreaDict, modifiedRegions)
        else:
            obj = self.getObjective(region2AreaDict)
        return obj

    def checkFeasibility(self, regionID, areaID,
                         region2AreaDict = None):
        """
        Check feasibility from a change region (remove an area from a region)
        """
        if not region2AreaDict:
            region2AreaDict = self.region2Area
        areas2Eval = list(region2AreaDict[regionID])
        a2r = set(region2AreaDict[regionID])
        aIDset = set([areaID])
        areas2Eval.remove(areaID)
        seedArea = areas2Eval[0]
        newRegion = set()
        newRegion.add(seedArea)
        newRegion.update(self.areas[seedArea].neighs)
        newRegion.intersection_update(areas2Eval)
        areas2Eval.remove(seedArea)
        flag = 1
        newAdded = newRegion.copy()
        newAdded.discard(seedArea)
        newNeighs = set()

        while flag:
            for area in newAdded:
                newNeighs.update(self.areas[area].neighs)
                areas2Eval.remove(area)
            newNeighs.intersection_update(a2r)
            newNeighs.difference_update(aIDset, newRegion, newAdded)
            newAdded = newNeighs.copy()
            newRegion.update(newAdded)

            ###
            # for area in newAdded:
            #     newNeighs = newNeighs | (((set(self.areas[area].neighs) & a2r) - aIDset) - newRegion)
            #     areas2Eval.remove(area)
            # newNeighs.difference_update(newAdded)
            # newAdded = newNeighs
            # newRegion.update(newAdded)
            ###

            if len(areas2Eval) == 0:
                feasible = 1
                flag = 0
                break
            elif newNeighs == set() and len(areas2Eval) > 0:
                feasible = 0
                flag = 0
                break

        return feasible

    def calculateRegionValueThreshold(self):
        if self.numRegionsType == "Exogenous":
            nr = list(range(self.pRegions))
        else:
            nr = list(range(len(list(self.region2Area.keys()))))
        for regionID in nr:
            self.regionValue[regionID] = 0
            areas2Eval = self.region2Area[regionID]
            for area in areas2Eval:
                self.regionValue[regionID] += self.areas[area].thresholdVar

    def improvingCandidates(self):
        """
        Select solutions that improve the current objective function.
        """
        intraCopy = deepcopy(self.intraBorderingAreas)
        region2AreaCopy = deepcopy(self.region2Area)
        area2RegionCopy = deepcopy(self.area2Region)
        self.neighSolutions = {}
        for area in list(intraCopy.keys()):
            regionIn = self.area2Region[area]
            regions4Move = list(self.intraBorderingAreas[area])
            if (len(self.region2Area[regionIn]) > 1):
                for region in regions4Move:
                    self.swapArea(area, region, region2AreaCopy, area2RegionCopy)
                    obj = self.recalcObj(region2AreaCopy)
                    self.swapArea(area, regionIn, region2AreaCopy, area2RegionCopy)
                    if obj < self.objInfo:
                        _feasible = self.checkFeasibility(regionIn, area, self.region2Area)
                        if _feasible == 1:
                            if self.numRegionsType == "Exogenous":
                                self.neighSolutions[(area, region)] = obj
                            elif self.numRegionsType == "EndogenousThreshold":
                                if (self.regionValue[region] >= self.regionalThreshold and
                                    self.regionValue[regionIn] >= self.regionalThreshold):
                                    self.neighSolutions[(area,region)] = obj

    def allCandidates(self):
        """
        Select neighboring solutions.
        """
        intraCopy = deepcopy(self.intraBorderingAreas)
        reg2AreaCp = deepcopy(self.region2Area)
        area2RegionCp = deepcopy(self.area2Region)
        neighSolutions = {}

        for area in list(intraCopy.keys()):
            regionIn = self.area2Region[area]
            regions4Move = list(self.intraBorderingAreas[area])
            if (len(self.region2Area[regionIn]) > 1):
                for region in regions4Move:
                    _feasible = self.checkFeasibility(regionIn, area)
                    if _feasible == 1:
                        self.swapArea(area, region, reg2AreaCp, area2RegionCp)
                        if self.numRegionsType == "Exogenous":
                            modifiedRegions = [region, regionIn]
                            obj = self.recalcObj(reg2AreaCp, modifiedRegions)
                            neighSolutions[(area, region)] = obj
                        elif (self.numRegionsType == "EndogenousThreshold" and
                              self.regionValue[region] >= self.regionalThreshold and
                              self.regionValue[regionIn] >= self.regionalThreshold):
                            obj = self.recalcObj(reg2AreaCp)
                            neighSolutions[(area, region)] = obj
                        self.swapArea(area, regionIn, reg2AreaCp, area2RegionCp)

        self.neighSolutions = neighSolutions

    def allMoves(self):
        """
        Select all posible moves.
        """
        moves = []
        for area in self.intraBorderingAreas:
            regionIn = self.area2Region[area]
            regions4Move = list(self.intraBorderingAreas[area])
            if len(self.region2Area[regionIn]) > 1:
                for region in regions4Move:
                    moves.append((area, region))
        return moves

    def swapArea(self, area, newRegion, region2AreaDict, area2RegionDict):
        """
        Take an area from a region and give it to another
        """
        oldRegion = area2RegionDict[area]
        region2AreaDict[oldRegion].remove(area)
        region2AreaDict[newRegion].append(area)
        area2RegionDict[area] = newRegion
        if self.objectiveFunctionType == "GWalt":
            a = self.areas[area]
            lendata = len(a.data)
            adata0 = a.data[0]
            self.NRegion[newRegion] += adata0
            self.NRegion[oldRegion] -= adata0
            for index in range(1, lendata):
                self.data[newRegion][index - 1] += a.data[index] * adata0
                self.data[oldRegion][index - 1] -= a.data[index] * adata0
        if self.numRegionsType == "EndogenousThreshold":
            self.regionValue[newRegion] += self.areas[area].thresholdVar
            self.regionValue[oldRegion] -= self.areas[area].thresholdVar

    def greedyMove(self, typeGreedy="random"):
        """
        Conduct a solution to the best posible with greedy moves
        """
        flag = 1
        self.round = 0
        while flag:
            self.improvingCandidates()
            self.round = 1
            if len(list(self.neighSolutions.keys())) == 0:
                flag = 0
            else:
                sortedk = sortedKeys(self.neighSolutions)
                if typeGreedy == "exact":
                    move = sortedk[np.random.randint(0, len(sortedk))]
                    area, region = move
                else:
                    values = list(self.neighSolutions.values())
                    minVal = min(self.neighSolutions.values())
                    indicesMin = indexMultiple(values, minVal)
                    nInd = len(indicesMin)
                    idx = list(range(nInd))
                    np.random.shuffle(idx)
                    minIndex = indicesMin[idx[0]]
                    area,region = list(self.neighSolutions.keys())[minIndex]
                self.moveArea(area, region)

        self.regions = self.returnRegions()

    def updateTabuList(self, newValue, aList, endInd):
        """
        Add a new value to the tabu list.
        """
        return [newValue] + aList[0:endInd-1]

    def tabuMove(self, tabuLength = 5, convTabu = 5, typeTabu="exact"):
        """
        Conduct a solution to the best posible with tabu search
        """
        is_exact_type = (typeTabu == "exact")
        is_rand_type = (typeTabu == "random")
        aspireOBJ = self.objInfo
        currentOBJ = self.objInfo
        aspireRegions = self.returnRegions()
        currentRegions = aspireRegions
        region2AreaAspire = deepcopy(self.region2Area)
        area2RegionAspire = deepcopy(self.area2Region)
        bestAdmisable = 9999999.0
        tabuList = [0]*tabuLength
        cBreak = []
        c = 1
        self.round = 0
        resList = []
        epsilon = 1e-10

        while c <= convTabu:
            if is_exact_type:
                self.objDict = makeObjDict(self)
                self.allCandidates()
            else:
                moves = self.allMoves()

            if ((is_exact_type and len(self.neighSolutions) == 0) or
                (is_rand_type and len(moves) == 0)):
                c += convTabu
            else:
                if is_exact_type:
                    sortedk = sortedKeys(self.neighSolutions)
                    end = len(sortedk)
                else:
                    end = len(moves)
                run = 0

                while run < end:
                    if is_exact_type:
                        move = sortedk[run]
                        area, region = move
                        obj4Move = self.neighSolutions[move]
                        candidate = 1
                    else:
                        candidate = 0
                        region2AreaCopy = deepcopy(self.region2Area)
                        area2RegionCopy = deepcopy(self.area2Region)
                        while (candidate == 0 and len(moves) > 0):
                            move = moves[np.random.randint(0, len(moves))]
                            moves.remove(move)
                            area, region = move
                            run += 1
                            regionIn = self.area2Region[area]
                            _feasible = self.checkFeasibility(regionIn, area)
                            if _feasible == 1:
                                self.swapArea(area, region, region2AreaCopy,
                                              area2RegionCopy)

                                if self.numRegionsType == "Exogenous":
                                    obj4Move = self.recalcObj(region2AreaCopy)
                                    candidate = 1
                                elif (self.numRegionsType == "EndogenousThreshold" and
                                      self.regionValue[region] >= self.regionalThreshold and
                                      self.regionValue[regionIn] >= self.regionalThreshold):
                                    obj4Move = self.recalcObj(region2AreaCopy)
                                    candidate = 1

                                self.swapArea(area, regionIn, region2AreaCopy,
                                              area2RegionCopy)

                    tabuCount = 0
                    if candidate == 0:
                        c += convTabu
                        continue

                    if move in tabuList:
                        if (aspireOBJ - obj4Move) > epsilon:
                            oldRegion = self.area2Region[area]
                            tabuList = self.updateTabuList((area, oldRegion),
                                                           tabuList, tabuLength)
                            self.moveArea(area, region)
                            self.objInfo = obj4Move
                            aspireOBJ = obj4Move
                            currentOBJ = obj4Move
                            aspireRegions = self.returnRegions()
                            currentRegions = aspireRegions
                            region2AreaAspire = deepcopy(self.region2Area)
                            area2RegionAspire = deepcopy(self.area2Region)
                            bestAdmisable = obj4Move
                            cBreak.append(c)
                            c = 1
                            run = end
                            resList.append([obj4Move, aspireOBJ])
                        else:
                            run += 1
                            tabuCount += 1
                            tabuList = self.updateTabuList((-1, 0),
                                                           tabuList, tabuLength)
                            if tabuCount == end:
                                c = convTabu
                    else:
                        oldRegion = self.area2Region[area]
                        tabuList = self.updateTabuList((area, oldRegion),
                                                       tabuList, tabuLength)
                        self.moveArea(area, region)
                        self.objInfo = obj4Move
                        currentOBJ = obj4Move
                        if (aspireOBJ - obj4Move) > epsilon:
                            aspireOBJ = obj4Move
                            aspireRegions = self.returnRegions()
                            currentRegions = self.returnRegions()
                            region2AreaAspire = deepcopy(self.region2Area)
                            area2RegionAspire = deepcopy(self.area2Region)
                            cBreak.append(c)
                            c = 1
                        else:
                            currentRegions = self.returnRegions()
                            c += 1
                        bestAdmisable = obj4Move
                        run = end
                        resList.append([obj4Move, aspireOBJ])

        self.objInfo = aspireOBJ
        self.regions = aspireRegions
        self.region2Area = deepcopy(region2AreaAspire)
        self.area2Region = deepcopy(area2RegionAspire)
        self.resList = resList
        self.cBreak = cBreak

    def AZPImproving(self):
        """
        """
        improve = 1
        while improve == 1:
            regions = list(range(0, self.pRegions))
            while len(regions) > 0:

                # step 3

                if len(regions) > 1:
                    randomRegion = np.random.randint(0, len(regions))
                else:
                    randomRegion = 0
                region = regions[randomRegion]
                regions.remove(region)

                # step 4

                borderingAreas = list(set(self.returnBorderingAreas(region)) & set(self.region2Area[region]))
                improve = 0
                while len(borderingAreas) > 0:

                    # step 5

                    randomArea = np.random.randint(0, len(borderingAreas))
                    area = borderingAreas[randomArea]
                    borderingAreas.remove(area)
                    posibleMove = list(self.intraBorderingAreas[area])
                    if len(self.region2Area[region]) >= 2:
                        f = self.checkFeasibility(region, area, self.region2Area)
                    else:
                        f = 0
                    if f == 1:
                        for move in posibleMove:
                            self.swapArea(area, move, self.region2Area, self.area2Region)
                            obj = self.recalcObj(self.region2Area)
                            self.swapArea(area, region, self.region2Area, self.area2Region)
                            if obj <= self.objInfo:
                                self.moveArea(area, move)
                                improve = 1
                                borderingAreas = list(set(self.returnBorderingAreas(region)) & set(self.region2Area[region]))
                                break

    def AZPSA(self, alpha = 0.85, maxit = 2):
        """ Openshaw's Simulated Annealing for AZP algorithm
        """
        basicMemory = ExtMem()
        localBasicMemory = ExtMem()
        T = 1
        k = 0
        while k < 3:
            improved = 0
            for i in range(maxit):
                localBasicMemory.updateExtendedMemory(self)
                self.modified_azp_for_sa(alpha, T)
                if self.objInfo < localBasicMemory.objInfo:
                    improved = 1
                if self.objInfo < basicMemory.objInfo:
                    basicMemory.updateExtendedMemory(self)
            T *= alpha
            if improved == 1:
                k = 0
            else:
                k += 1
        self.recoverFromExtendedMemory(basicMemory)

    def modified_azp_for_sa(self, alpha, temperature):
        """
        Openshaw's modified step #5 for AZP-SA algorithm
        """
        totalMoves = 0
        acceptedMoves = 0
        bestOBJ = self.objInfo
        currentOBJ = self.objInfo
        bestRegions = self.returnRegions()
        currentRegions = self.returnRegions()
        region2AreaBest = deepcopy(self.region2Area)
        area2RegionBest = deepcopy(self.area2Region)

        improve = 1
        while improve == 1:
            regions = list(range(0, self.pRegions))
            while len(regions) > 0:

                #  step 3

                if len(regions) > 1:
                    randomRegion = np.random.randint(0, len(regions) - 1)
                else:
                    randomRegion = 0
                region = regions[randomRegion]
                regions.remove(region)

                # step 4

                borderingAreas = list(set(self.returnBorderingAreas(region)) & set(self.region2Area[region]))
                improve = 0
                while len(borderingAreas) > 0:

                    # step 5

                    randomArea = np.random.randint(0, len(borderingAreas))
                    area = borderingAreas[randomArea]
                    borderingAreas.remove(area)
                    posibleMove = list(self.intraBorderingAreas[area])
                    if len(self.region2Area[region]) >= 2:
                        f = self.checkFeasibility(region, area, self.region2Area)
                    else:
                        f = 0
                    if f == 1:
                        for move in posibleMove:

                            #  if len(region2AreaCopy[area2RegionCopy[area]]) > 1:

                            self.swapArea(area, move, self.region2Area, self.area2Region)
                            obj = self.recalcObj(self.region2Area)
                            self.swapArea(area, region, self.region2Area, self.area2Region)
                            if obj <= bestOBJ:
                                self.moveArea(area, move)
                                improve = 1
                                self.objInfo = obj
                                bestOBJ = obj
                                currentOBJ = obj
                                bestRegions = self.returnRegions()
                                currentRegions = self.returnRegions()
                                region2AreaBest = deepcopy(self.region2Area)
                                area2RegionBest = deepcopy(self.area2Region)
                                borderingAreas = list(set(self.returnBorderingAreas(region)) & set(self.region2Area[region]))
                                break
                            else:
                                random = np.random.rand(1)[0]
                                totalMoves += 1
                                if (np.exp(-(obj - currentOBJ) / (currentOBJ * temperature))) > random:
                                    acceptedMoves += 1
                                    self.moveArea(area, move)
                                    self.objInfo = obj
                                    currentOBJ = obj
                                    currentRegions = self.returnRegions()

                                    #  print "--- NON-improving move (area, region)", area, move
                                    #  print "--- New Objective Function value: ", obj
                                    #  step 4
                                    borderingAreas = list(set(self.returnBorderingAreas(region)) & set(self.region2Area[region]))
                                    break
        self.objInfo = bestOBJ
        self.region2Area = deepcopy(region2AreaBest)
        self.area2Region = deepcopy(area2RegionBest)
        self.getIntraBorderingAreas()

    def AZPTabuMove(self, tabuLength=5, convTabu=5):
        """
        Tabu search algorithm for Openshaws AZP-tabu (1995)
        """
        aspireOBJ = self.objInfo
        currentOBJ = self.objInfo
        aspireRegions = self.returnRegions()
        region2AreaAspire = deepcopy(self.region2Area)
        area2RegionAspire = deepcopy(self.area2Region)
        currentRegions = deepcopy(aspireRegions)
        tabuList = np.zeros(tabuLength)
        tabuList = tabuList.tolist()
        cBreak = []
        c = 1
        self.round = 0
        resList = []
        epsilon = 1e-10

        while c <= convTabu:
            self.objDict = makeObjDict(self)
            self.allCandidates()
            if len(self.neighSolutions) == 0:
                c += convTabu
            else:
                minFound = 0
                neighSolutionsCopy = deepcopy(self.neighSolutions)
                c += 1
                neighNoTabuKeys = list(set(neighSolutionsCopy.keys()) - set(tabuList))
                neighNoTabuDict = dict((key, neighSolutionsCopy[key]) for key in neighNoTabuKeys)
                if len(neighNoTabuDict) > 0:
                    move = min(neighNoTabuDict, key = lambda x: neighNoTabuDict.get(x))
                    obj4Move = self.neighSolutions[move]
                    moveNoTabu = move
                    obj4MoveNoTabu = obj4Move
                    if (currentOBJ - obj4Move) >= epsilon:
                        minFound = 1
                    else:
                        neighTabuKeys = list(set(neighSolutionsCopy.keys()) & set(tabuList))
                        neighTabuDict = dict((key, neighSolutionsCopy[key]) for key in neighTabuKeys)
                        if len(neighTabuDict) > 0:
                            move = min(neighTabuDict, key = lambda x: neighTabuDict.get(x))
                            obj4Move = self.neighSolutions[move]
                            moveTabu = move
                            obj4MoveTabu = obj4Move
                            if (aspireOBJ - obj4Move) > epsilon:
                                minFound = 1
                if minFound == 1:
                    area, region = move
                    obj4Move = self.neighSolutions[move]
                    oldRegion = self.area2Region[area]
                    tabuList = self.updateTabuList((area, oldRegion), tabuList, tabuLength)
                    self.moveArea(area, region)
                    self.objInfo = obj4Move
                    if (aspireOBJ - obj4Move) > epsilon:
                        aspireOBJ = obj4Move
                        aspireRegions = self.returnRegions()
                        region2AreaAspire = deepcopy(self.region2Area)
                        area2RegionAspire = deepcopy(self.area2Region)
                        c = 1
                    currentOBJ = obj4Move
                    currentRegions = self.returnRegions()
                else:
                    move = moveNoTabu
                    area, region = move
                    obj4Move = self.neighSolutions[move]
                    oldRegion = self.area2Region[area]
                    tabuList = self.updateTabuList((area, oldRegion), tabuList, tabuLength)
                    self.moveArea(area, region)
                    self.objInfo = obj4Move
                    currentOBJ = obj4Move
                    currentRegions = self.returnRegions()
        self.objInfo = aspireOBJ
        self.regions = aspireRegions
        self.region2Area = deepcopy(region2AreaAspire)
        self.area2Region = deepcopy(area2RegionAspire)
        self.resList = resList

    def reactiveTabuMove(self, convTabu=99):
        """
        AZP
        Openshaw's Reactive Tabu algorithm
        """

        #  step 2

        tabuLength = 1
        tabuList = np.zeros(tabuLength)
        tabuList = tabuList.tolist()
        rAvg = 1
        K1 = 3
        K2 = 3
        visitedSolutions = []
        allVisitedSolutions = {}
        self.round = 0
        epsilon = 1e-10
        aspireOBJ = self.objInfo
        aspireRegions = self.returnRegions()
        region2AreaAspire = deepcopy(self.region2Area)
        area2RegionAspire = deepcopy(self.area2Region)
        c = 1
        while c <= convTabu:
            improved = 0

            #  step 3

            self.objDict = makeObjDict(self)
            self.allCandidates()
            if len(self.neighSolutions) == 0:
                c += convTabu
            else:
                neighSolutionsCopy = deepcopy(self.neighSolutions)
                neighNoTabuKeys = list(set(neighSolutionsCopy.keys()) - set(tabuList))
                neighNoTabuDict = dict((key, neighSolutionsCopy[key]) for key in neighNoTabuKeys)

                # step 4

                if len(neighNoTabuDict) > 0:
                    move = min(neighNoTabuDict, key = lambda x: neighNoTabuDict.get(x))
                    obj4Move = self.neighSolutions[move]
                else:
                    c += convTabu
                    break;

                #  step 5

                area, region = move
                obj4Move = self.neighSolutions[move]
                oldRegion = self.area2Region[area]
                tabuList = self.updateTabuList((area, oldRegion), tabuList, tabuLength)
                self.moveArea(area, region)
                self.objInfo = obj4Move

                #  update aspirational

                if  (aspireOBJ - obj4Move) > epsilon:
                    aspireOBJ = obj4Move
                    aspireRegions = self.returnRegions()
                    region2AreaAspire = deepcopy(self.region2Area)
                    area2RegionAspire = deepcopy(self.area2Region)
                    improved = 1

                #  step 6

                currentSystem = self.returnRegions()
                nVisits = visitedSolutions.count(currentSystem)
                if nVisits == 0:

                    #  zoning system not found (go to step 10)
                    #  step 10

                    visitedSolutions.append(currentSystem)

                #  step 7

                elif nVisits > K1:
                    try:
                        nVisitsAll = allVisitedSolutions[currentSystem]
                    except:
                        nVisitsAll = 0
                    nVisitsAll =+ 1
                    allVisitedSolutions[currentSystem] = nVisitsAll
                    if nVisitsAll >= K2:

                        #  go to step 11
                        #  step 11a

                        visitedSolutions = []
                        self.objDict = makeObjDict(self)
                        self.allCandidates()
                        moveIndex = list(range(len(self.neighSolutions)))
                        np.random.suffle(moveIndex)
                        for move in moveIndex[0:int(1 + 0.5 * rAvg)]:
                            area, region = move
                            obj4Move = self.neighSolutions[move]
                            oldRegion = self.area2Region[area]
                            tabuList = self.updateTabuList((area,oldRegion), tabuList, tabuLength)
                            self.moveArea(area, region)
                            obj4Move = self.neighSolutions[move]

                            #  update aspirational

                            if (aspireOBJ-obj4Move) > epsilon:
                                aspireOBJ = obj4Move
                                aspireRegions = self.returnRegions()
                                region2AreaAspire = deepcopy(self.region2Area)
                                area2RegionAspire = deepcopy(self.area2Region)
                                improved = 1

                #  step 8

                elif nVisits < K1:
                    rAvg += 1
                    tabuLength = 1.1*tabuLength

                    #  step 9

                    if tabuLength > rAvg:
                        tabuLength = max(0.9 * tabuLength, 1)
                    tabuLength = int(round(tabuLength))

                    #  step 10

                    visitedSolutions.append(currentSystem)
                if improved == 1:
                    c = 1
                else:
                    c += 1

        self.objInfo = aspireOBJ
        self.regions = aspireRegions
        self.region2Area = deepcopy(region2AreaAspire)
        self.area2Region = deepcopy(area2RegionAspire)

    def moveArea(self, areaID, regionID):
        """
        Move an area to a region
        """
        oldRegion = self.area2Region[areaID]
        self.region2Area[oldRegion].remove(areaID)
        self.region2Area[regionID].append(areaID)
        self.area2Region[areaID] = regionID
        a = self.areas[areaID]
        toUpdate = [areaID] + a.neighs
        if self.objectiveFunctionType == "GWalt":
            self.NRegion[regionID] += a.data[0]
            self.NRegion[oldRegion] -= a.data[0]
        if self.numRegionsType == "EndogenousThreshold":
            self.regionValue[regionID] += self.areas[areaID].thresholdVar
            self.regionValue[oldRegion] -= self.areas[areaID].thresholdVar
        try:
            for index in range(1, len(a.data)):
                self.data[regionID][index - 1] += a.data[index] * a.data[0]
            #for index in range(1, len(a.data)):
                self.data[oldRegion][index - 1] -= a.data[index] *a.data[0]
        except:
            pass
        for area in toUpdate:
            regionIn = self.area2Region[area]
            areasIdsIn = self.region2Area[regionIn]
            areasInNow = [self.areas[aID] for aID in areasIdsIn]
            areasInRegion = set(areasIdsIn)
            aNeighs = set(self.areas[area].neighs)
            neighsInOther = aNeighs - areasInRegion # neighs of this area in other regions
            if len(neighsInOther) == 0 and area in self.intraBorderingAreas:
                self.intraBorderingAreas.pop(area)
            else:
                borderRegions = set([])
                for neigh in neighsInOther:
                    borderRegions = borderRegions | set([self.area2Region[neigh]])
                if area in self.intraBorderingAreas:
                    self.intraBorderingAreas.pop(area)
                self.intraBorderingAreas[area] = borderRegions
                #_tmp = set()
                #for neig in self.areas[area].neighs:
                #    _tmp.add(self.area2Region[neig])

                #_tmp.discard(self.area2Region[area])
                #self.intraBorderingAreas[area] = _tmp
        self.calcObj()

    def recoverFromExtendedMemory(self, extendedMemory):
        """
        Recover a solution form the extended memory
        """
        self.objInfo = extendedMemory.objInfo
        self.area2Region = extendedMemory.area2Region
        self.region2Area = extendedMemory.region2Area
        self.intraBorderingAreas = extendedMemory.intraBorderingAreas
