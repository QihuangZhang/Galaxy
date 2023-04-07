import numpy as np
from . util import *


class PeakCalling(object):
    def __init__(self, AnnDataUnk, AnnDataRef):
        self.AnnDataUnk = AnnDataUnk
        self.AnnDataRef = AnnDataRef
        self.mz_valueUnk = AnnDataUnk.var
        self.mz_valueRef = AnnDataRef.var
        self.meanspectrumUnk = np.mean(AnnDataUnk.X, axis = 0)
        self.meanspectrumRef = np.mean(AnnDataRef.X, axis = 0)
    def get_peaks(self, mz, intensity, threshold):
        deriv = np.divide(np.diff(intensity), np.diff(mz))
        cutpoint = np.quantile(np.abs(deriv),threshold)
        increase = deriv > cutpoint
        increase = np.insert(increase, 0, False)
        decrease = deriv < - cutpoint
        decrease = np.append(decrease, False)
        peak = np.logical_or(increase, decrease)
        ### Exclude the single peak
        e_left = np.empty_like(peak)
        e_right = np.empty_like(peak)
        e_left[-1] = False
        e_left[:-1] = peak[1:]
        e_right[0] = False
        e_right[1:] = peak[:-1]
        simplet = np.logical_and(peak, np.logical_not(np.logical_or(e_left, e_right))) 
        d_left = np.empty_like(simplet)
        d_right = np.empty_like(simplet)
        d_left[-1] = False
        d_left[:-1] = simplet[1:]
        d_right[0] = False
        d_right[1:] = simplet[:-1]
        peak = np.logical_or(peak, np.logical_or(d_left, d_right))
        return peak
    def point_clusters(self, points, percentile = 0.75):
        points_sorted = np.sort(points)
        dist_points = np.diff(points_sorted)
        dist_points_sort = np.quantile(dist_points, percentile)
        #
        clusters = []
        eps = dist_points_sort
        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in list(points_sorted[1:]):
            if point <= curr_point + eps:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)
        return clusters
    def callpeak (self, threshold = 0.9):
        self.peakUnk = self.get_peaks(mz = np.array(self.mz_valueUnk["m/z"]),  intensity = self.meanspectrumUnk, threshold = threshold)
        self.peakRef = self.get_peaks(mz = np.array(self.mz_valueRef["m/z"]),  intensity = self.meanspectrumRef, threshold = threshold)
        self.meanspectrumUnkpeak = self.meanspectrumUnk[self.peakUnk]
        self.meanspectrumRefpeak = self.meanspectrumRef[self.peakRef]
    def grouppeaks (self, percentile = 0.9):
        self.clusterUnk = self.point_clusters (np.array(self.mz_valueUnk["m/z"][list(self.peakUnk)]), percentile = percentile)
        self.clusterRef = self.point_clusters (np.array(self.mz_valueRef["m/z"][list(self.peakRef)]), percentile = percentile)
        self.mzcombine = np.concatenate((np.array(self.mz_valueUnk["m/z"][list(self.peakUnk)]), np.array(self.mz_valueRef["m/z"][list(self.peakRef)])))
        self.jointcluster = self.point_clusters (np.sort(self.mzcombine), percentile)
    def filter_clusters (self, mzboundary):
        self.jointcluster =  [ clusteri for clusteri in self.jointcluster if clusteri[-1]<mzboundary ]
    def importpeak (self, peakUnk, peakRef):
        self.peakUnk =  np.zeros( self.mz_valueUnk.shape[0], dtype = bool)
        self.peakRef = self.peakUnk
        for i in range(peakUnk.shape[0]):
            self.peakUnk[peakUnk[i]] = True
        for j in range(peakRef.shape[0]):
            self.peakRef[peakRef[j]] = True
        self.meanspectrumUnkpeak = self.meanspectrumUnk[self.peakUnk]
        self.meanspectrumRefpeak = self.meanspectrumRef[self.peakRef]


class PeakCalling_single(PeakCalling):
    def __init__(self, AnnData):
        super().__init__(AnnData, AnnData)
    def callpeak (self, threshold = 0.9):
        self.peakRef = self.get_peaks(mz = np.array(self.mz_valueRef["m/z"]),  intensity = self.meanspectrumRef, threshold = threshold)
        self.meanspectrumRefpeak = self.meanspectrumRef[self.peakRef]
    def grouppeaks (self, percentile = 0.9):
        self.clusterRef = self.point_clusters (np.array(self.mz_valueRef["m/z"][list(self.peakRef)]), percentile)
        self.jointcluster = self.clusterRef
    def filter_clusters (self, mzboundary):
        self.jointcluster =  [ clusteri for clusteri in self.jointcluster if clusteri[-1]<mzboundary ]
    def importpeak (self, peakUnk, peakRef):
        self.peakRef = self.peakUnk
        for j in range(peakRef.shape[0]):
            self.peakRef[peakRef[j]] = True
        self.meanspectrumRefpeak = self.meanspectrumRef[self.peakRef]
   

class PeakCallingmv(object):
    def __init__(self, AnnDataUnk, AnnDataRef):
        self.AnnDataUnk = AnnDataUnk
        self.AnnDataRef = AnnDataRef
        self.mz_valueUnk = AnnDataUnk.var
        self.mz_valueRef = AnnDataRef.var
        self.meanspectrumUnk = np.mean(AnnDataUnk.X, axis = 0)
        self.meanspectrumRef = np.mean(AnnDataRef.X, axis = 0)
        self.meanspectrumUnkmv = np.convolve(self.meanspectrumUnk, np.ones(3)/3, mode='valid')
        self.meanspectrumRefmv = np.convolve(self.meanspectrumRef, np.ones(3)/3, mode='valid')
        self.mz_valueUnk_mv = self.mz_valueUnk[1:(self.mz_valueUnk.shape[0]-1)]
        self.mz_valueRef_mv = self.mz_valueRef[1:(self.mz_valueRef.shape[0]-1)]
    def get_peaks(self, mz, intensity, threshold):
        deriv = np.divide(np.diff(intensity), np.diff(mz))
        cutpoint = np.quantile(np.abs(deriv),threshold)
        increase = deriv > cutpoint
        increase = np.insert(increase, 0, False)
        decrease = deriv < - cutpoint
        decrease = np.append(decrease, False)
        peak = np.logical_or(increase, decrease)
        return peak
    def point_clusters(self, points, percentile = 0.75):
        points_sorted = np.sort(points)
        dist_points = np.diff(points_sorted)
        dist_points_sort = np.quantile(dist_points, percentile)
        #
        clusters = []
        eps = dist_points_sort
        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in list(points_sorted[1:]):
            if point <= curr_point + eps:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)
        return clusters
    def callpeak (self, threshold = 0.9):
        self.peakUnk = self.get_peaks(mz = np.array(self.mz_valueUnk_mv["m/z"]),  intensity = self.meanspectrumUnkmv, threshold = threshold)
        self.peakRef = self.get_peaks(mz = np.array(self.mz_valueRef_mv["m/z"]),  intensity = self.meanspectrumRefmv, threshold = threshold)
        self.meanspectrumUnkpeak = self.meanspectrumUnkmv[self.peakUnk]
        self.meanspectrumRefpeak = self.meanspectrumRefmv[self.peakRef]
    def grouppeaks (self, percentile = 0.9):
        self.clusterUnk = self.point_clusters (np.array(self.mz_valueUnk_mv["m/z"][list(self.peakUnk)]))
        self.clusterRef = self.point_clusters (np.array(self.mz_valueRef_mv["m/z"][list(self.peakRef)]))
        self.mzcombine = np.concatenate((np.array(self.mz_valueUnk_mv["m/z"][list(self.peakUnk)]), np.array(self.mz_valueRef_mv["m/z"][list(self.peakRef)])))
        self.jointcluster = self.point_clusters (np.sort(self.mzcombine), percentile)
    def filter_clusters (self, mzboundary):
        self.jointcluster =  [ clusteri for clusteri in self.jointcluster if clusteri[-1]<mzboundary ]


