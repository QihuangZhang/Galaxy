import pandas as pd
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr as pearsonr
from tqdm import tqdm
import random
from . util import *


class AnnDataMALDI(object):
    def __init__(self, AnnDataUnk, AnnDataRef):
        self.AnnDataUnk = AnnDataUnk
        self.AnnDataRef = AnnDataRef
        self.mz_valueUnk = AnnDataUnk.var
        self.mz_valueRef = AnnDataRef.var
        self.meanspectrumUnk = np.mean(AnnDataUnk.X, axis = 0)
        self.meanspectrumRef = np.mean(AnnDataRef.X, axis = 0)
    def ClusterPrep (self, cluster):
        """
            Computer the similarity matrix accorss each peakgroup.
            ------------
            :parameter cluster: (list) the cluster object output from point_clusters()
            :return: (Tensor) List of latent codes
        """
        mz_unk = list(self.mz_valueUnk["m/z"])
        select_list_1 = in_range_lookup(mz_unk, cluster)
        mz_ref = list(self.mz_valueRef["m/z"])
        select_list_2 = in_range_lookup(mz_ref, cluster)
        select_list = [(e_i|e_j) for (e_i,e_j) in zip(select_list_1, select_list_2)]
        #
        self.ref_clusters = comp_clusters(select_list, self.mz_valueRef.index)
        self.unk_clusters = get_unk_comp_clusters(self.ref_clusters, self.mz_valueUnk.index)
        if (len(self.ref_clusters)>len(self.unk_clusters)):
            self.ref_clusters = self.ref_clusters[0:len(self.unk_clusters)]
        self.nclusters = len(self.ref_clusters)
        #
        self.midpoints_unk = [clusteri[len(clusteri)//2] for clusteri in self.unk_clusters]
        #
    def get_corr_peakgroup_refined (self, cluster):
        """
            Computer the similarity matrix accorss each peakgroup.
            ------------
            :param mz_valueUnk: (indexed dataFrame) the MZvalues of the unknown spectrum
            :param mz_valueRef: (indexed dataFrame) the MZvalues of the reference spectrum
            :parameter cluster: (list) the cluster object output from point_clusters()
            :return: (Tensor) List of latent codes
        """
        self.cluster = cluster
        self.ClusterPrep (cluster)
        PearsonMatrix = np.zeros((self.nclusters, self.nclusters))
        PearsonMatrixFull = np.zeros((self.nclusters*9, self.nclusters*9))
        # nclusters = 10  # trail
        for i_unk in tqdm(range(self.nclusters)):
            for i_ref in range(max(i_unk-2,0),min(i_unk+3,self.nclusters)):
                midindexint = list(self.mz_valueUnk.index).index(self.midpoints_unk[i_unk])
                headindexint_ref = list(self.mz_valueRef.index).index(self.ref_clusters[i_ref][0])
                mzdiff = self.mz_valueUnk["m/z"][midindexint] - self.mz_valueRef["m/z"][list(self.mz_valueRef.index).index(self.ref_clusters[i_ref][len(self.ref_clusters[i_ref])//2])]
                Pearson_Record = -2
                Can_align = True
                for slide_unk in range(-4,5):
                    for slide_ref in range(-4,5):
                        indexlist_unk = slide_unk + np.arange(midindexint-len(self.ref_clusters[i_ref])//2,midindexint + len(self.ref_clusters[i_ref]) - len(self.ref_clusters[i_ref])//2)
                        indexlist_ref = slide_ref + np.arange(headindexint_ref, headindexint_ref + len(self.ref_clusters[i_ref])) # [i for i in range(len(self.mz_valueRef.index)) if self.mz_valueRef.index[i] in ref_clusters[i_ref]]
                        try:
                            specsub_unk = self.meanspectrumUnk[indexlist_unk.tolist()] 
                            specsub_ref = self.meanspectrumRef[indexlist_ref.tolist()]
                            PearsonMatrixFull[i_unk*9 + slide_unk + 4, i_ref*9 + slide_ref + 4] = pearsonr(specsub_unk, specsub_ref)[0]
                            if PearsonMatrixFull[i_unk*9 + slide_unk + 4, i_ref*9 + slide_ref + 4] > Pearson_Record:
                                Pearson_Record = PearsonMatrixFull[i_unk*9 + slide_unk+ 4, i_ref*9 + slide_ref+ 4]
                        except IndexError:
                            Can_align = False
                if Can_align:
                    PearsonMatrix[i_unk, i_ref] = Pearson_Record/(abs(mzdiff)+1)
                else:
                    PearsonMatrix[i_unk, i_ref] = -1
        self.PearsonMatrix = PearsonMatrix
        self.PearsonMatrixFull = PearsonMatrixFull
    def group_align_onestep (self, matrix, criteria, origin):
        if all(matrix.shape):
            ind = np.argwhere(matrix == matrix.max())
            maxvalue = matrix[ind[0,0],ind[0,1]]
            ind0 = np.array([ind[0,:]])
            if maxvalue>criteria:
                record = list(ind0 + origin)
                results1 = self.group_align_onestep (matrix[0:ind0[0,0],0:ind0[0,1]], criteria, origin )
                results2 = self.group_align_onestep (matrix[(ind0[0,0]+1):,(ind0[0,1]+1):], criteria, ind0 + origin + 1 )
                record = record + results1 + results2
                return record
            else:
                return []
        return []
    def greedy_match (self, criteria = 0): 
        """
            Perform greedy algorithm to the adjusted similarity matrix and align the optimal group.
            ------------
            :param matrix: (matrix) a similarity matrix between m/z groups clustered by each spectrum
            :return: (matrix n x 2) matrix of matching relationship for the groups
        """
        alignlist = self.group_align_onestep (self.PearsonMatrix, criteria, origin = (0,0))
        align_results = np.empty((len(alignlist),2))
        for i, ind in  enumerate(alignlist):
            align_results[i,0] = ind[0]
            align_results[i,1] = ind[1]
        align_results_sort = align_results[align_results[:, 1].argsort()]
        self.align_group =  align_results_sort.astype(int)
        self.align_group = np.unique(self.align_group, axis = 0)
        self.nclusters = self.align_group.shape[0]
    def fine_align (self, threshould = 0.2, ignore = False):
        aligned_mz_clusters_ref = []
        aligned_mz_clusters_unk = []
        changerecord = []
        #
        for i in tqdm(range(self.align_group.shape[0])):
            submatrix = self.PearsonMatrixFull[(self.align_group[i,0]*9):(self.align_group[i,0]*9 + 9), (self.align_group[i,1]*9):(self.align_group[i,1]*9 + 9)]
            diagnallist = [np.nanmean(np.diag(submatrix, k=i)) for i in range(-4,5)]
            diagnallist = [-1 if value!=value else value for value in diagnallist]  # filter out the value with NaN and change it into 0
            if (max(diagnallist)>threshould)|ignore:
                change = -4 + diagnallist.index(max(diagnallist))
                midindexint = list(self.mz_valueUnk.index).index(self.midpoints_unk[self.align_group[i,0]])
                headindexint_ref = list(self.mz_valueRef.index).index(self.ref_clusters[self.align_group[i,1]][0])
                indexlist_ref = np.arange(headindexint_ref, headindexint_ref + len(self.ref_clusters[self.align_group[i,1]])) # [i for i in range(len(mz_valueRef.index)) if mz_valueRef.index[i] in self.ref_clusters[i_ref]]
                indexlist_unk = - change + np.arange(midindexint-len(self.ref_clusters[self.align_group[i,0]])//2,midindexint + len(self.ref_clusters[self.align_group[i,0]]) - len(self.ref_clusters[self.align_group[i,0]])//2)
                aligned_mz_clusters_ref.append(indexlist_ref)
                aligned_mz_clusters_unk.append(indexlist_unk)
                changerecord.append(- change)
        self.aligned_mz_clusters_unk = aligned_mz_clusters_unk
        self.aligned_mz_clusters_ref = aligned_mz_clusters_ref
        self.changerecord = changerecord
    def summarize (self):
        self.unknownalign = np.concatenate(self.aligned_mz_clusters_unk)
        self.referenalign = np.concatenate(self.aligned_mz_clusters_ref)


class MALDI_SIM(object):
    def __init__(self, AnnDataMALDI):
        self.origindata = AnnDataMALDI.AnnDataRef.copy()
        self.aligned_mz_clusters_ref = AnnDataMALDI.aligned_mz_clusters_ref.copy()
        self.changerecord = AnnDataMALDI.changerecord.copy()
        self.shiftplot_data()
    def shiftplot_data(self):
        self.mz_valueRef = self.origindata.var.copy()
        shiftdata = np.zeros((len(self.aligned_mz_clusters_ref),3))
        for i in tqdm(range(len(self.aligned_mz_clusters_ref))):
            shiftdata[i,0] = self.mz_valueRef["m/z"][self.aligned_mz_clusters_ref[i]][0]
            shiftdata[i,1] = self.mz_valueRef["m/z"][self.aligned_mz_clusters_ref[i]][-1]
            shiftdata[i,2] = self.changerecord[i]
        shiftdatadf = pd.DataFrame(shiftdata)
        shiftdatadf = shiftdatadf.rename(columns={0: 'mzvalue_start', 1: 'mzvalue_end', 2: 'shift_fromorigin'})
        shiftdatadf["shift_fromprev"] = np.concatenate(([shiftdatadf["shift_fromorigin"][0]],np.diff(shiftdatadf["shift_fromorigin"])))   
        self.shiftdatadf = shiftdatadf
    def addin (self, add_at_mz, addnumber):
        nearest_idx = np.where(abs(self.mz_valueRef-add_at_mz)==abs(self.mz_valueRef-add_at_mz).min())[0].max()
        for i in range(addnumber):
            self.arraydata = np.insert(self.arraydata, nearest_idx+1, self.arraydata[:,nearest_idx], axis=1) 
            self.mz_valueRef = np.insert(self.mz_valueRef, nearest_idx+1, self.mz_valueRef[nearest_idx]+ (i+1)*0.0001) 
    def delout (self, del_at_mz, delnumber):
        for i in range(delnumber):
            nearest_idx = np.where(abs(self.mz_valueRef-del_at_mz)==abs(self.mz_valueRef-del_at_mz).min())[0].max()
            self.arraydata = np.delete(self.arraydata, nearest_idx, axis=1) 
            self.mz_valueRef = np.delete(self.mz_valueRef, nearest_idx)
    def get_at_mz (self, idx):
        if idx >0:
            return (self.shiftdatadf.iloc[idx-1,1] + self.shiftdatadf.iloc[idx,0])/2
        else:
            return self.shiftdatadf.iloc[idx,0]/2
    def region_shuffle (self, ary, nregion):
        arylength = len(ary)
        intervals = list((np.array(range(1,nregion))*arylength/nregion).astype("int"))
        if nregion > 1:
            listary = np.split(ary, intervals, axis=0)
        else:
            listary = [ary]
        [random.shuffle(listaryi) for listaryi in  listary]
        newary = np.concatenate(listary)
        return newary
    def SIMULATEdata (self, shuffle, sigma, nregion):
        self.mz_valueRef = np.array(self.origindata.var["m/z"]).copy()
        self.arraydata = self.origindata.X.copy()
        unitdiff = self.mz_valueRef[1] - self.mz_valueRef[0]
        shift_resample = np.array(self.shiftdatadf["shift_fromprev"]).copy()
        if shuffle == True:
            shift_resample = self.region_shuffle(shift_resample, nregion = nregion)
        ## Shifting the unit
        for i in tqdm(range(self.shiftdatadf.shape[0])):
            if shift_resample[i]>0:
                self.addin (add_at_mz = self.get_at_mz(i), addnumber = int(shift_resample[i]))
            if shift_resample[i]<0:
                self.delout (del_at_mz = self.get_at_mz(i), delnumber = int(abs(shift_resample[i])))
        ## Adding Noises
        noise = np.random.lognormal(mean = 0.0, sigma = sigma, size = self.arraydata.shape[0] * self.arraydata.shape[1])
        noise.shape = self.arraydata.shape
        self.newarray = self.arraydata + noise
        self.truemz = self.mz_valueRef
        self.newmz = np.array(range(0,len(self.truemz))) * unitdiff + self.truemz[0]
        self.currentshift = shift_resample
        self.shiftdatadf["currentshift_fromprev"] = shift_resample
        self.shiftdatadf["currentshift_fromorg"] = np.cumsum(shift_resample)
        # mzdf = pd.DataFrame({ "truemz": SD.truemz, "newmz":SD.newmz})
    def getAnnSim (self, shuffle = False, sigma = 0.1, nregion = 1):
        self.SIMULATEdata(shuffle, sigma, nregion)
        mz_value = pd.DataFrame(list(self.newmz), index = list(self.newmz)).astype('float')
        mz_value = mz_value.rename(columns = {0:"m/z"})
        MALDISimData = sc.AnnData(X = self.newarray, var = mz_value, obs = self.origindata.obs)
        return MALDISimData
    def MSE (self, unkidx, refidx):
        refmz = np.array(self.origindata.var["m/z"]).copy()
        MSE = (np.square(self.truemz[unkidx] - refmz[refidx]))
        return [MSE, self.truemz[unkidx], refmz[refidx]]


class PGmzalign(MALDI_SIM):
    def __init__(self, AnnDataMALDI):
        self.unknowndata = AnnDataMALDI.AnnDataUnk.copy()
        self.aligned_mz_clusters_unk = AnnDataMALDI.aligned_mz_clusters_unk.copy()
        super(PGmzalign, self).__init__(AnnDataMALDI)
    def shiftplot_data(self):
        self.mz_valueRef = self.origindata.var.copy()
        shiftdata = np.zeros((len(self.aligned_mz_clusters_ref),4))
        for i in tqdm(range(len(self.aligned_mz_clusters_ref))):
            shiftdata[i,0] = self.mz_valueRef["m/z"][self.aligned_mz_clusters_ref[i]][0]
            shiftdata[i,1] = self.mz_valueRef["m/z"][self.aligned_mz_clusters_ref[i]][-1]
            shiftdata[i,2] = self.changerecord[i]
            shiftdata[i,3] = self.aligned_mz_clusters_ref[i][0] - self.aligned_mz_clusters_unk[i][0]
        shiftdatadf = pd.DataFrame(shiftdata)
        shiftdatadf = shiftdatadf.rename(columns={0: 'mzvalue_start', 1: 'mzvalue_end', 2: 'shift_fromorigin', 3: 'shift_unit_fromorigin'})
        shiftdatadf["shift_fromprev"] = np.concatenate(([shiftdatadf["shift_fromorigin"][0]],np.diff(shiftdatadf["shift_fromorigin"])))
        shiftdatadf["shift_unit_fromprev"] = np.concatenate(([shiftdatadf["shift_unit_fromorigin"][0]],np.diff(shiftdatadf["shift_unit_fromorigin"])))      
        self.shiftdatadf = shiftdatadf
    def SIMULATEdata (self):
        self.mz_valueRef = np.array(self.unknowndata.var["m/z"]).copy()
        self.arraydata = self.unknowndata.X.copy()
        shift_resample = np.array(self.shiftdatadf["shift_unit_fromprev"]).copy()
        ## Shifting the unit
        for i in tqdm(range(self.shiftdatadf.shape[0])):
            if shift_resample[i]>0:
                self.addin (add_at_mz = self.get_at_mz(i), addnumber = int(abs(shift_resample[i])))
            if shift_resample[i]<0:
                self.delout (del_at_mz = self.get_at_mz(i), delnumber = int(abs(shift_resample[i])))
        self.newmz = np.array(self.origindata.var["m/z"]).copy()
        self.currentshift = shift_resample
        self.shiftdatadf["currentshift_fromprev"] = shift_resample
        self.shiftdatadf["currentshift_fromorg"] = np.cumsum(shift_resample)
    def getAnnSim (self):
        self.SIMULATEdata()
        mz_value = pd.DataFrame(list(self.newmz), index = list(self.newmz)).astype('float')
        mz_value = mz_value.rename(columns = {0:"m/z"})
        dim = min(self.arraydata.shape[1], mz_value.shape[0])
        MALDISimData = sc.AnnData(X = self.arraydata[:,range(dim)], var = mz_value.iloc[range(dim),:], obs = self.unknowndata.obs)
        return MALDISimData





# ----------------------------------------------------------------------------------
# The following classes are for peak calling which is not essential of the method
# ----------------------------------------------------------------------------------


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


