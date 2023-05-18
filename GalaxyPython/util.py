import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr as pearsonr
from tqdm import tqdm
import math

def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
	# Prefilter function
	if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
		raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
	id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
	adata._inplace_subset_obs(id_tmp)
	adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname 
	print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
   

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
	if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
		raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
	id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
	id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
	adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
	id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
	id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
	id_tmp=np.logical_and(id_tmp1,id_tmp2)
	adata._inplace_subset_var(id_tmp)


# def get_peaks(mz, intensity, threshold):
# 	deriv = np.divide(np.diff(intensity), np.diff(mz))
# 	cutpoint = np.quantile(np.abs(deriv),threshold)
# 	increase = deriv > cutpoint
# 	increase = np.insert(increase, 0, False)
# 	decrease = deriv < - cutpoint
# 	decrease = np.append(decrease, False)
# 	peak = np.logical_or(increase, decrease)
# 	return peak


# def point_clusters(points, percentile = 0.75):
# 	points_sorted = np.sort(points)
# 	dist_points = np.diff(points_sorted)
# 	dist_points_sort = np.quantile(dist_points, percentile)
# 	#
# 	clusters = []
# 	eps = dist_points_sort
# 	curr_point = points_sorted[0]
# 	curr_cluster = [curr_point]
# 	for point in list(points_sorted[1:]):
# 		if point <= curr_point + eps:
# 			curr_cluster.append(point)
# 		else:
# 			clusters.append(curr_cluster)
# 			curr_cluster = [point]
# 		curr_point = point
# 	clusters.append(curr_cluster)
# 	return clusters


def save_clusterresults (cluster, filepath):
	rangetxt = np.zeros((len(cluster),2))
	for i, points in enumerate(cluster):
		rangetxt[i,0] = points[0]
		rangetxt[i,1] = points[-1]
	np.savetxt("{filepath}.csv".format(filepath = filepath), rangetxt ,  delimiter= ",")

def rotate(l, n):
	return l[n:] + l[:n]

def group_range (cluster):
	lows = np.array([i[0] for i in cluster])
	ups = np.array([i[-1] for i in cluster])
	return [lows, ups]	

def in_range_lookup(mzvalues, cluster, expand = True):
	lows, ups = group_range(cluster)
	TFlist = [np.any((lows <= x) & (x <= ups)) for x in mzvalues]
	if expand:
		TFlistr = rotate(TFlist,1)
		TFlistl = rotate(TFlist,-1)
		newTFlist = [any((mz1, mz2, mz3))  for (mz1, mz2, mz3) in zip(TFlist, TFlistr, TFlistl) ]
	else:
		newTFlist = TFlist
	return newTFlist


def comp_clusters(TFlist, mzvalues):
	clusters = []
	curr_cluster = []
	for i, curr_point in enumerate(TFlist):
		if curr_point:
			curr_cluster.append(mzvalues[i])
		else:
			if curr_cluster:
				clusters.append(curr_cluster)
				curr_cluster = []
	return clusters


def find_nearest(array, value):
	"""
		find the index of the closest element to a value in an array.
		------------
		:param array: (iarray) an array that is searching from
		:param value: (float) the value that is looking for
		:return: (int) index of the nearest point
	"""
	nearest_idx = np.where(abs(array-value)==abs(array-value).min())[0] 
	return nearest_idx[0]

def get_unk_comp_clusters(ref_clusters, mzvalues):
	clusters = []
	mzvalues_array = np.array(mzvalues)
	mzvalues_array_float = np.asarray(mzvalues_array, dtype=np.float64, order='C')
	for i, curr_cluster in enumerate(ref_clusters):
		ref_ini_point = curr_cluster[0]
		indexnearest_unk = find_nearest(mzvalues_array_float, float(ref_ini_point))
		if ((indexnearest_unk+len(curr_cluster)) <= len(mzvalues)):
			clusters.append(mzvalues[range(indexnearest_unk,indexnearest_unk+len(curr_cluster))])
		curr_cluster = []
	return clusters


def spectrum_save (mz, intensity, name):
	originaltable= pd.DataFrame(np.array(mz))
	originaltable = originaltable.rename(columns = {0:"mz"})
	originaltable["intensity"] = intensity
	originaltable.to_csv ("{name}.csv".format(name = name))


def GKernal(x0, x, y, sig=1.):
    """
    creates gaussian kernel
    """
    ax = x - x0
    Kernal = np.exp(-0.5 * np.square(ax) / np.square(sig))
    Weighted = Kernal * y
    return np.sum(Weighted)  / np.sum(Kernal)


def gridding (mz, intensity, start = None, end = None, increment = 0.125, epsilon = 0.025):
    if start == None:
        start = math.floor(np.min(mz))-1
    if end == None:
        end = math.ceil(np.max(mz))+1
    target_grid = np.arange(start, end, increment)
    intensity_out = np.zeros(target_grid.shape[0])
    ## Fit a Gaussian Kernal Smoother (Kernal Regreassion)
    for i in range(target_grid.shape[0]):
        calculation = GKernal(target_grid[i], mz, intensity,  sig = epsilon)
        if calculation == calculation:
            intensity_out[i] = calculation
        else:
            intensity_out[i] = 0
    return [target_grid, intensity_out]




# def get_corr_peakgroup_prototype (mz_value1, mz_value2, cluster, meanspectrum1, meanspectrum2):
# 	"""
# 		Computer the similarity matrix accorss each peakgroup.
# 		------------
# 		:param mz_value1: (indexed dataFrame) the MZvalues of the unknown spectrum
# 		:param mz_value2: (indexed dataFrame) the MZvalues of the reference spectrum
# 		:parameter cluster: (list) the cluster object output from point_clusters()
# 		:return: (Tensor) List of latent codes
# 	"""
# 	mz_unk = list(mz_value1["m/z"])
# 	select_list_1 = in_range_lookup(mz_unk, cluster)
# 	mz_ref = list(mz_value2["m/z"])
# 	select_list_2 = in_range_lookup(mz_ref, cluster)
# 	select_list = [(e_i|e_j) for (e_i,e_j) in zip(select_list_1, select_list_2)]
# 	#
# 	ref_clusters = comp_clusters(select_list, mz_value2.index)
# 	unk_clusters = get_unk_comp_clusters(ref_clusters, mz_value1.index)
# 	nclusters = len(ref_clusters)
# 	#
# 	midpoints_unk = [clusteri[len(clusteri)//2] for clusteri in unk_clusters]
# 	PearsonMatrix = np.zeros((nclusters, nclusters))
# 	mzdiffs = []
# 	#
# 	for i_unk in tqdm(range(nclusters)):
# 		for i_ref in range(nclusters):
# 			midindexint = list(mz_value1.index).index(midpoints_unk[i_unk])
# 			indexlist_unk = np.arange(midindexint-len(ref_clusters[i_ref])//2,midindexint + len(ref_clusters[i_ref]) - len(ref_clusters[i_ref])//2)
# 			headindexint_ref = list(mz_value2.index).index(ref_clusters[i_ref][0])
# 			indexlist_ref = np.arange(headindexint_ref, headindexint_ref + len(ref_clusters[i_ref])) # [i for i in range(len(mz_value2.index)) if mz_value2.index[i] in ref_clusters[i_ref]]
# 			specsub_unk = meanspectrum1[indexlist_unk.tolist()] 
# 			specsub_ref = meanspectrum2[indexlist_ref.tolist()]
# 			PearsonMatrix[i_unk, i_ref] = pearsonr(specsub_unk, specsub_ref)[0]
# 			mzdiff = mz_value1["m/z"][midindexint] - mz_value2["m/z"][list(mz_value2.index).index(ref_clusters[i_ref][len(ref_clusters[i_ref])//2])]
# 			mzdiffs.append(mzdiff)
# 	return [PearsonMatrix, mzdiffs]



# def get_corr_peakgroup_refined (mz_value1, mz_value2, cluster, meanspectrum1, meanspectrum2):
# 	"""
# 		Computer the similarity matrix accorss each peakgroup.
# 		------------
# 		:param mz_value1: (indexed dataFrame) the MZvalues of the unknown spectrum
# 		:param mz_value2: (indexed dataFrame) the MZvalues of the reference spectrum
# 		:parameter cluster: (list) the cluster object output from point_clusters()
# 		:return: (Tensor) List of latent codes
# 	"""
# 	mz_unk = list(mz_value1["m/z"])
# 	select_list_1 = in_range_lookup(mz_unk, cluster)
# 	mz_ref = list(mz_value2["m/z"])
# 	select_list_2 = in_range_lookup(mz_ref, cluster)
# 	select_list = [(e_i|e_j) for (e_i,e_j) in zip(select_list_1, select_list_2)]
# 	#
# 	ref_clusters = comp_clusters(select_list, mz_value2.index)
# 	unk_clusters = get_unk_comp_clusters(ref_clusters, mz_value1.index)
# 	nclusters = len(ref_clusters)
# 	#
# 	midpoints_unk = [clusteri[len(clusteri)//2] for clusteri in unk_clusters]
# 	PearsonMatrix = np.zeros((nclusters, nclusters))
# 	PearsonMatrixFull = np.zeros((nclusters*9, nclusters*9))
# 	#
# 	# nclusters = 10  # trail
# 	for i_unk in tqdm(range(nclusters)):
# 		for i_ref in range(max(i_unk-2,0),min(i_unk+3,nclusters)):
# 			midindexint = list(mz_value1.index).index(midpoints_unk[i_unk])
# 			headindexint_ref = list(mz_value2.index).index(ref_clusters[i_ref][0])
# 			mzdiff = mz_value1["m/z"][midindexint] - mz_value2["m/z"][list(mz_value2.index).index(ref_clusters[i_ref][len(ref_clusters[i_ref])//2])]
# 			Pearson_Record = -2
# 			for slide_unk in range(-4,5):
# 				for slide_ref in range(-4,5):
# 					indexlist_unk = slide_unk + np.arange(midindexint-len(ref_clusters[i_ref])//2,midindexint + len(ref_clusters[i_ref]) - len(ref_clusters[i_ref])//2)
# 					indexlist_ref = slide_ref + np.arange(headindexint_ref, headindexint_ref + len(ref_clusters[i_ref])) # [i for i in range(len(mz_value2.index)) if mz_value2.index[i] in ref_clusters[i_ref]]
# 					specsub_unk = meanspectrum1[indexlist_unk.tolist()] 
# 					specsub_ref = meanspectrum2[indexlist_ref.tolist()]
# 					PearsonMatrixFull[i_unk*9 + slide_unk + 4, i_ref*9 + slide_ref + 4] = pearsonr(specsub_unk, specsub_ref)[0]
# 					if PearsonMatrixFull[i_unk*9 + slide_unk + 4, i_ref*9 + slide_ref + 4] > Pearson_Record:
# 						Pearson_Record = PearsonMatrixFull[i_unk*9 + slide_unk+ 4, i_ref*9 + slide_ref+ 4]
# 			PearsonMatrix[i_unk, i_ref] = Pearson_Record/(abs(mzdiff)+1) 
# 	return [PearsonMatrix, PearsonMatrixFull]



# def greedy_align (matrix): 
# 	"""
# 		Perform greedy algorithm to the adjusted similarity matrix and align the optimal group.
# 		------------
# 		:param matrix: (matrix) a similarity matrix between m/z groups clustered by each spectrum
# 		:return: (matrix n x 2) matrix of matching relationship for the groups
# 	"""
# 	align_results = np.empty((matrix.shape[0],2))
# 	i = 0
# 	ind = np.argwhere(matrix == matrix.max()) 
# 	maxvalue = matrix[ind[0,0],ind[0,1]]
# 	while maxvalue>0:
# 		align_results[i,:] = ind
# 		i = i + 1
# 		matrix[ind[0,0],:] = -1
# 		matrix[:,ind[0,1]] = -1
# 		ind = np.argwhere(matrix == matrix.max()) 
# 		maxvalue = matrix[ind[0,0],ind[0,1]]
# 	align_select = align_results[0:i,:]
# 	align_results_sort = align_results[align_results[:, 1].argsort()]
# 	return align_results_sort.astype(int)


# def fine_align (align_group, Pearson_full, mz_value1, mz_value2, cluster, threshould = 0.2, ignore = False):
# 	mz_unk = list(mz_value1["m/z"])
# 	select_list_1 = in_range_lookup(mz_unk, cluster)
# 	mz_ref = list(mz_value2["m/z"])
# 	select_list_2 = in_range_lookup(mz_ref, cluster)
# 	select_list = [(e_i|e_j) for (e_i,e_j) in zip(select_list_1, select_list_2)]
# 	#
# 	ref_clusters = comp_clusters(select_list, mz_value2.index)
# 	unk_clusters = get_unk_comp_clusters(ref_clusters, mz_value1.index)
# 	nclusters = len(ref_clusters)
# 	midpoints_unk = [clusteri[len(clusteri)//2] for clusteri in unk_clusters]
# 	#
# 	aligned_mz_clusters_ref = []
# 	aligned_mz_clusters_unk = []
# 	changerecord = []
# 	#
# 	for i in tqdm(range(align_group.shape[0])):
# 		submatrix = Pearson_full[(align_group[i,0]*9):(align_group[i,0]*9 + 9), (align_group[i,1]*9):(align_group[i,1]*9 + 9)]
# 		diagnallist = [np.mean(np.diag(submatrix, k=i)) for i in range(-4,5)]
# 		if (max(diagnallist)>threshould)|ignore:
# 			change = -4 + diagnallist.index(max(diagnallist))
# 			midindexint = list(mz_value1.index).index(midpoints_unk[align_group[i,0]])
# 			headindexint_ref = list(mz_value2.index).index(ref_clusters[align_group[i,1]][0])
# 			indexlist_ref = np.arange(headindexint_ref, headindexint_ref + len(ref_clusters[align_group[i,1]])) # [i for i in range(len(mz_value2.index)) if mz_value2.index[i] in ref_clusters[i_ref]]
# 			indexlist_unk = - change + np.arange(midindexint-len(ref_clusters[align_group[i,0]])//2,midindexint + len(ref_clusters[align_group[i,0]]) - len(ref_clusters[align_group[i,0]])//2)
# 			aligned_mz_clusters_ref.append(indexlist_ref)
# 			aligned_mz_clusters_unk.append(indexlist_unk)
# 			changerecord.append(- change)
# 	return [aligned_mz_clusters_unk, aligned_mz_clusters_ref, changerecord]


# def shiftplot_data(finealignresult, mz_value2):
# 	shiftdata = np.zeros((len(finealignresult[1]),3))
# 	for i in tqdm(range(len(finealignresult[1]))):
# 		shiftdata[i,0] = mz_value2["m/z"][finealignresult[1][i]][0]
# 		shiftdata[i,1] = mz_value2["m/z"][finealignresult[1][i]][-1]
# 		shiftdata[i,2] = finealignresult[2][i]
# 	shiftdatadf = pd.DataFrame(shiftdata)
# 	shiftdatadf = shiftdatadf.rename(columns={0: 'mzvalue_start', 1: 'mzvalue_end', 2: 'shift_fromorigin'})
# 	shiftdatadf["shift_fromprev"] = np.concatenate(([shiftdatadf["shift_fromorigin"][0]],np.diff(shiftdatadf["shift_fromorigin"])))   
# 	return shiftdatadf

