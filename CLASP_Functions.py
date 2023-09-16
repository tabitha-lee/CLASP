# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 13:50:17 2022

@author: tabyl
"""

#Functions in CLASP
######---------------------------------------------------------------------------------------------------########
#Spatial and Value Cluster Functions 
######---------------------------------------------------------------------------------------------------########

#Find Bins for Max Points
#Freedman-Diaconis's Rule
def n_bin(x):
    c=2.5
    h=c*stats.iqr(x,nan_policy='omit')/(len(x))**(1/3)
    if h == 0:
        x = np.array(pd.DataFrame(x).replace(0,np.nan))
        h=c*stats.iqr(x,nan_policy='omit')/(len(x))**(1/3)
    nbin = round((np.nanmax(x) - np.nanmin(x)) / h)
    return(nbin)

#Find Bandwith for Kernel Density Smoothing
def n_bandwith(x):
    s = np.std(x)
    n_bandwith = (min((0.9*s), (2/3)*stats.iqr(x,nan_policy='omit'))/(len(x))**(1/5))
    return n_bandwith

def geodesic_point_buffer(lat, lon, km):
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    cir = transform(project, buf).exterior.coords[:] #My alteration
    return list(map(list, zip(*cir))) #My alteration-to get the points to be in list[lat,lon]

#Find the points within the Grid Boxes
#Bounds: LatMin, LonMax, LatMin, LatMax, Lon_Center_Range, Lat_Center_Range
#From: https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy
def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                    max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter

#https://stackoverflow.com/questions/46335488/how-to-efficiently-find-the-bounding-box-of-a-collection-of-points
def bounding_box_naive(center_lon,center_lat,points_lon,points_lat,latinterval, loninterval, lonmin, lonmax, latmin, latmax):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    #Use Reverse mid point formula
    min_y0 = center_lat - (latinterval/2) #LatMin of bounding box
    max_y0 = center_lat + (latinterval/2) #LatMax of bounding box
    min_x0 = center_lon - (loninterval/2) #LonMin of bounding box
    max_x0 = center_lon + (loninterval/2) #LonMax of bounding box
    
    min_y = points_lat[np.where(points_lat == min_y0)[0]]
    max_y = points_lat[np.where(points_lat == max_y0)[0]]
    min_x = points_lon[np.where(points_lon == min_x0)[0]]
    max_x = points_lon[np.where(points_lon == max_x0)[0]]

    if min_x.size == 0:
        min_x = lonmin
    if max_x.size == 0:
        max_x = lonmax
    if min_y.size == 0:
        min_y = latmin
    if max_y.size == 0:
        max_y = latmax
    
    return float(min_y), float(max_y), float(min_x), float(max_x)

#Get Radius Values
def GetRadiusValues(km_distance, SourcePoint_InRegion):
    #Set distance in KM
    #km_distance = 10
    #Loop through Source Point Locations to get the buffer radius values
    circle_list = []
    for a,b in zip(SourcePoint_InRegion[:,0],SourcePoint_InRegion[:,1]):
        circle = geodesic_point_buffer(b, a, km_distance) #Lat,Lon,km
        circle_list.append(circle)
    return circle_list


#Find TROPOMI Lat/Lon Range that the Buffer Circles reach
def TROPOMI_Circles(km_distance, SourcePoint_InRegion, D1):
    Km_to_degrees_conversion = 0.01/1
    radius = km_distance*Km_to_degrees_conversion
    Mask = []
    for i in range(0,len(SourcePoint_InRegion)):
        center = np.array([SourcePoint_InRegion[:,0][i],SourcePoint_InRegion[:,1][i]])
        mask = (D1[:,1] - center[0])**2 + (D1[:,2] - center[1])**2 > radius**2
        Mask.append(pd.DataFrame(mask))
    return Mask

#Create Mask for in TROPOMI Circles    
def LargeMask(Mask):
     #Create one large mask 
    Mask_Total = Mask[0]
    for i in Mask:
        Mask_Total = Mask_Total & i
    return Mask_Total

#Check for Volume of Good Pixels
def CheckGoodPixels(D1):
    #Percent of 0 C Values
    num_nan = np.count_nonzero(np.isnan(D1[:,3]))
    #num_0s = sum(z == 0 for z in D1[:,3])
    per_0s = (num_nan/len((D1[:,3])))*100
    #If statement to check
    if per_0s > 85:
        print("Too low of good values!")
        return True
    return False #Means that we can use D1 because we have enough pixels to see good data

#Get breaks from piecewise function
def RCF_ThresholdPoints(points_plots,numbins):
    #Alternative way for Relative Cumulative Frequency  
    res = stats.cumfreq(points_plots,numbins=numbins) #'doane'
    x = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,res.cumcount.size)
    
    #PieceWise Linear Fit of Relative Cumulative Frequency Curve
    x=x
    y=res.cumcount
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    #3 breaks - this is variable!
    breaks = my_pwlf.fit(3)
    return breaks

#Remove background values
def RemoveBackground(maginflu,breaks,D1):
    if maginflu == 'y':
        points2 = D1[D1[:,4] >= breaks[2]]
    else: #if maginflu ==  'n':
        points2 = D1[D1[:,4] >= breaks[1]]
    return points2

#Get the Spatial Clusters Using DBSCAN
def GetSpatialClusters(points2,sp_res):
    clustering = DBSCAN(eps=2*sp_res, min_samples=1).fit(points2[:,2:4])
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print("Estimated number of Spatial clusters: %d" % n_clusters_)
   
    #Add cluster labels 
    points2_labels = np.column_stack((points2,labels))
    unique_labels = np.unique(points2_labels[:,5])
    return points2_labels, unique_labels, n_clusters_

#Check if Source Points are removed & Make necessary alterations to the data
##########################This is old 
def RemoveSourcePointClusters(points2_labels,source,unique_labels):    
    Source_Clus = np.empty(shape=(0,6))
    NoSource_Clus = np.empty(shape=(0,6))
    for k in unique_labels:
        indi_clus = points2_labels[np.where(points2_labels[:,5] == k)]
        Latmin_clus = min(indi_clus[:,3])
        Latmax_clus = max(indi_clus[:,3])
        Lonmin_clus = min(indi_clus[:,2])
        Lonmax_clus = max(indi_clus[:,2])
        
        if not 'SourcePoint_LonLat' in locals():
            NoSource_Clus = np.concatenate([NoSource_Clus, indi_clus], axis=0)
            #Check what user wants - Source/No Source
            if source == 'y':
                data_new =  points2_labels 
            if source == 'n':
                data_new = points2_labels 
                
        if 'SourcePoint_LonLat' in locals(): 
            #See if source points are in cluster bounds
            #Check All Source Points in the Region
            SourcePoint_InClus = SourcePoint_InRegion[(SourcePoint_InRegion[:,0]>= Lonmin_clus) & (SourcePoint_InRegion[:,0] <= Lonmax_clus) & (SourcePoint_InRegion[:,1] >= Latmin_clus) & (SourcePoint_InRegion[:,1] <= Latmax_clus)]
            if len(SourcePoint_InClus) != 0:
                 Source_Clus = np.concatenate([Source_Clus, indi_clus], axis=0)
            else:
                NoSource_Clus = np.concatenate([NoSource_Clus, indi_clus], axis=0)        
            #Check what user wants - Source/No Source
            if source == 'y':
                data_new =  points2_labels 
            else:
                data_new = NoSource_Clus     
        
        return data_new
    
#Bin Oversampled Values
def BinValues(points2_MagCol, numbins2=1):
    if numbins2 == 1:
        numbins2 = int(n_bin(points2_MagCol))
    else:
        numbins2 = numbins2
    #Histogram
    hist, bins = np.histogram(points2_MagCol, bins=numbins2)
    return bins
    
#Merge Points together based on distance & bin numbers
def MergeOnDistance(data_new, bins, sp_res, eps_scale): 
    #Add histogram delination to Cluster data
    points3 = np.column_stack((data_new,np.digitize(data_new[:,4],bins)))
    #Convert to list to do next step
    points3_list = list(zip(points3[:,0],points3[:,1],points3[:,2],points3[:,3],points3[:,4],points3[:,5],points3[:,6]))
    #Combine Points together
    #5 is a variable term: can be changed if want cluster closer or farther away
    #0.1 is the oversampled resolution 
    Eps1 = sp_res* eps_scale
    Clus=[]
    while len(points3_list):
        #for i in range(0,len(points1),1):
        locus = points3_list.pop() #points1[i,:]
        cluster = [x for x in points3_list if euclidean(locus[2:4],x[2:4]) <= Eps1]
        Clus.append(cluster+[locus])
        for x in cluster:
            points3_list.remove(x) #points1 = points1[points1 != x]
    return Clus
 
#Get Mean Lat,Lon, and Bin for each cluster
def ClusterMeans(Clus):
    Clus_Latmean = []
    Clus_Lonmean = []
    Clus_Binmean = []
    for i in Clus:
        df = np.array(i)
        Clus_Latmean.append(stats.mode(df[:,3])[0])
        Clus_Lonmean.append(stats.mode(df[:,2])[0])
        Clus_Binmean.append(stats.mode(df[:,6])[0])
    
    #Reorder the Clus_Binmean to get increasing values
    for i,j in zip(np.unique(Clus_Binmean),range(1,len(np.unique(Clus_Binmean))+1)):
        Clus_Binmean = np.where(Clus_Binmean != i, np.array(Clus_Binmean),j)
    return Clus_Latmean,Clus_Lonmean,Clus_Binmean
    
#Get Nearest Neighbors of Clusters
def Clus_NearestNeighbors(data_new,bins,Clus_Lonmean,Clus_Latmean,Clus_Binmean):
    #Replace NaN as 0's - to remove werid plotting
    where_are_NaNs = np.isnan(data_new[:,4])
    data_new[:,4][where_are_NaNs] = 0
    
    #Find nearest neighbors of points
    D_tree_d = np.column_stack((data_new[:,2:4], np.digitize(data_new[:,4],bins)))
    Max_tree_d = np.column_stack((Clus_Lonmean,Clus_Latmean,Clus_Binmean)) #Use the averaged local max C
    
    #Commenting this out for now - checking to see if new changes can allow this to be removed
    # #Check to see if we have empty array
    # if Max_tree_d.size == 0:
    #     print("There were no good values to cluster around aka not enough datapoints")
    #     continue
    
    #Ball Tree Method (Ball Tree is a type of nearest neighbors section)
    tree_d = BallTree(Max_tree_d, leaf_size=1)
    #Find closest points
    #Could also do query_radius and define a radius
    k_neighbors = 1
    distances_d, indices_d = tree_d.query(D_tree_d, k=k_neighbors)
    
    Points_closest_d = []
    for cl in indices_d.transpose():
        #cl is the index number
        closest_latlon = np.row_stack(Max_tree_d[o] for o in cl)
        #closest_distance = pd.DataFrame(np.row_stack(Max_tree[o] for o in cl))
        Points_closest_d.append(closest_latlon)
        
    D1_points_d = np.column_stack((data_new, distances_d, np.array(Points_closest_d).reshape(len(D_tree_d),3)))
    
    bin_edges = np.histogram_bin_edges(data_new[:,4], bins=len(np.unique(D1_points_d[:,9])))
    D1_points_d_2 = np.column_stack((data_new, distances_d, np.array(Points_closest_d).reshape(len(D_tree_d),3),np.digitize(data_new[:,4],bin_edges)))

    
    return D1_points_d_2

#Merge Clusters to get final bin values
def MergeClusters_MinPoints_DeltaM(D1_points_d_3,minpoints,DeltaM):
    start_val = 1
    end_val = max(np.unique(D1_points_d_3.iloc[:,10]))
    while start_val <= end_val:
        a = start_val
        a_0 = start_val-1 
        #Try and Merge based on average cluster column density values
        #Get Column density average for 1st cluster
        ca_1 = np.nanmean(D1_points_d_3[(D1_points_d_3.iloc[:,10] == a)].iloc[:,4])
        #Column denisty average of previous cluster
        ca_0 = np.nanmean(D1_points_d_3[(D1_points_d_3.iloc[:,10] == a_0)].iloc[:,4])
        #Percent Difference
        PD = abs(((ca_1 - ca_0)/(ca_0)))*100
        #Check if they are within DeltaM Value (Maybe a Percent Difference)
        if PD <= DeltaM:
            #Change label to previous label value (a_0)
            D1_points_d_3.iloc[:,10] = np.where((D1_points_d_3.iloc[:,10])!= a, D1_points_d_3.iloc[:,10],a_0)
        if len(np.where((D1_points_d_3.iloc[:,10]) == a_0)[0]) < minpoints:
            #Change Label based on Min # of points to 0
            D1_points_d_3.iloc[:,10] = np.where((D1_points_d_3.iloc[:,10])!= a_0, D1_points_d_3.iloc[:,10],0)
        if a_0 == 1:
              #Change Label based on Min # of points to 0
            D1_points_d_3.iloc[:,10] = np.where((D1_points_d_3.iloc[:,10])!= a_0, D1_points_d_3.iloc[:,10],0)
        start_val = start_val + 1
    #Mask 0 values 
    data_good = np.ma.masked_where(D1_points_d_3.iloc[:,10] == 0, D1_points_d_3.iloc[:,10])
    
    #Reset values to plot the cluster labels
    u, ind = np.unique(data_good, return_inverse = True)
    u = u.argsort().argsort()
    ret = u[ind]
    #print("Estimated number of Magnitude clusters: %d" % n_clusters_)
    
    return data_good,ret

######---------------------------------------------------------------------------------------------------########
#Temporal Cluster Functions
######---------------------------------------------------------------------------------------------------########
#Relabel all the Spatial Clusters - so merging them does not become too complicated
#This is, for now a fix 
def RelabelClusters(GG_label_d_all):
    num_days_analyzed = len(GG_label_d_all)
    label_start = 0
    for i in range(0,num_days_analyzed):
        GG_label_d_all_0 = GG_label_d_all[i]    
        clusters_0 = np.unique(GG_label_d_all_0.iloc[:,5])
        for j in clusters_0:
            GG_label_d_all_0.iloc[:,5] = np.where((GG_label_d_all_0.iloc[:,5])!= j, GG_label_d_all_0.iloc[:,5],label_start)
            GG_label_d_all[i] = GG_label_d_all_0
            label_start = label_start+1
    return GG_label_d_all

#Flatten All day's data
def FlattenDays(GG_label_d_all_11):
    GG_label_d_all_flat = pd.DataFrame([])
    for i in GG_label_d_all:
        GG_label_d_all_flat = GG_label_d_all_flat.append(pd.DataFrame(i))
    return GG_label_d_all_flat

#Find Frequency
def FindFrequency(GG_label_d_all_flat):
    freq_all = GG_label_d_all_flat.groupby([GG_label_d_all_flat.iloc[:,2],GG_label_d_all_flat.iloc[:,3]],group_keys=False).size().reset_index()
    return freq_all

#Remove Infrequent Dates
def RemoveInfrequentDates(freqthreshold, breaks_T,freq_all):
    if freqthreshold == 'y':
        pointsT = freq_all[freq_all.iloc[:,2] >= breaks_T[2]]
    else:
        pointsT = freq_all[freq_all.iloc[:,2] >= breaks_T[1]]
    return pointsT

def MergeOnTime(pointsT, sp_res, eps_scale,bins_temporal):
    #Add histogram delination to Cluster data
    GG_all_bins = np.column_stack((pointsT,np.digitize(pointsT.iloc[:,2],bins_temporal)))
    #Convert to list to do next step
    points3_list = list(zip(GG_all_bins[:,0],GG_all_bins[:,1],GG_all_bins[:,2],GG_all_bins[:,3]))
    #Combine Points together
    Eps1 = sp_res*eps_scale
    ClusT=[]
    while len(points3_list):
            locus = points3_list.pop() 
            cluster = [x for x in points3_list if euclidean(locus[0:2],x[0:2]) <= Eps1]
            ClusT.append(cluster+[locus])
            for x in cluster:
                points3_list.remove(x) 
    return ClusT, GG_all_bins
    
def ClusterMeans_Temporal(ClusT):
    Clus_LatmeanT = []
    Clus_LonmeanT = []
    Clus_BinmeanT = []
    for i in ClusT:
        df = np.array(i)
        Clus_LatmeanT.append(stats.mode(df[:,1])[0])
        Clus_LonmeanT.append(stats.mode(df[:,0])[0])
        Clus_BinmeanT.append(stats.mode(df[:,3])[0])    
    #Reorder the Clus_Binmean to get singly increasing values
    for i,j in zip(np.unique(Clus_BinmeanT),range(1,len(np.unique(Clus_BinmeanT))+1)):
        Clus_BinmeanT = np.where(Clus_BinmeanT != i, np.array(Clus_BinmeanT),j)
    return Clus_LatmeanT,Clus_LonmeanT,Clus_BinmeanT

def Clus_NearestNeighbors_Temporal(GG_all_bins, Clus_LatmeanT,Clus_LonmeanT,Clus_BinmeanT, mindates, minpoints):
    D_tree_d = np.column_stack((GG_all_bins[:,0:2], GG_all_bins[:,3])) #Lat and Lon and Bin Label
    Max_tree_d = np.column_stack((Clus_LonmeanT,Clus_LatmeanT,Clus_BinmeanT)) 
    
    tree_c = BallTree(Max_tree_d,  leaf_size=1)
    #Find closest points
    k_neighbors = 1
    distances_c, indices_c = tree_c.query(D_tree_d, k=k_neighbors)
    
    #4b)Get Data Points from the indices
    Points_closest_c = []
    for cl in indices_c.transpose():
        #cl is the index number
        closest_latlon = np.row_stack(Max_tree_d[o] for o in cl)
        Points_closest_c.append(closest_latlon)
        
    CC_points_c = np.column_stack((GG_all_bins, distances_c,np.array(Points_closest_c).reshape(len(D_tree_d),3)))

    bin_edges_T = np.histogram_bin_edges(CC_points_c[:,2], bins=len(np.unique(CC_points_c[:,7]))) #len(np.unique(CC_points_c[:,7]))
   # _ = plt.hist(CC_points_c[:,2], bins=bin_edges_T)
    CC_digitize = np.digitize(CC_points_c[:,2],bin_edges_T)
    CC_points_d = []
    for n in range(CC_points_c[:,2].size):
        #print(bin_edges_T[CC_digitize[n]-1], "<=", CC_points_c[:,2][n], "<", bin_edges_T[CC_digitize[n]])
        CC_points_d.append(bin_edges_T[CC_digitize[n]-1]) 
  
    CC_points_c_2 = np.column_stack((GG_all_bins, distances_c,np.array(Points_closest_c).reshape(len(D_tree_d),3),CC_points_d))

    start_val = 1
    end_val = max(np.unique(CC_points_c_2 [:,8]))
    while start_val <= end_val+1:
        a = start_val 
        a_0 = start_val-1  
        if len(CC_points_c_2 [np.where((CC_points_c_2 [:,8]) == a_0)]) == 0:
            start_val = start_val + 1
            continue
        if np.max(CC_points_c_2 [np.where((CC_points_c_2 [:,8]) == a_0)][:,2]) <= mindates:
            #If frequency of dates is less than minpoints
            #Change Label based on Min # of dates to 0
            CC_points_c_2 [:,8] = np.where((CC_points_c_2 [:,8])!= a_0, CC_points_c_2 [:,8],0)
        if len(CC_points_c_2 [np.where((CC_points_c_2 [:,8]) == a_0)]) < minpoints:
            #Change Label based on Min # of points to 0
            CC_points_c_2 [:,8] = np.where((CC_points_c_2 [:,8])!= a_0, CC_points_c_2 [:,8],0)
        start_val = start_val + 1
    
    #Mask 0 values 
    data_good_T = np.ma.masked_where(CC_points_c_2 [:,8] == 0, CC_points_c_2 [:,8])
    return CC_points_c_2 , data_good_T
            

def GetSpatialClusters_Temporal(data_temporal,sp_res):
    clustering = DBSCAN(eps=2*sp_res, min_samples=1).fit(np.array(data_temporal)[:,0:2])
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print("Estimated number of clusters: %d" % n_clusters_)
    
    #Add cluster labels 
    pointsT_labels = np.column_stack((data_temporal,labels))
    unique_labelsT = np.unique(pointsT_labels[:,9])
    return pointsT_labels, unique_labelsT

def RemoveSourcePointClusters_Temporal(pointsT_labels,source,unique_labelsT):   
    Source_Clus = np.empty(shape=(0,11))
    NoSource_Clus = np.empty(shape=(0,11))
    for k in unique_labelsT:
        indi_clus = pointsT_labels[np.where(pointsT_labels[:,9] == k)]
        Latmin_clus = min(indi_clus[:,1])
        Latmax_clus = max(indi_clus[:,1])
        Lonmin_clus = min(indi_clus[:,0])
        Lonmax_clus = max(indi_clus[:,0])
        
        if not 'SourcePoint_LonLat' in locals():
            NoSource_Clus = np.concatenate([NoSource_Clus, indi_clus], axis=0)
            #Check what user wants - Source/No Source
            if source == 'y':
                data_new_temporal = pointsT_labels
            if source == 'n':
                data_new_temporal = pointsT_labels
        
        if 'SourcePoint_LonLat' in locals():     
            print('n')
            #See if source points are in cluster bounds
            #Check All Source Points in the Region
            SourcePoint_InClus = SourcePoint_InRegion[(SourcePoint_InRegion[:,0]>= Lonmin_clus) & (SourcePoint_InRegion[:,0] <= Lonmax_clus) & (SourcePoint_InRegion[:,1] >= Latmin_clus) & (SourcePoint_InRegion[:,1] <= Latmax_clus)]
            
            if len(SourcePoint_InClus) != 0:
                Source_Clus = np.concatenate([Source_Clus, indi_clus], axis=0)
            else:
                NoSource_Clus = np.concatenate([NoSource_Clus, indi_clus], axis=0)
        
            #Check what user wants - Source/No Source
            if source == 'y':
                data_new_temporal = pointsT_labels #np.concatenate([Source_Clus, NoSource_Clus],axis=0)
            else:
                data_new_temporal = NoSource_Clus    
        return data_new_temporal

#https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
