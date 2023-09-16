# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:50:30 2022

@author: tabyl
"""

######---------------------------------------------------------------------------------------------------########
######---------------------------------------------------------------------------------------------------########
#Get SOURCE POINT INFORMATION!
    #Including: NEI, CEMS, Population, Lightning, Fires, Oil/Gas, ?
    #Use a function Call?

######---------------------------------------------------------------------------------------------------########
# CLustering of Atmospheric Satelliate Products
# Author: Tabitha Lee (tclee3@uh.edu)
# CLASP: clustering of atmospheric satelliate products

#Libraries Used
import numpy as np
import pandas as pd
import glob
import os 
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.switch_backend('Agg')
#Need to set Environment for the use of basemap
os.environ['PROJ_LIB'] = r'/share/apps/anaconda3/2020.07/lib/python3.8/'  #os.environ['PROJ_LIB'] = r'C:/Users/tabyl/Anaconda3/Library/share' #OR Tabitha L #
from mpl_toolkits.basemap import Basemap
from scipy.spatial.distance import euclidean
from scipy import stats
from sklearn.neighbors import BallTree
import pwlf
from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry import Point
proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')
from sklearn.cluster import DBSCAN

def CLASP(LatMax, LatMin, LonMax, LonMin, O, sp_res, maginflu, source, eps_scale, minpoints, DeltaM, temporalclusters, freqthreshold, mindates):
    """Implementation of CLUstering of Atmospheric Satelliate Products
        See: (A GitHub site)
        
        -Uses DBSCAN to create minimum time step spatial clusters (DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
        -Uses Ball and Tree nearest neighbor clustering to create minimum time step Magnitude Clusters (Ball and Tree: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html )
        -Uses Ball and Tree nearest neighbor clustering to create Temporal Clusters (Ball and Tree: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html )
        
        Parameters
        ----------  
            LatMax - Mqximum latitude value in decimal degrees (S=negative, N=positive)
            LatMin - Minimum latitude value in decimal degree (S=negative, N=positive)
            LonMax - Maximum longitude value in decimal degree (W=negative, E=positive)
            LonMin - Minimum longitude value in decimal degree (W=negative, E=positive)
            
            O - Oversampled Product 
                **Must be in this order: ['Date','Longitude','Latitude','Magnitude Value'] 
                {array([[date1,longitude1,latitude1,magnitude1],...,[daten,longituden,latituden,,agnituden]]), type:numpy array}
                
            sp_res - Spatial Resolution of the Oversampled Product
                **CLASP assumes that the data is oversampled to a regular grd (see:). Thus, this should be one number (equal spatial resoluton on the latitudanal or longitudanl axis)
                e.g. 0.01 {type=int}
            
            maginflu - Do you want to look at the relative high mangitude points only?   
                {'y' or 'n', type:str}
                
            source - Do you want to show clusters that are associated with source points?
                {'y' or 'n', type:str}
                
            minpoints - The minimum number of points required to make a cluster
                {default=1, type:int}
            
            eps_scale - Scale value for maximum Euclidean distance two points can have to be included in the same cluster
                Recommend: oversampled resolution * number, 
                e.g. eps_scale = 0.1*5 = .5 {default=sp_res (sp_res*1), type:int}
                
            DeltaM - Maximum Percentage magnitude difference two clusters can have
                e.g. 5% {default=5, type:int}
                
            temporalclusters - Do you want to find Temporal Clusters?
                {'y' or 'n', type:str}
            
            freqthreshold - Do you want to look at the relative frequent days only?   
                {'y' or 'n', type:str}
            
            mindates - The minimum number of days required to make a cluster
                {default=1, type:int}
            
            
        Returns
        -------
            CLASP_output - Oversampled product described spatialy, by magnitude, and temporally
            {array([[date1,longitude1,latitude1,magnitude1,spatialclusterlabel1, magnitdueclusterlabel1, frequency, temporalclusterlabel, spatialtemporalclusterlabel],...,
                [daten,longituden,latituden,magnituden,spatialclusterlabeln, magnitdueclusterlabeln, frequency, temporalclusterlabel, spatialtemporalclusterlabel]]), type:numpy array}
               
                
    """

    #0) Initilize list to hold each day's cluster information
    GG_label_d_all = []
   
    #2) Subset Data Based on Lat/Lon Bounds
    C_sample = O[(O[:,1] >= LonMin) & (O[:,1] <= LonMax) & (O[:,2] >= LatMin) & (O[:,2] <= LatMax)]
    
    #Source Point Information - User has to include!!
    #2a) Check and Subset source point data
    if 'SourcePoint_LonLat' in locals():    
            SourcePoint_InRegion = SourcePoint_LonLat[(SourcePoint_LonLat[:,0]>= LonMin) & (SourcePoint_LonLat[:,0] <= LonMax) & (SourcePoint_LonLat[:,1] >= LatMin) & (SourcePoint_LonLat[:,1] <= LatMax)]
    if not 'SourcePoint_LonLat' in locals():
        print('There are no source points included')
        WithNoSourcePoints = input('Do you want to continue with no source point information? (y or n)')
        if WithNoSourcePoints == 'y':
            print('No source points will be included')
            source = 'n'
        else:
            print('Please add source point information')
            return None
        
    #2b) Get the individual dates
    Sample_ByDates = [C_sample[C_sample[:,0]==k] for k in np.unique(C_sample[:,0])]
    
    #This section finds the Spatial and Column Density Clusters for EACH day
    #3) Start loop to go through each day of oversampled data
    for i in range(0,len(Sample_ByDates)):
        #Get a single day
        D1 = Sample_ByDates[i]
        
        #4) Check value of good pixels
        if CheckGoodPixels(D1) == True:
            print('Moving to next day because this date does not have enough quality pixels')
            continue
        
        #6) Manually Add Index Values to Track the Local Maximums
        index_val = np.array(range(0,len(D1)))
        index_val = index_val.reshape(len(D1),1)
        D1 = np.hstack((index_val,D1))
        #Get Date
        date1 = pd.to_datetime(D1[0,1],format='%Y%m%d')
        
        # #7) Check Anthroflu and see what data we need to use
        try:
            points_plots = D1[:,4][~np.isnan(D1[:,4])] #Gets rid of Nan Values
        except ValueError:
            print('Not enough data or it was all NAN values')
            continue
        try:
            # hist, bin_edges = np.histogram(points_plots)
            # numbins = int(len(bin_edges))
            numbins = int(n_bin(points_plots)) #If this return errors then use above commented out lines
        except ValueError:
            print('0 bins')
            continue
        if numbins < 2:
            print('1 bin')
            continue
        
        #8) Find Threshold points using Relative Cumulative Frequency
        breaks = RCF_ThresholdPoints(points_plots,numbins)
        
        #9) Remove Background Values at Set Threshold
        points2 = RemoveBackground(maginflu,breaks,D1)
        
        #11) Use DBSCAN to get spatial cluster labels
        points2_labels, unique_labels, n_clusters_ = GetSpatialClusters(points2,sp_res)
        if n_clusters_ < 2: 
            print("Only 1 spatial cluster found")
            continue
        
        #12) Remove clusters that have a source point inside the cluster
        ##########################This is old 
        data_new = RemoveSourcePointClusters(points2_labels,source,unique_labels)
        
        #10) Add Histogram Bin number to the points to track the column density variations
        bins = BinValues(data_new[:,4])
        
        #13) Merge Column density clusters together based on their distance
        #This ensures there are not too many column density clusters 
        Clus = MergeOnDistance(data_new,bins,sp_res,eps_scale)
        
        #14) Get the Column Density Values Cluster centers
        Clus_Latmean, Clus_Lonmean, Clus_Binmean = ClusterMeans(Clus)
        
        #15) Find Nearest Neighbors of the Column Density Cluster Center Values to form the Column Density Clusters
        D1_points_d = Clus_NearestNeighbors(data_new,bins,Clus_Lonmean,Clus_Latmean,Clus_Binmean)
        
        #16) Merge Column Density Clusters using the desired minimum number of points in a cluster and difference in column density values
        #Changing to dataframe for easier merging of the clusters - would like to fix this soon....
        D1_points_d_3 = pd.DataFrame(D1_points_d)
        data_good, ret = MergeClusters_MinPoints_DeltaM(D1_points_d_3,minpoints,DeltaM)
       
        GG_label_d_all.append(D1_points_d_3)
        #End the for/while loop which creates (1) Spatial and (2) Column Density clusters for EACH day
         
    
    #19) Flatten all day's data
    GG_label_d_all_flat = FlattenDays(GG_label_d_all)
    GG_label_all_notemporal = pd.concat([GG_label_d_all_flat.iloc[:,1:6],GG_label_d_all_flat.iloc[:,9]], axis=1)
    
    #This section finds the Temporal Clusters for the time period of interest
    if temporalclusters == 'y':
        print("Temporal Cluster Section")
    else:
        return G_label_all_notemporal
    
    #20) Get Temporal Clusters
    #20a) Find Threshold points using Relative Cumulative Frequency
    freq_all = GG_label_d_all_flat.groupby([GG_label_d_all_flat.iloc[:,2],GG_label_d_all_flat.iloc[:,3]],group_keys=False).size().reset_index()
    numbins_T = int(n_bin(freq_all.iloc[:,2]))
    breaks_T = RCF_ThresholdPoints(freq_all.iloc[:,2],numbins_T)
    
    #20b) Remove Infrequent Dates and Set mindates if not defined
    pointsT = RemoveInfrequentDates(freqthreshold,breaks_T,freq_all)
        
    #20c) Get Bins for Temporal Values
    numbins_temporal = int(np.ceil(np.log2(len(np.unique(GG_label_d_all_flat.iloc[:,1])))) + 1) 
    bins_temporal = BinValues(pointsT.iloc[:,2],numbins_temporal)
    
    #13) Merge Column density clusters together based on their distance
    #This ensures there are not too many temporal clusters 
    ClusT, GG_all_bins = MergeOnTime(pointsT,sp_res, eps_scale,bins_temporal)
    
    #14) Get the Column Density Values Cluster centers
    Clus_LatmeanT, Clus_LonmeanT, Clus_BinmeanT = ClusterMeans_Temporal(ClusT)
    
    #15) Find Nearest Neighbors of the Column Density Cluster Center Values to form the Column Density Clusters
    CC_points_c_2, data_good_T = Clus_NearestNeighbors_Temporal(GG_all_bins, Clus_LatmeanT,Clus_LonmeanT,Clus_BinmeanT, mindates, minpoints)
    #Replace 0 with Nan
    data_good_data = data_good_T.data.reshape(len(CC_points_c_2),-1)
    data_good_data = np.where(data_good_data==0, np.nan, data_good_data)
    data_temporal = np.hstack((CC_points_c_2,data_good_data))
    
    #Remove Source Point Infulences - Again - Just to make sure
    ##########################This is old 
    #20bb) Get Spatial Clusters with Temproal Points to find the spatial location of the temporal clusters
    pointsT_labels, unique_labelsT = GetSpatialClusters_Temporal(data_temporal,sp_res)
    
    #21) Remove Temporal Clusters near the source points as a final check! Whoo!
    data_new_temporal = RemoveSourcePointClusters_Temporal(pointsT_labels,source,unique_labelsT)
    data_new_temporal_df = pd.DataFrame(data_new_temporal)  
        
    #Reshape data to have final output with data and cluster labels
    #Date, Lon, Lat, Magnitude Value, Spatial Cluster Label for individual day, Magnitude Cluster label for individual days, Temporal Cluster Label
    #Merge together everything and MASK unneeded values
    CLASP_output_SpatialMagnitude = pd.concat([GG_label_d_all_flat.iloc[:,1:6], GG_label_d_all_flat.iloc[:,9]], axis=1)
    CLASP_output_SpatialMagnitude.columns = ['Date','Lon','Lat','Mag','SP_C','MG_C']
    CLASP_output_Temporal = pd.concat([data_new_temporal_df.iloc[:,0:3], data_new_temporal_df.iloc[:,8], data_new_temporal_df.iloc[:,10]], axis=1)
    CLASP_output_Temporal.columns = ['Lon','Lat','Freq','T_C','SP_T_C']
    O_df = pd.DataFrame(O)
    O_df.columns = ['Date','Lon','Lat','Mag']
    
    CLASP_output_SPMG = pd.merge(O_df,CLASP_output_SpatialMagnitude, how="outer", on=["Date", "Lon","Lat", "Mag"])
    CLASP_output_SPMGT = pd.merge(CLASP_output_SPMG, CLASP_output_Temporal, how="outer", on=["Lon","Lat"])
    CLASP_output_SPMGT.iloc[:,7].replace(0, np.nan, inplace=True)
    
    if temporalclusters == 'y':
        CLASP_output = np.array(CLASP_output_SPMGT)
    else:
        CLASP_output = np.array(CLASP_output_SPMG)
    
    return CLASP_output


#Plot CLASP Outputs
if len(CLASP_output.columns) == 6 or len(CLASP_output.columns) == 9:
    Outputs_SpMg = [CLASP_output[CLASP_output.iloc[:,0]==k] for k in np.unique(CLASP_output.iloc[:,0])]
    for C in range(0,len(Outputs_SpMg)):
        CC= Outputs_SpMg[C]
        #Spatial Clusters
        cmap =  plt.get_cmap('winter')
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
    
        bounds = np.linspace(0, len(np.unique(CC.iloc[:,5])),len(np.unique(CC.iloc[:,5]))+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ticks = np.linspace(1, len(np.unique(CC.iloc[:,5])),len(np.unique(CC.iloc[:,5])))
    
        fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
        m = Basemap(projection='cyl',resolution='l',
                llcrnrlat=LatMin,urcrnrlat=LatMax,
                llcrnrlon=LonMin,urcrnrlon=LonMax)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries()
        m.drawparallels(np.arange(round(LatMin), round(LatMax), 8), labels=[1, 0, 0, 0], linewidth=0.0,fontsize=12)
        m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0,fontsize=12)
        #Plot Data
        s = plt.scatter(CC.iloc[:,1], CC.iloc[:,2], c=CC.iloc[:,5], cmap=cmap, s=50) #Or 2,1 instead of 10,9 #c=[col]
        #Colorbar
        #ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8]) #Colorbar vertical
        ax2 = fig.add_axes([0, 0, 0.8, 0.03]) #Colorbar horizontal
        #[left, bottom, width, height]
        mpl.colorbar.ColorbarBase(ax2,cmap=cmap, norm=norm, orientation="horizontal", spacing='proportional',boundaries = bounds, ticks = ticks, format='%1i') #spacing='proportional', 
        #Plot Facility Points
        #m.scatter(SourcePoint_LonLat['Facility Longitude'],SourcePoint_LonLat['Facility Latitude'],c='black', marker="*",s=50)
        #plt.title('TROPOMI Tropospheric NO\u2082 ' +(date1).strftime('%m-%d-%Y')+ ' No. Clusters:'+str(len(np.unique(D1_points_d_3.iloc[:,5]))),size=12) 
        #plt.savefig(r"~/"+(date1).strftime('%Y%m%d') + ".png")
        plt.show()
        #plt.close()
        
        #Magnitude Clusters
        cmap = plt.get_cmap('jet') 
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
    
        bounds = np.linspace(0, len(np.unique(CC.iloc[:,5])),len(np.unique(CC.iloc[:,5]))+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ticks = np.linspace(1, len(np.unique(CC.iloc[:,5])),len(np.unique(CC.iloc[:,5])))
    
        fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
        m = Basemap(projection='cyl',resolution='l',
            llcrnrlat=LatMin,urcrnrlat=LatMax,
            llcrnrlon=LonMin,urcrnrlon=LonMax)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries()
        m.drawparallels(np.arange(round(LatMin), round(LatMax), 8), labels=[1, 0, 0, 0], linewidth=0.0,fontsize=12)
        m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0,fontsize=12)
        #Plot Data
        s = plt.scatter(CC.iloc[:,1], CC.iloc[:,2], c=CC.iloc[:,6], cmap=cmap, ) #Or s=50 2,1 instead of 10,9 #c=[col]
        #Colorbar
        ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional',boundaries = bounds, ticks = ticks, format='%1i') #spacing='proportional', 
        #Plot Facility Points
        #m.scatter(SourcePoint_LonLat['Facility Longitude'],SourcePoint_LonLat['Facility Latitude'],c='black', marker="*",s=50)
        #plt.title('TROPOMI Tropospheric NO\u2082 ' +(date1).strftime('%m-%d-%Y')+ ' No. Clusters:'+str(len(np.unique(data_good))-1),size=12) 
        #plt.savefig(r"/~/"+(date1).strftime('%Y%m%d') + ".png")
        plt.show()
        # #plt.close()

#Plot Temporal Clusters
if len(CLASP_output.columns) == 9:
    # create the new map
    cmap1 = plt.get_cmap('Spectral')
    cmaplist1 = [cmap1(i) for i in range(cmap1.N)]
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist1, cmap1.N)
    
    #Bounds are based on the number of unique dates
    #len counts nan - messing up the number of bounds
    bounds = np.linspace(0, np.count_nonzero(~np.isnan(np.unique(CLASP_output.iloc[:,7]))), np.count_nonzero(~np.isnan(np.unique(CLASP_output.iloc[:,7])))+1) 
    vals = bounds[:-1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    ticks = np.linspace(1, np.count_nonzero(~np.isnan(np.unique(CLASP_output.iloc[:,7]))),np.count_nonzero(~np.isnan(np.unique(CLASP_output.iloc[:,7]))))
    
    fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
    m = Basemap(projection='cyl',resolution='l',
            llcrnrlat=LatMin,urcrnrlat=LatMax,
            llcrnrlon=LonMin,urcrnrlon=LonMax)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawparallels(np.arange(round(LatMin), round(LatMax), 6), labels=[1, 0, 0, 0], linewidth=0.0, fontsize=12)
    m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0, fontsize=12)
    #Temporal Clusters
    m.scatter(CLASP_output.iloc[:,1], CLASP_output.iloc[:,2], c=CLASP_output.iloc[:,7],cmap=cmap1) 
    #Source Points
    #m.scatter(SourcePoint_LonLat[:,0],SourcePoint_LonLat[:,1],c='black', marker="*",s=50)
    #Colorbar
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, norm=norm, boundaries = bounds, values=vals, format='%1i') #spacing='proportional', 
    cbar.set_ticks(vals + .2)
    cbar.set_ticklabels(ticks)
    #plt.title('TROPOMI Tropospheric NO\u2072 Temporal' +(date1).strftime('%m-%d-%Y')+ ' No. Clusters:'+str(len(np.unique(pointsT_labels[:,7]))),size=12) 
    #plt.savefig(r"~/"+(date1).strftime('%Y%m%d') + ".png")
    plt.show()
    
    #Spatial Locations of the temporal clusters
    # create the new map
    cmap1 = plt.get_cmap('winter')
    cmaplist1 = [cmap1(i) for i in range(cmap1.N)]
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist1, cmap1.N)
    
    #Bounds are based on the number of unique dates
    bounds = np.linspace(np.nanmin(np.unique(CLASP_output.iloc[:,8])), np.nanmax(np.unique(CLASP_output.iloc[:,8])),len(np.unique(CLASP_output.iloc[:,8]))) 
    vals = bounds[:-1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    ticks = np.linspace(1, np.nanmax(np.unique(CLASP_output.iloc[:,8]))+1,len(np.unique(CLASP_output.iloc[:,8]))-1) 
    
    fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
    m = Basemap(projection='cyl',resolution='l',
            llcrnrlat=LatMin,urcrnrlat=LatMax,
            llcrnrlon=LonMin,urcrnrlon=LonMax)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawparallels(np.arange(round(LatMin), round(LatMax), 8), labels=[1, 0, 0, 0], linewidth=0.0,fontsize=12)
    m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0,fontsize=12)
    #Plot Data
    s = plt.scatter(CLASP_output.iloc[:,1], CLASP_output.iloc[:,2], c=CLASP_output.iloc[:,8], cmap=cmap1 ) 
    #Colorbar
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, norm=norm, spacing='proportional',boundaries = bounds, values=vals, format='%1i') #spacing='proportional', 
    cbar.set_ticks(vals + .2)
    cbar.set_ticklabels(ticks)
    #plt.title('TROPOMI Tropospheric NO\u2082 ' +(date1).strftime('%m-%d-%Y')+ ' No. Clusters:'+str(len(np.unique(data_good))-1),size=12) 
    #plt.savefig(r"~/"+(date1).strftime('%Y%m%d') + ".png")
    plt.show()
    
    
#Plot CLASP Outputs
if len((CLASP_output[0,:])) == 6 or len((CLASP_output[0,:])) == 9:
    Outputs_SpMg = [CLASP_output[CLASP_output[:,0]==k] for k in np.unique(CLASP_output[:,0])]
    for C in range(0,len(Outputs_SpMg)):
        CC= Outputs_SpMg[C]
        #Spatial Clusters
        cmap =  plt.get_cmap('winter')
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
    
        bounds = np.linspace(0, len(np.unique(CC[:,4])),len(np.unique(CC[:,4]))+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ticks = np.linspace(1, len(np.unique(CC[:,4])),len(np.unique(CC[:,4])))
    
        fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
        m = Basemap(projection='cyl',resolution='l',
                llcrnrlat=LatMin,urcrnrlat=LatMax,
                llcrnrlon=LonMin,urcrnrlon=LonMax)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries()
        m.drawparallels(np.arange(round(LatMin), round(LatMax), 8), labels=[1, 0, 0, 0], linewidth=0.0,fontsize=12)
        m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0,fontsize=12)
        #Plot Data
        s = plt.scatter(CC[:,1], CC[:,2], c=CC[:,4], cmap=cmap, s=50) #Or 2,1 instead of 10,9 #c=[col]
        #Colorbar
        #ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8]) #Colorbar vertical
        ax2 = fig.add_axes([0, 0, 0.8, 0.03]) #Colorbar horizontal
        #[left, bottom, width, height]
        mpl.colorbar.ColorbarBase(ax2,cmap=cmap, norm=norm, orientation="horizontal", spacing='proportional',boundaries = bounds, ticks = ticks, format='%1i') #spacing='proportional', 
        #Plot Source Points
        #m.scatter(SourcePoint_LonLat[:,0],SourcePoint_LonLat[:,1],c='black', marker="*",s=50)
        plt.show()
        
        #Magnitude Clusters
        cmap = plt.get_cmap('jet') 
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
    
        bounds = np.linspace(0, len(np.unique(CC[:,5])),len(np.unique(CC[:,5]))+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ticks = np.linspace(1, len(np.unique(CC[:,5])),len(np.unique(CC[:,5])))
    
        fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
        m = Basemap(projection='cyl',resolution='l',
            llcrnrlat=LatMin,urcrnrlat=LatMax,
            llcrnrlon=LonMin,urcrnrlon=LonMax)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries()
        m.drawparallels(np.arange(round(LatMin), round(LatMax), 8), labels=[1, 0, 0, 0], linewidth=0.0,fontsize=12)
        m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0,fontsize=12)
        #Plot Data
        s = plt.scatter(CC[:,1], CC[:,2], c=CC[:,5], cmap=cmap, ) #Or s=50 2,1 instead of 10,9 #c=[col]
        #Colorbar
        ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional',boundaries = bounds, ticks = ticks, format='%1i') #spacing='proportional', 
        #Plot Source Points
        #m.scatter(SourcePoint_LonLat[:,0],SourcePoint_LonLat[:,1],c='black', marker="*",s=50)
        plt.show()

#Plot Temporal Clusters
if len((CLASP_output[0,:])) == 9:
    #Temporal Cluster Plot
    bounds = np.linspace(0, np.count_nonzero(~np.isnan(np.unique(CLASP_output[:,7]))), np.count_nonzero(~np.isnan(np.unique(CLASP_output[:,7])))+1) 
    vals = bounds[:-1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    ticks = np.linspace(1, np.count_nonzero(~np.isnan(np.unique(CLASP_output[:,7]))),np.count_nonzero(~np.isnan(np.unique(CLASP_output[:,7]))))
    
    fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
    m = Basemap(projection='cyl',resolution='l',
            llcrnrlat=LatMin,urcrnrlat=LatMax,
            llcrnrlon=LonMin,urcrnrlon=LonMax)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawparallels(np.arange(round(LatMin), round(LatMax), 6), labels=[1, 0, 0, 0], linewidth=0.0, fontsize=12)
    m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0, fontsize=12)
    #Temporal Clusters
    m.scatter(CLASP_output[:,1], CLASP_output[:,2], c=CLASP_output[:,7],cmap=cmap1) 
    #Source Points
    #m.scatter(SourcePoint_LonLat[:,0],SourcePoint_LonLat[:,1],c='black', marker="*",s=50)
    #Colorbar
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, norm=norm, boundaries = bounds, values=vals, format='%1i') #spacing='proportional', 
    cbar.set_ticks(vals + .2)
    cbar.set_ticklabels(ticks)
    plt.show()
    
    #Spatial Locations of the temporal clusters
    # create the new map
    cmap1 = plt.get_cmap('winter')
    cmaplist1 = [cmap1(i) for i in range(cmap1.N)]
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist1, cmap1.N)
    
    #Bounds are based on the number of unique dates
    bounds = np.linspace(np.nanmin(np.unique(CLASP_output[:,8])), np.nanmax(np.unique(CLASP_output[:,8])),len(np.unique(CLASP_output[:,8]))) 
    vals = bounds[:-1]
    norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)
    ticks = np.linspace(1, np.nanmax(np.unique(CLASP_output[:,8]))+1,len(np.unique(CLASP_output[:,8]))-1) 
    
    fig, ax  = plt.subplots(figsize=(5,5), dpi=300)
    m = Basemap(projection='cyl',resolution='l',
            llcrnrlat=LatMin,urcrnrlat=LatMax,
            llcrnrlon=LonMin,urcrnrlon=LonMax)
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries()
    m.drawparallels(np.arange(round(LatMin), round(LatMax), 8), labels=[1, 0, 0, 0], linewidth=0.0,fontsize=12)
    m.drawmeridians(np.arange(round(LonMin), round(LonMax),12), labels=[0, 0, 0, 1], linewidth=0.0,fontsize=12)
    #Plot Data
    s = plt.scatter(CLASP_output[:,1], CLASP_output[:,2], c=CLASP_output[:,8], cmap=cmap1 ) 
    #Colorbar
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmap1, norm=norm, spacing='proportional',boundaries = bounds, values=vals, format='%1i') #spacing='proportional', 
    cbar.set_ticks(vals + .2)
    cbar.set_ticklabels(ticks)
    plt.show()
