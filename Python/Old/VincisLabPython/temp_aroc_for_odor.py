
# -*- coding: utf-8 -*-

def resp_odor (filePath,mouse,Date, Odor={}):
    """
    Created on Sat Oct  8 10:28:45 2022
   
    @author: VincisLabMain
    """
   
    # import the libraries needed
    #from pathlib import Path
    import numpy as np
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    from VincisLab_Python.EnsembleDecoding.decoding_tools import truncate, smooth_all_spike_trains
    from VincisLab_Python.Slid_prb import Sliding_probability
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
       
    #filePath = Path('D:\DataTemp') # set the base folder
    #mouse = 'CB319' # ID of mouse to analyze
    #Date = '092822' # date
   
    # load the summary dataframe
    fileName_2 = 'SU_Analysis/Sum_Long.csv'
    Sum_Long = pd.read_csv(filePath / mouse / Date / fileName_2)
    Sum_Long['resp'] = [1]*len(Sum_Long)
    Sum_Long['resp'] = Sum_Long['resp'].astype('object')
    Sum_Long['sign'] = [1]*len(Sum_Long)
    Sum_Long['sign'] = Sum_Long['sign'].astype('object')
    Sum_Long['lat'] = [1]*len(Sum_Long)
    Sum_Long['lat'] = Sum_Long['sign'].astype('object')
   
    # load the dataframe -> the zero crossing one
    fileName = 'SU_Analysis/allN_ZeroCross.pickle'
    with open(filePath / mouse / Date / fileName,'rb') as file:
        allN = pickle.load(file)
       
       
    list_stim = allN['Taste'].unique()
    list_neuron = allN['Neuron'].unique()
    lf_sp = []
    for neu in tqdm(list_neuron): # ---> for all the neurons
        resp = []                 # ---> empty list for responsivness
        sign = []                 # ---> empty list for sign (1=exc, -1=inh)
        lat = []                  # ---> empty list for latencies
        lif_spars_1 = []          # ---> empty list for life sparseness part I (see below)
        lif_spars_2 = []          # ---> empty list for life sparseness part II (see below)
       
        for od in list_stim:      # ---> for all the odors
           
            allN_filt = allN[allN['Taste']==od]                 # ---> filt the spikes dataframe based on the odor and...
            allN_filt_2 = allN_filt[allN_filt['Neuron']==neu]   # neuron
            allN_filt_2_smooth = smooth_all_spike_trains(allN_filt_2,window_len=300)
           
            evok = truncate(allN_filt_2, 'post-taste')          # ---> consider 2 s post odor for evoke spiking (see truncate)
            copy_data = evok.copy()
            copy_data.drop(copy_data.iloc[:, :7], inplace=True, axis=1)
            copy_data.mean(axis=1)*1000
            vect_evok = np.array(copy_data.mean(axis=1)*1000)
       
            basel = truncate(allN_filt_2, 'pre-taste')          # ---> consider 2 s pre odor for baseline spiking (see truncate)
            copy_data_basel = basel.copy()
            copy_data_basel.drop(copy_data_basel.iloc[:, :7], inplace=True, axis=1)
            copy_data_basel.mean(axis=1)*1000
            vect_basel = np.array(copy_data_basel.mean(axis=1)*1000)
       
            maxval = max(vect_basel)
            if maxval !=0:
                increments = 100
                Baseline_Prob = Sliding_probability(vect_basel,maxval,increments)
                Baseline_Prob.reverse()
                Baseline_Prob.insert(0,0.0)
                Baseline_Prob.insert(len(Baseline_Prob)+1,1.0)
           
                Evoked_Prob = Sliding_probability(vect_evok,maxval,increments)
                Evoked_Prob.reverse()
                Evoked_Prob.insert(0,0.0)
                Evoked_Prob.insert(len(Baseline_Prob)+1,1.0)
           
                auroc = np.trapz(Evoked_Prob,Baseline_Prob,axis=-1)
            else:
                auroc = 0.5
               
            # get a distribution of auroc values by shuffling baseline and evoked trials 1000 times
            all_vector = np.concatenate((vect_basel,vect_evok))
            shuffl_auroc = []
           
            for _ in range(1000):
                temp_1,temp_2 = train_test_split(all_vector, test_size=len(vect_basel)) # randomly assign basel and evoke firing rate to two vectors
               
                maxval_sh = max(temp_1)
               
                if maxval_sh !=0:
                    increments = 100
                    Baseline_Prob_shuf = Sliding_probability(temp_1,maxval_sh,increments)
                    Baseline_Prob_shuf.reverse()
                    Baseline_Prob_shuf.insert(0,0.0)
                    Baseline_Prob_shuf.insert(len(temp_1)+1,1.0)
                   
                    Evoked_Prob_shuf = Sliding_probability(temp_2,maxval_sh,increments)
                    Evoked_Prob_shuf.reverse()
                    Evoked_Prob_shuf.insert(0,0.0)
                    Evoked_Prob_shuf.insert(len(Evoked_Prob_shuf)+1,1.0)
                   
                    shuffl_auroc.append(np.trapz(Evoked_Prob_shuf,Baseline_Prob_shuf,axis=-1))
                else:
                    shuffl_auroc.append(0.5)
           
            auroc_shuffl = np.array(shuffl_auroc)
       
            fig, ax = plt.subplots()
            plt.hist(auroc_shuffl,bins=10),plt.vlines(x=auroc,ymin=0,ymax=10,color='r')
            plt.title(Odor[od])
       
            if (auroc > np.percentile(auroc_shuffl,99)) | (auroc < np.percentile(auroc_shuffl,1)):
                resp.append(od)
                if auroc > np.percentile(auroc_shuffl,99):
                    sign.append(1)
                    # smooth the
                    temp_for_lat = np.array(allN_filt_2_smooth.iloc[:,2007:].mean())
                    temp_for_lat_basel = allN_filt_2_smooth.iloc[:,7:2007].mean()
                    if temp_for_lat.max() > temp_for_lat_basel.mean() + temp_for_lat_basel.std():
                        lat.append(np.where(temp_for_lat>temp_for_lat_basel.mean() + temp_for_lat_basel.std())[0][0])
                    else:
                        del sign[-1]
                        del resp[-1]
                elif auroc < np.percentile(auroc_shuffl,1):
                    sign.append(-1)
                    temp_for_lat = np.array(allN_filt_2_smooth.iloc[:,2007:].mean())
                    temp_for_lat_basel = allN_filt_2_smooth.iloc[:,7:2007].mean()
                    if vect_basel.mean()-(2*vect_basel.mean())>0:
                        lat.append(np.where(temp_for_lat>temp_for_lat_basel.mean() - temp_for_lat_basel.std())[0][0])
                    else:
                         lat.append(np.where(temp_for_lat==temp_for_lat.min())[0][0])
               
            # for lifetime sparsness
            M_basel = vect_basel.mean()
            M_evok = vect_evok.mean()
            if M_evok > M_basel:
                if M_basel == 0:
                    lif_spars_1.append((M_evok)/len(list_stim))
                    lif_spars_2.append((M_evok)**2/len(list_stim))
                else:
                    lif_spars_1.append((M_evok/M_basel)/len(list_stim))
                    lif_spars_2.append((M_evok/M_basel)**2/len(list_stim))
            else:
                lif_spars_1.append(0)
                lif_spars_2.append(0)
               
        lif_spars_1 = np.array(lif_spars_1)
        lif_spars_2 = np.array(lif_spars_2)
       
        lf_sp.append(1-((lif_spars_1.sum()**2)/lif_spars_2.sum())/1-(1/len(list_stim)))
        Sum_Long.at[neu,'resp'] = resp
        Sum_Long.at[neu,'sign'] = sign
        Sum_Long.at[neu,'lat'] = lat
           
    Sum_Long['Life_spars'] = lf_sp
    Sum_Long.to_csv(filePath / mouse / Date / fileName_2)
   
    return Sum_Long

