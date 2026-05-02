import numpy as np
import pandas as pd

def unique_stable(arr):
    # Get unique values and their corresponding indices in the original array
    # 'return_index=True' is the key here
    unique_values, indices = np.unique(arr, return_index=True)

    # The 'indices' array is sorted by default (1, 2, 0, 5, 7) if values were sorted (1, 2, 3, 4, 5)
    # to maintain original order, sort the indices
    sorted_indices = np.sort(indices)

    # Use the sorted indices to select elements from the original array
    # This reconstructs the array with only unique elements in their first-occurrence order
    unique_ordered_data = arr[sorted_indices]
    
    return unique_ordered_data

def interpolate_to_new_time(df, new_time, upsampling_interpolation_method='linear'):
    old_time = df.index
    new_index = pd.Index(np.unique(np.concatenate(
        (old_time, new_time))), name='time')
    tmp = df.reindex(index=
                     new_index
                     ).interpolate(
                         method=upsampling_interpolation_method).reindex(index=new_time)
    tmp.index.name = df.index.name
    return tmp

def interval_union(interval_list1,interval_list2):
    '''
    interval_list1 : np.array, (N,2) where N = number of intervals
    interval_list2 : np.array, (N,2) where N = number of intervals

    e.g.
    interval_list1 = np.array([[1,11],[6,8],[12,20],[25,30]])
    interval_list2 = np.array([[1,10]])

    Returns
    -------
    interval_list: np.array, (N,2)

    '''
    # make sure 1 dimensional arrays are tamed.
    interval_list1 = interval_list1.reshape((-1,2)) 
    interval_list2 = interval_list2.reshape((-1,2))
    
    interval_list1 = interval_list1.tolist()
    interval_list2 = interval_list2.tolist()

    # find all pairwise intersections
    intersect_tally=np.zeros((len(interval_list1),len(interval_list2)))
    for i in range(len(interval_list1)):
        for j in range(len(interval_list2)):
            if _intersection(interval_list1[i],interval_list2[j]) is not None:
                intersect_tally[i,j]=1

    union_set=[]
    union_list2_ind=np.argwhere(np.sum(intersect_tally,axis=0)==0).ravel()
    union_list1_ind=np.argwhere(np.sum(intersect_tally,axis=1)==0).ravel()

    # for those in set 1 or set 2 that has no intersections, append zero row sum or column sum
    for i in union_list1_ind:
        union_set.append(interval_list1[i])
    for i in union_list2_ind:
        union_set.append(interval_list2[i])

    # for those that has intersections, find mutually intersecting interva;s
    for i in np.argwhere(np.sum(intersect_tally,axis=1)!=0).ravel():
        intvl_1=np.array([interval_list1[i]])

        intvl_1s=[]
        # first, find the intersecting one in set 2
        j_all=np.argwhere(intersect_tally[i,:]).ravel()
        intvl_2s=np.array([interval_list2[j] for j in j_all])

        # then track back in set 1, to find all the ones in set 1 intersecting this one in set 2
        for j in j_all:
            i_all=np.argwhere(intersect_tally[:,j]).ravel()
            for i in i_all:
                intvl_1s.append(interval_list1[i])

        # put all the related/intersecting intervals together, find union
        union_tmp=np.concatenate([np.array(intvl_1s),intvl_2s],axis=0)

        union_set.append([np.min(union_tmp[:,0]),np.max(union_tmp[:,1])])
    union_set=np.unique(union_set,axis=0)
    return union_set


def mergeIntervals(intervals):
    # Sort the array on the basis of start values of intervals.
    intervals.sort()
    stack = []
    # insert first interval into stack
    stack.append(intervals[0])
    for i in intervals[1:]:
        # Check for overlapping interval,
        # if interval overlap
        if stack[-1][0] <= i[0] <= stack[-1][-1]:
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)

    return stack

def find_trial_id(c_end,log_df):
    '''
    find the trial in which an event time c_end occurs

    c_end is a float or np array
    log_df is pd datafrmae
    '''
    trial_id = None

    # if it is between outerwell (trial t) and home poke (trial t+1)
    tmp=c_end>log_df.timestamp_O
    trial_id_min=tmp[::-1].idxmax()
    trial_id_max=(c_end<log_df.timestamp_H).idxmax()
    if (trial_id_max-trial_id_min)==1:
        trial_id=trial_id_min
        return trial_id
    elif (trial_id_max-trial_id_min)>1: #trial t+1 did not poke home
        if (c_end-log_df.loc[trial_id_min].timestamp_O)<=5: #less than 5 seconds from outer poke:
            trial_id=trial_id_min
            return trial_id

    # if it is between home poke (trial t) and outerwell (trial t+1)
    tmp=c_end>log_df.timestamp_H
    trial_id_min=tmp[::-1].idxmax()
    trial_id_max=(c_end<log_df.timestamp_O).idxmax()
    if (trial_id_max-trial_id_min)==0:
        trial_id=trial_id_min
        return trial_id
    elif (trial_id_max-trial_id_min)>1: #trial t did not poke home
        if (c_end-log_df.loc[trial_id_min].timestamp_H)<=5: #less than 5 seconds from outer poke:
            trial_id=trial_id_min
            return trial_id
        

def _intersection(interval1, interval2):
    """Takes the (set-theoretic) intersection of two intervals"""
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    intersection = np.array([start, end]) if end > start else None
    return intersection


def _union(interval1, interval2):
    """Takes the (set-theoretic) union of two intervals"""
    if _intersection(interval1, interval2) is None:
        return np.array([interval1, interval2])
    return np.array(
        [min(interval1[0], interval2[0]), max(interval1[1], interval2[1])]
    )
    
def intersection_of_lists(arr1, arr2):
    ### intersections of list of 2-element list
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    
    # 1. View arrays as a structured data type (e.g., 'i,i' for two integers)
    # This lets NumPy treat each row as a single comparable item
    dtype = [('f1', int), ('f2', int)]
    arr1_structured = arr1.view(dtype)
    arr2_structured = arr2.view(dtype)

    # 2. Find which elements in arr1 are also in arr2 using numpy.isin
    # This returns a boolean mask
    mask = np.isin(arr1_structured, arr2_structured)

    # 3. Get the indices from the mask using numpy.nonzero
    # The indices correspond to the rows in the original arr1 that are in the intersection
    arr1_indices = np.nonzero(mask)[0]

    # 4. To get the corresponding indices in arr2, you can create a temporary map
    # or use a loop for a more direct index retrieval in arr2
    arr2_indices = []
    for item in arr1[arr1_indices]:
        # Use np.where to find the index of the matching row in arr2
        # The condition below finds where the row in arr2 matches the current item from arr1
        # We use (arr2 == item).all(axis=1) to check if all elements in the row match
        match_indices = np.where((arr2 == item).all(axis=1))[0]
        if match_indices.size > 0:
            arr2_indices.append(match_indices[0]) # Append the first matching index


    return arr1[arr1_indices], arr1_indices.tolist(), arr2_indices

def select_list_elements(list1, inds):
    return [list1[ind] for ind in inds]