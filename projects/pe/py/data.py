import numpy as np

def load_data(path, chfilter, secfilter ):
    """Loads a dataset for training

    Args:
        path (string): Path to file to load
        
    Returns:
        numpy array : col(0): sector; col(1): charge; col(2-7): inputs; col(8-10): outputs 
    """
    
    file_strings = []
    with open(path,'r') as f:
        accumulator = []
        for line in f:
            data = line.split(":")
            first_part = data[0].lstrip().split()
            output = data[1].lstrip().split()
            input = data[2].split("==>")[1].lstrip().split()[:6]
            sector = int(first_part[0])
            charge = first_part[1]
            accumulator.append([int(sector), int(charge), float(input[0]), float(input[1]), float(input[2]), float(input[3]), float(input[4]), float(input[5]), float(output[0]), float(output[1]), float(output[2])])
    
    with open(path,'r') as f:
        file_strings = f.readlines()
    data = np.array(accumulator)
    data, filter = filter_data(data, chfilter, secfilter)
    idx_strings = []
    for f,s in zip(filter, file_strings):
        if f:
            idx_strings.append(s.strip())
    
    
    return data, idx_strings
            
def filter_data(data, chfilter, secfilter):
    """Returns a new array with only those adapting to the filtering criteria

    Args:
        data (numpy array): The dataset to filter
        chfilter (numpy array): Charges to keep
        secfilter (numpy array): Sectors to keep

    Returns:
        numpy array: The filtered dataset
    """
    filter = np.ones(data.shape[0], dtype = np.bool8)
    if chfilter:
        filter = np.zeros(data.shape[0], dtype = np.bool8)
        for chf in chfilter:
            filter = filter | (data[:,1] == chf)
    
    ch_filtered = None
    if chfilter:
        ch_filtered = data[filter]
    else:
        ch_filtered = data
    
    if secfilter:
        filter = np.zeros(ch_filtered.shape[0], dtype = np.bool8)
        for secf in secfilter:
            filter = filter | (ch_filtered[:,0] == secf)

    if secfilter:
        filtered = ch_filtered[filter]
    else:
        filtered = ch_filtered
    
    return filtered, filter
