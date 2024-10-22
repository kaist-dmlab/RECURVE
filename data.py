from torch.utils.data import Dataset
import numpy as np


def list_flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


def generate_boundary_labels(label_list, ratio):
    boundary_list = []
    segment_len_list = []
    label_seg_list = []

    for video_label in label_list:

        label_seg_list.append(np.zeros(len(video_label)))
        boundaries = []
        segment_len = []
        length = 0
        for ind, (prev_label, curr_label) in enumerate(zip(video_label, video_label[1:])):
            length += 1
            if prev_label != curr_label:
                boundaries.append(ind)
                segment_len.append(length)
                length = 0
        if length != 0:
            segment_len.append(length)
        if len(boundaries) != len(segment_len)-1:
            segment_len.append(1)
        boundary_list.append(boundaries)
        segment_len_list.append(segment_len)

    for i in range(len(boundary_list)):
        for j in range(len(boundary_list[i])):
            lhs_boundary_length = segment_len_list[i][j] * ratio
            rhs_boundary_length = segment_len_list[i][j + 1] * ratio
            boundary_length = np.minimum(
                lhs_boundary_length, rhs_boundary_length)

            start_ind = int(boundary_list[i][j] - boundary_length) + 1
            end_ind = int(boundary_list[i][j] + boundary_length) + 1
            label_seg_list[i][start_ind:end_ind] = 1

    return np.array(list_flatten(boundary_list)).flatten(), \
        np.array(list_flatten(segment_len_list)).flatten(), \
        np.array(list_flatten(label_seg_list)).flatten()

def generate_fixed_bls(*bls, total_len, length=10):
    '''
    Generate boundary label array with shape(total_len, ) with different graduality from 0(gradual) to N(abrupt) as boundary class
    Boundary labels form a range from original boundary indice.
    *bls: iterable of N boundary indice lists
    length: boundary length at each side of true boundary indice.
    '''


    boundary_label_dict = {}
    ts_list_per_abruptness = []
    for bl, cp_list in enumerate(bls):
        for cp in cp_list:
            boundary_label_dict[cp]=bl+1
        ts_list_per_abruptness.append([])

    all_bls = sorted(list_flatten(bls))
    all_bls_arr = np.array(all_bls)
    seg_len_list = (all_bls_arr[1:]-all_bls_arr[:-1]).tolist()
    seg_len_list.insert(0, all_bls[0])
    seg_len_list.append(total_len-all_bls[-1])


    boundary_labels = np.zeros(total_len)
    for bi, boundary in enumerate(all_bls):
        boundary_class = boundary_label_dict[boundary]
        boundary_labels[boundary-length:boundary+length]=boundary_class
        ts_list_per_abruptness[boundary_class-1] += list(range(boundary-length,boundary+length))
    return boundary_labels, ts_list_per_abruptness

def generate_gradual_bls(*bls, total_len, ratio=0.1):
    '''
    Generate boundary label array with shape(total_len, ) with different graduality from 0(gradual) to N(abrupt) as boundary class
    Boundary labels form a range from original boundary indice.
    *bls: iterable of N boundary indice lists
    ratio: boundary label ratio centered at each boundary index
    '''


    boundary_label_dict = {}
    ts_list_per_abruptness = []
    for bl, cp_list in enumerate(bls):
        for cp in cp_list:
            boundary_label_dict[cp]=bl+1
        ts_list_per_abruptness.append([])

    all_bls = sorted(list_flatten(bls))

    all_bls_arr = np.array(all_bls)
    seg_len_list = (all_bls_arr[1:]-all_bls_arr[:-1]).tolist()
    seg_len_list.insert(0, all_bls[0])
    seg_len_list.append(total_len-all_bls[-1])


    boundary_labels = np.zeros(total_len)
    for bi, boundary in enumerate(all_bls):
        lhs_boundary_length = seg_len_list[bi] * ratio
        rhs_boundary_length = seg_len_list[bi + 1] * ratio
        boundary_length = int(np.minimum(lhs_boundary_length, rhs_boundary_length))
        boundary_length = np.maximum(boundary_length, 1)
        boundary_class = boundary_label_dict[boundary]
        boundary_labels[boundary-boundary_length:boundary+boundary_length]=boundary_class
        ts_list_per_abruptness[boundary_class-1] += list(range(boundary-int(seg_len_list[bi]*(1-ratio)),boundary+int(seg_len_list[bi + 1]*(1-ratio))))
    return boundary_labels, ts_list_per_abruptness

def generate_classpair_bls(bls, train_labels, total_len, ratio=0.1):
    '''
    Generate boundary label array with shape(total_len, ) with different boundary class based on class pair
    bls: a boundary indice list
    ratio: boundary label ratio centered at each boundary index

    Returns 
    id_dict: class pair id dictionary {classpair id: ( class left, class right)}
    boundary_labels: boundary class pair labels (0,...,N) where N = the number of unique class pairs
    train_labels: long class labels where labels at the boundary region are erased as -1
    '''
    ratio=0.1
    erased_labels = train_labels.copy()
    seg_len_list = (np.array(bls)[1:]-np.array(bls)[:-1]).tolist()
    seg_len_list.insert(0, bls[0])
    seg_len_list.append(total_len-bls[-1])

    boundary_labels = np.zeros(total_len)
    boundary_id_dict = {}
    boundary_class = 1
    for bi, boundary in enumerate(bls):
        class_pair = (train_labels[boundary], train_labels[boundary+1])
        if not class_pair in boundary_id_dict:
            boundary_id_dict[class_pair]=boundary_class
            boundary_class+=1
        lhs_boundary_length = seg_len_list[bi] * ratio
        rhs_boundary_length = seg_len_list[bi + 1] * ratio
        boundary_length = int(np.minimum(lhs_boundary_length, rhs_boundary_length))
        boundary_labels[boundary-boundary_length:boundary+boundary_length]=boundary_id_dict[class_pair]
    id_dict = {boundary_id_dict[k]:k for k in boundary_id_dict}
    return id_dict, boundary_labels, erased_labels


class TNCDS(Dataset):
    def __init__(self, data, window, n_range=100, n_num=64):
        self.data = data.astype(np.float32)
        self.window = window 
        self.n_num = n_num # the number of neighbors/non-neighbors
        self.n_range = n_range # the prev/next range of neighborhood given a target instance
        assert(self.n_range%2==0)

    def __getitem__(self, index):
        index += self.n_range
        x = self.data[index:index+self.window]
        pos_ind = np.random.choice(int(index)+np.arange(-self.n_range, self.n_range), size=self.n_num)
        neg_ind = np.random.choice(np.arange(self.n_range, len(self.data) - self.window - self.n_range), size=self.n_num)
        pos = np.array([self.data[selected_ind:selected_ind+self.window] for selected_ind in pos_ind])
        neg = np.array([self.data[selected_ind:selected_ind+self.window] for selected_ind in neg_ind])
        return x, pos, neg
    
    def __len__(self):
        return (len(self.data) - 2*self.window - 2*self.n_range) + 1

class WindowedTS_NoLabel(Dataset):
    def __init__(self, data, window, slide=1):
        self.data = data
        self.window = window
        self.slide = slide

    def __getitem__(self, index):
        x = self.data[index*self.slide:index*self.slide+self.window]
        return x

    def __len__(self):
        return (len(self.data) - self.window)//self.slide + 1
    
class TSCP2DS(Dataset):
    def __init__(self, data, window, slide=1):
        self.data = data
        self.window = window # we sample double windows for pos
        self.slide = slide

    def __getitem__(self, index):
        x1 = self.data[index*self.slide:index*self.slide+self.window]
        x2 = self.data[(index+1)*self.slide:(index+1)*self.slide+self.window]
        return x1, x2 #return pos and neg -> nceloss
        # batch = [(1,2), (3,4), (5,6), ] or [1,3,5,7,...,2,4,6,8,...]
    def __len__(self):
        return (len(self.data) - 2*self.window)//self.slide + 1
    