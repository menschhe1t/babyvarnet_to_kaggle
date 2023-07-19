import h5py
import random
from utils.data.transforms import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from pathlib import PosixPath

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.examples = []

        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        #################################################### 예외처리
        #print(fname)
        a=''
        if fname == PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc4_141.h5'):
            a = PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc4_1.h5')
            #print('path changed')

        elif fname == PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc8_99.h5'):
            a = PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc8_2.h5')
        #    print('path changed')
        elif fname == PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/val/image/brain_acc8_190.h5'):
            a = PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/val/image/brain_acc8_189.h5')
         #   print('path changed')
        else :
            a = fname
        fname=a
        ####################################################
        
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        
        #################################################### 예외처리
        a=''
        if fname == PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc4_141.h5'):
            a = PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc4_1.h5')
        #    print('path changed')

        elif fname == PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc8_99.h5'):
            a = PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/train/image/brain_acc8_2.h5')
        #    print('path changed')
        elif fname == PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/val/image/brain_acc8_190.h5'):
            a = PosixPath('/kaggle/input/fmrikaggle2try/2023_snu_fastmri_dataset_onlyimage/val/image/brain_acc8_189.h5')
        #    print('path changed')
        else :
            a = fname
        fname = a
        
        #print('getitem')
        #print(fname)
        ####################################################
        with h5py.File(fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
        return self.transform(input, target, attrs, fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
