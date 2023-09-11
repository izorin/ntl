import numpy as np
from numpy import ndarray

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, robust_scale
from sktime.transformations.series.impute import Imputer

import torch


class Cutout:
    def __init__(self, length: int=-1):
        self.length = length
    
    def __call__(self, x: ndarray) -> ndarray:
        if self.length == -1: 
            return x
        
        elif self.length == 0:
            print('empty subsequence')
            raise ValueError
        
        else:
            start = np.random.randint(max(x.shape) - self.length)
            return x[start: start + self.length]    
        
# def cutout_fn(x: np.array, length: int=-1) -> np.array:
#     if length == -1: 
#         return x
    
#     elif length == 0:
#         print('empty subsequence')
#         raise ValueError
    
#     else:
#         start = np.random.randint(x.shape.max() - length)
#         return x[start:start+length]



class FillNA:
    def __init__(self, method: str=None):
        self.method = method
        
    def __call__(self, x: ndarray) -> ndarray:
        if self.method is None or self.method == 'linear':
            return Imputer(method='linear').fit_transform(x)
        
        elif self.method == 'drift':
            return Imputer(method=self.method).fit_transform(x)
        
        else:
            print(f'unknown method: `{self.method}`')
            raise ValueError
    
        
# def fillna_fn(x: np.array, method: str=None) -> np.array:
#     if method is None or method == 'linear':
#         return Imputer(method='linear').fit_transform(x)
    
#     elif method == 'drift':
#         return Imputer(method=method).fit_transform(x)
    
#     else:
#         print(f'unknown method: `{method}`')
#         raise ValueError


class Scale:
    def __init__(self, method: str=None):
        self.method = method
        
    def __call__(self, x: ndarray) -> ndarray:
        if self.method is None or self.method == 'minmax':
            return MinMaxScaler().fit_transform(x)
        
        elif self.method == 'standard':
            return StandardScaler().fit_transform(x)
        
        elif self.method == 'maxabs':
            return MaxAbsScaler().fit_transform(x)
        
        elif self.method == 'robust':
            return robust_scale(x)
        
        else:
            print(f'unknown method: `{self.method}`')
            raise ValueError    

# def scale_fn(x: np.array, method: str=None) -> np.array:
    
#     if method is None or method == 'minmax':
#         return MinMaxScaler().fit_transform(x)
    
#     elif method == 'standard':
#         return StandardScaler().fit_transform(x)
    
#     elif method == 'maxabs':
#         return MaxAbsScaler().fit_transform(x)
    
#     elif method == 'robust':
#         return robust_scale(x)
    
#     else:
#         print(f'unknown method: `{method}`')
#         raise ValueError


class Reshape:
    # TODO check what dimension to reshape as square (where is the sequence length dimension)
    def __init__(self, shape: tuple[int], pad: bool=True):
        self.shape = shape
        self.pad = pad
        
    def __call__(self, x: ndarray) -> ndarray:
        x = x.squeeze()
        L = max(x.shape)
        if len(self.shape) == 2:
            M, N = self.shape
            if L // M != N:
                if self.pad:
                    pad_len = M - L % M
                    x = np.pad(x, (0, pad_len), mode='constant', constant_values=0)
                else:
                    cut_len = L % M
                    x = x[:-cut_len]
                    
        
        return x.reshape(*self.shape)
    
# def reshape_fn(x: np.array, shape: tuple[int]) -> np.array:
#     return x.reshape(*shape, -1)


class ToTensor:
    def __init__(self, dtype: torch.dtype=torch.float32, device: str='cpu'):
        self.dtype = dtype
        self.device = device
        
    def __call__(self, x: ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype, device=self.device)
    


# def totensor_fn(x: np.array, dtype: torch.dtype=torch.float32, device: str='cpu') -> torch.Tensor:
#     return torch.from_numpy(x, dtype=dtype, device=device)
