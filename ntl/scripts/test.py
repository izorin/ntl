import numpy as np
from tqdm.auto import tqdm
import time


def inside_loop(idx):
    a = np.arange(10)
    t = tqdm(a, leave=False)
    t.set_description(str(idx))
    s = 0
    for i, x in enumerate(t):
        s += x
        time.sleep(0.2)
        
    # t.clear()
    
    
    return s

def outside_loop():
    total = []
    t = tqdm(range(10))
    for i in t:
        s = inside_loop(i)
        total.append(s)
        time.sleep(0.2)
    
    print(total)
    
    


if __name__ == '__main__':
    outside_loop()