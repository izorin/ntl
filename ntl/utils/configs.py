import os 
import sys 

def define_pathes():
    os_name = None
    if os_name == 'linux': 
        module_path = ''
        data_path = ''
    elif os_name == 'darwin':
        module_path = ''
        data_path = ''
        
    else:
        raise ValueError
    
    return module_path, data_path
    
    
    
if __name__ == '__main__':
    define_pathes()