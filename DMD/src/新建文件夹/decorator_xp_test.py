import numpy as np 
try:
    import cupy as cp
except Exception:
    cp = None
global xp
xp = np

def xp_agnostic(func):
    def wrapper(*args, **kwargs):
        global xp
        try:
            xp = cp.get_array_module(args[0])
        except Exception:
            xp = np
        print(xp)
        return func(*args, **kwargs)
    return wrapper

@xp_agnostic
def simple_sum(a, b):
    return xp.sum(a, b)

if __name__ == '__main__':
    
    import numpy as np 
    import cupy as cp

    a = cp.array([1, 2])
    b = cp.array([2])

    bb = simple_sum(a, b)
    print(a)
    print(type(bb))