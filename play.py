import os
os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
from src.pde import pde_list

pde_objects = [x() for x in pde_list]
for pde in pde_objects:
    print(pde.__class__.__name__, end=' ')
    bbox = np.array(pde.bbox)
    area = bbox[1::2]-bbox[::2]
    print("dbboxs:", area, end=" -> ")
    print("sigma=", area/2000)
