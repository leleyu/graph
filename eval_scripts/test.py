import numpy as np

import sys
path = sys.argv[1]

d = np.fromfile(path, dtype=np.float32)
print(d)
