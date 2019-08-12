# install_numba_and_pyculib_by_pip
Installation instructions for numba and pyculib by pip, tested on Ubuntu.

## Install numba ##

```shell
$ pip install numba
```

Set environment variable

```shell
$ export NUMBAPRO_CUDALIB=/usr/local/cuda/lib64/
$ export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
$ export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/
```

First example - vector add

```python
import numpy as np
from numba import vectorize

@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
  return a + b

# Initialize arrays
N = 1000000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)
# Add arrays on GPU
C = Add(A, B)
```

## Install pyculib ##
```shell
$ pip install pyculib
$ pip install scipy
```

Download [pyculib_soring](https://github.com/numba/pyculib_sorting), build and install it manually. Before building, have to modify “build_sorting_libs.py” in order to fit GPU architecture.

```diff
# assume GPU architecture is 6.1
--- a/build_sorting_libs.py
+++ b/build_sorting_libs.py
@@ -46,6 +46,7 @@ def gencode_flags():
     GENCODE_SM50 = GENCODE_SMXX.format(CC=50)
     GENCODE_SM52 = GENCODE_SMXX.format(CC=52)
     GENCODE_SM53 = GENCODE_SMXX.format(CC=53)
+    GENCODE_SM61 = GENCODE_SMXX.format(CC=61)
 
     # Provide forward-compatibility to architectures beyond CC 5.3
     GENCODE_COMPUTEXX = "-gencode arch=compute_{CC},code=compute_{CC}"
@@ -53,14 +54,15 @@ def gencode_flags():
 
     # Concatenate flags
     SM = []
-    SM.append(GENCODE_SM20)
-    SM.append(GENCODE_SM30)
-    SM.append(GENCODE_SM35)
-    SM.append(GENCODE_SM37)
-    SM.append(GENCODE_SM50)
-    SM.append(GENCODE_SM52)
-    SM.append(GENCODE_SM53)
-    SM.append(GENCODE_COMPUTE53)
+    SM.append(GENCODE_SM61)
     return ' '.join(SM)
```

Make a soft-link for sorting library of pyculib.

```shell
# in pyculib_sorting folder
ln -s ./lib/pyculib_radixsort.so /usr/lib/pyculib_radixsort.so
ln -s ./lib/pyculib_segsort.so   /usr/lib/pyculib_segsort.so
```
or 
```shell
ln -s ./lib/pyculib_radixsort.so /usr/local/lib/python3.6/dist-packages/pyculib/sorting/pyculib_radixsort.so
ln -s ./lib/pyculib_segsort.so /usr/local/lib/python3.6/dist-packages/pyculib/sorting/pyculib_segsort.so
```

First pyculib example

```python
import numpy as np
from pyculib import rand as curand
prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
rand = np.empty(100000)
prng.uniform(rand)
print(rand[:10])
```
