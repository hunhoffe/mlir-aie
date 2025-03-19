import numpy as np

# New IRON API imports
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

# matrix height and width
MH = 16
MW = 128

# tile height and width
TH = 8
TW = 16

#################### Start: Directly from Paper

mty = np.ndarray[(MH, MW), np.dtype[np.int32]]
tty = np.ndarray[(TH, TW), np.dtype[np.int32]]
fi = ObjectFifo(tty)
fo = ObjectFifo(tty)


def core_fn(of_in, of_out):
    a = of_in.acquire(1)
    b = of_out.acquire(1)
    for i in range_(TH):
        for j in range_(TW):
            b[i, j] = a[i, j] + 1
    of_in.release(1)
    of_out.release(1)


my_worker = Worker(core_fn, fn_args=[fi.cons(), fo.prod()])

rt = Runtime()
with rt.sequence(mty, mty) as (dati, dato):
    rt.start(my_worker)

    ### Start Modified from Paper
    tap = TensorAccessPattern([MH, MW], sizes=[1, 1, TH, TW], strides=[0, 0, MW, 1], offset=0)
    rt.fill(fi.prod(), dati, tap)
    rt.drain(fo.cons(), dato, tap, wait=True)
    #### End Modified from Paper

    #### Original from paper
    # rt.drain(fo.cons(), dato, sizes=[1, 1, TH, TW], strides=[0, 0, MW, 1], wait=True)
    # rt.fill(fi.prod(), dati, sizes=[1, 1, TH, TW], strides=[0, 0, MW, 1])
print(Program(NPU1Col1(), rt).resolve_program(SequentialPlacer()))

####################### End: Directly from Paper
