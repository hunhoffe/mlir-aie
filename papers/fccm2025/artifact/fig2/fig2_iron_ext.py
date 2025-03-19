# matrix height and width
MH = 
MW = 

# tile height and width
TH =
TW = 

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
    rt.fill(fi.prod(), dati, sizes=[1, 1, TH, TW], strides=[0, 0, MW, 1])
    rt.drain(fo.cons(), dato, sizes=[1, 1, TH, TW], strides=[0, 0, MW, 1], wait=True)
print(Program(NPU1Col1(), rt).resolve_program(SequentialPlacer()))

####################### End: Directly from Paper
