# matrix height and width
MH = 
MW = 

# tile height and width
TH =
TW = 

#################### Start: Directly from Paper

with mlir_mod_ctx() as ctx:
  @device(AIEDevice.npu1_1col)
  def device_body():
    mty = np.ndarray[(MH, MW), np.dtype[np.int32]]
    tty = np.ndarray[(TH, TW), np.dtype[np.int32]]
    shm0 = tile(0, 0)
    aie2 = tile(0, 2)
    fi = object_fifo("in", shm0, aie2, 2, tty)
    fo = object_fifo("out", aie2, shm0, 2, tty)

    @core(aie2)
    def core_body():
      for _ in range_(sys.maxsize):
        a = fi.acquire(ObjectFifoPort.Consume, 1)
        b = fo.acquire(ObjectFifoPort.Produce, 1)
        for i in range_(TH):
          for j in range_(TW):
            b[i, j] = a[i, j] + 1
        fi.release(ObjectFifoPort.Consume, 1)
        fo.release(ObjectFifoPort.Produce, 1)

    @runtime_sequence(mty, mty)
    def sequence(dati, dato):
      in_task = shim_dma_single_bd_task(fi, dati, sizes=[1, 1, TH, TW], strides=[0, 0, MW, 1])
      out_task = shim_dma_single_bd_task(fo, dato, issue_token=True, sizes=[1, 1, TH, TW], strides=[0, 0, MW, 1])
      dma_start_task(in_task, out_task)
      dma_await_task(out_task)
      dma_free_task(in_task)
  print(ctx.module)

####################### End: Directly from Paper
