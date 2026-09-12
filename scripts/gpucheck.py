# NVML-free GPU 巡检: nvidia-smi 因 595.91 用户态 vs 595.84 内核模块不匹配而不可用时用这个。
# 只走 CUDA driver API, 不建 context(不占显存), 再列出本用户持有 /dev/nvidia* 的进程。
import ctypes, os, glob
lib = ctypes.CDLL('libcuda.so.1')
rc = lib.cuInit(0)
v = ctypes.c_int(); lib.cuDriverGetVersion(ctypes.byref(v))
print('cuInit rc=%d (0=OK)  driver_api=%d.%d' % (rc, v.value//1000, (v.value%1000)//10))
n = ctypes.c_int(); lib.cuDeviceGetCount(ctypes.byref(n))
for i in range(n.value):
    d = ctypes.c_int(); lib.cuDeviceGet(ctypes.byref(d), i)
    nm = ctypes.create_string_buffer(128); lib.cuDeviceGetName(nm, 128, d)
    tot = ctypes.c_size_t(); lib.cuDeviceTotalMem_v2(ctypes.byref(tot), d)
    print('GPU%d %s  total=%.1f GiB' % (i, nm.value.decode(), tot.value/2**30))
procs = {}
for fd in glob.glob('/proc/[0-9]*/fd/*'):
    try:
        if os.readlink(fd).startswith('/dev/nvidia'):
            procs.setdefault(fd.split('/')[2], 0)
    except OSError:
        pass
print('holding /dev/nvidia* (only this user visible):')
for pid in sorted(procs, key=int):
    try:
        cl = open('/proc/%s/cmdline' % pid).read().replace(chr(0), ' ').strip()[:110]
    except OSError:
        cl = '?'
    print('  %s %s' % (pid, cl))
