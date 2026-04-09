## EFA

Need to add EFA resources to enable InfiniBand

in the container, check if EFA is available

```bash
# Check if EFA is available
ls /dev/infiniband
```

### Debug EFA and NCCL

if log says

```bash
NET/Plugin: Could not find: libnccl-net.so.
Initialized NET plugin Socket
Using network Socket

NCCL INFO Channel 23/0 : 1[1] -> 2[2] via P2P/CUMEM/read
```

NCCL is falling back to its internal socket transport over `eth0`

### Solution

Verify `efa` and make sure `libnccl-net-ofi.so` and `libnccl-net.so` exist.

```bash
fi_info -p efa

sudo find / -name 'libnccl-net*.so' 2>/dev/null

ldconfig -p | grep -E 'libnccl-net(|-ofi)\\.so'
```

Add the environment variable to use the OFI plugin for NCCL, which is optimized for EFA.

```bash
export NCCL_NET_PLUGIN=ofi
```

When success, it says

```bash
NET/OFI Initializing aws-ofi-nccl 1.16.2
NCCL INFO NET/OFI Setting provider_filter to efa
Initialized NET plugin Libfabric
Assigned NET plugin Libfabric to comm
Using network Libfabric
```

### Use AWS provided images

Homepage:https://aws.github.io/deep-learning-containers/
Github release: https://github.com/aws/deep-learning-containers/releases
Image list: https://gallery.ecr.aws/deep-learning-containers/pytorch-training

Use public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.33

NVIDIA driver 570.211.01, the practical CUDA toolkit range is 12.x

### Torchrun hostname solver

Use --rdzv-backend=c10d, and --rdzv-endpoint=2x8xa100-80gb-0.svc-2x8xa100-80gb:29500

Static hostname may mismatch and cause hang on AWS, when using `--master-addr=2x8xa100-80gb-0.svc-2x8xa100-80gb`
