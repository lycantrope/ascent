import torch
import torch.distributed as dist


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list: list[torch.Tensor], tensor: torch.Tensor):
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        dist.all_gather([t.contiguous() for t in tensor_list], tensor.contiguous())
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True)
            for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            if op is None:
                continue
            op.wait()

        return None, grad_list[rank]


all_gather = AllGather.apply
