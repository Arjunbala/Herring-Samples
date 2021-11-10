# My custom DDP

from torch.nn.modules import Module

class DistributedDataParallel(Module):
    # Keeping device_ids to keep compatibility with example code I have
    def __init__(self, module, device_ids=None):
        super(DistributedDataParallel, self).__init__()
        self.module = module
        for param in self.module.parameters():
            print(type(param), param.size())
            if param.requires_grad:
                param.register_hook(lambda grad: self._grad_hook(grad))

    def _grad_hook(self, grad):
        print("Original grad: ", grad)
        return 0.1*grad

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
