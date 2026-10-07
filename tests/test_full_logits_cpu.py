"""Compare the exported chunked full-logit kernel to a dense CPU reference."""
import importlib.util
from pathlib import Path
import unittest
try:
    import torch
except ImportError:
    torch=None
ROOT=Path(__file__).resolve().parents[1]

@unittest.skipIf(torch is None,'Install torch to run kernel checks')
class FullLogitTests(unittest.TestCase):
    def test_loss_and_gradients_with_unequal_vocabularies(self):
        spec=importlib.util.spec_from_file_location('full_logits',ROOT/'vendor/verl-full/verl/workers/full_logits.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        torch.manual_seed(42)
        for chunk in [1,3,10]:
            h=torch.randn(5,4,dtype=torch.float64,requires_grad=True)
            w=torch.randn(7,4,dtype=torch.float64,requires_grad=True)
            oh=torch.randn(5,4,dtype=torch.float64);ow=torch.randn(7,4,dtype=torch.float64)
            th=torch.randn(5,4,dtype=torch.float64);tw=torch.randn(11,4,dtype=torch.float64)
            slp=torch.log_softmax(oh@ow.T,dim=-1)
            tlp=torch.log_softmax(th@tw.T,dim=-1)[:,:7]
            adv=-(slp-tlp)*slp.exp()
            lp=torch.log_softmax(h@w.T,dim=-1)
            dense=(-adv.detach()*(lp-lp.detach()).exp()).sum()/len(h)
            dg=torch.autograd.grad(dense,(h,w))
            value=m.FullSupportSurrogate.apply(h,w,oh,ow,th,tw,1.,1.,chunk)
            cg=torch.autograd.grad(value,(h,w))
            torch.testing.assert_close(value,dense,atol=1e-10,rtol=1e-10)
            for actual,expected in zip(cg,dg):torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)
if __name__=='__main__':unittest.main()
