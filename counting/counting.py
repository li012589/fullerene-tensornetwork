import torch
import numpy as np
from numpy.testing import assert_array_almost_equal
torch.set_grad_enabled(False)

T1 = torch.zeros(2,2,2,2,2).to(torch.float64)
T2 = torch.zeros(2,2,2,2,2,2).to(torch.float64)
ran = [torch.tensor(0).to(torch.float64),torch.tensor(1).to(torch.float64)]

for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                for i4,s4 in enumerate(ran):
                    if s0+s1+s2+s3+s4 == 1:
                        print(i0,i1,i2,i3,i4)
                        T1[i0,i1,i2,i3,i4] = 1

for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                for i4,s4 in enumerate(ran):
                    for i5,s5 in enumerate(ran):
                        if ((s0+s2+s4 == 2) and (s0+s1+s2+s3+s4+s5==2)) or (s0+s1+s2+s3+s4+s5==0):
                            print(i0,i1,i2,i3,i4,i5)
                            T2[i0,i1,i2,i3,i4,i5] = 1

fan = torch.einsum("abcdef,jeghi,lkgdnm,hkostu,lpqro,xwvsry->abcnmpqyxfjiutvw",T2,T1,T2,T2,T1,T2)
fan = fan.reshape(2,2**7,2,2**7)
res = torch.einsum("aqlm,bpkq,cojp,dnio,emfn->abcdefijkl",fan,fan,fan,fan,fan)

z = torch.einsum("abcdelkjif,abcde,fijkl->",res,T1,T1)

print(z)
print(2*z.item()) # Z(2) ssymmetry