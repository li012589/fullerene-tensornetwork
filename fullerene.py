import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-K",type = float,default=1,help="")

args = parser.parse_args()
torch.set_grad_enabled(False)

K = torch.tensor(args.K).double()

T1 = torch.empty(2,2,2,2,2).double()
ran = [torch.tensor(1).double(),torch.tensor(-1).double()]
for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                for i4,s4 in enumerate(ran):
                    T1[i0,i1,i2,i3,i4] = torch.exp(-K*(s0*s1+s1*s2+s2*s3+s3*s4+s4*s0))

T2 = torch.empty(2,2,2,2,2,2).double()
for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                for i4,s4 in enumerate(ran):
                    for i5,s5 in enumerate(ran):
                        T2[i0,i1,i2,i3,i4] = torch.exp(-K*(s0*s1+s1*s2+s2*s3+s3*s4+s4*s5+s5*s0))


fan = torch.einsum("abcdef,gfeih,edlkji,qhijop,jkmno,pontsr->abcdlkmntsrpqhgf",T2,T1,T2,T2,T1,T2)
fan = fan.reshape(2,2,2**6,2,2,2**6)

res = torch.einsum("eaoijk,abnhio,bcmghn,cdlfgm,dekjfl->abcdejfghi",fan,fan,fan,fan,fan)
z = torch.einsum("abcdejfghi,abcde,jihgf->",res,T1,T1)

lnz = torch.log(z)

print(lnz)

import pdb
#pdb.set_trace()