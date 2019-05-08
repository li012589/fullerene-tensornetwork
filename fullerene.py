import torch
import numpy as np

T1 = torch.empty(2,2,2,2,2)
ran = [1,-1]
for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                for i4,s4 in enumerate(ran):
                    T1[i0,i1,i2,i3,i4] = torch.exp(-(s0*s1+s1*s2+s2*s3+s3*s4+s4*s0))

T2 = torch.empty(2,2,2,2,2,2)
ran = [1,-1]
for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                for i4,s4 in enumerate(ran):
                    for i5,s5 in enumerate(ran):
                        T2[i0,i1,i2,i3,i4] = torch.exp(-(s0*s1+s1*s2+s2*s3+s3*s4+s4*s5+s5*s0))



