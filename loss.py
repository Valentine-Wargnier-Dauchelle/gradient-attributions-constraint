
from expected_gradients import AttributionPriorExplainer

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

######################################################################

class GradientLoss(nn.Module):

    def __init__(self, mulIn=True, steps=30, ref_data=None, k_ref=1, batch_size=None, p_null_ref=0):
        super(GradientLoss,self).__init__()

        self.BCE = torch.nn.BCELoss()
        self.mulIn = mulIn
        self.steps = steps
        self.exp_grads = None
        if batch_size is not None:
            self.exp_grads = AttributionPriorExplainer(ref_data, batch_size, k=k_ref, scale_by_inputs=mulIn, p_null_ref=p_null_ref)
  
    def forward(self, source, img, target, size=None, middle=False):
        assert(img.requires_grad==True)
        ##IG
        if self.steps>0:
            attributions = torch.zeros_like(img).to(Device.device)
            integrated_gradients = IntegratedGradients(source, multiply_by_inputs=self.mulIn)
            for n in range(size[0]):
                for m in range(size[1]):
                    for l in range(size[2]):
                        attributions += integrated_gradients.attribute(img, target=(0,n,m,l), n_steps = self.steps, internal_batch_size=5) # internal_batch_size for memory

            attributions /= np.prod(size)
        ##EG
        elif self.exp_grads is not None:
            attributions = self.exp_grads.shap_values(source, img)
        ##G
        else:
            mode='trilinear' if len(img.size())==5 else 'bilinear'
            output = source(img)       
            output = torch.mean(output, list(range(1, len(output.size())))) 
            attributions = torch.autograd.grad(output, img, grad_outputs=torch.ones_like(output), create_graph=True)[0]
        
        loss = torch.tensor(0.).to(Device.device)
		target_init = target
        if target_init is not None:
            while target.size()[1] != attributions.size()[1]:
                target = torch.cat((target, target_init), 1)
            att_toComp =  attributions[~target.isnan()]
            target = target[~target.isnan()]
            if middle:
                att_toComp = att_toComp[:, int(att_toComp.size()[1]/2)::]
                target = target[:, int(target.size()[1]/2)::]
            if target.nelement()==0:
                loss = torch.tensor(0).to(Device.device)
            elif self.loss=="BCEwL":
                loss = self.BCE(att_toComp, target.float())
            elif self.loss == "ross":
                loss = torch.sum((1-target)*torch.square(att_toComp))
            else:
                raise NameError('Unknown gradient loss')

        
        return loss, attributions
