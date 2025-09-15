 # %%  IMPORTS

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import MRzeroCore as mr0
import pulseqzero
import sequence_script as seqs
import reconstruction 
import optimization as opt
pp = pulseqzero.pp_impl

def make_obj_simulation(sz,fov,slice_thickness,build=True,inhomogeneity=False):
    if 1:
        # (i) load a phantom object from file
        # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
        obj_p = mr0.VoxelGridPhantom.load_mat('numerical_brain_cropped.mat')
        obj_p = obj_p.interpolate(sz[0], sz[1], 1)

    # Manipulate loaded data
        obj_p.T2dash[:] = 30e-3
        obj_p.D *= 0 
        if inhomogeneity:
            obj_p.B0 *= 1    # alter the B0 inhomogeneity
            #obj_p.B1[:] = 1
            obj_p.B1 *= 1  # alter the B1 inhomogeneity
        else:#no inhomogeneity
            obj_p.B0 *= 0   
            obj_p.B1[:] = 1  
        # Store PD for comparison
        PD = obj_p.PD.squeeze()
        B0 = obj_p.B0.squeeze()
    #obj_p.plot()
    obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
    if build:
        # Convert Phantom into simulation data
        obj_p = obj_p.build()
    return obj_p

# %%  SIMULATE  the external.seq file and add acquired signal to ADC plot + Recon
if __name__=="__main__":
    base_resolution=42
    Ex_FA=90    # excitation flip angle
    Ref_FA_target= torch.full((base_resolution,), 180)  # refocusing flip angle
    fov=200e-3
    slice_thickness=8e-3
    # base_resolution=42, # This line was causing the error
    TE_ms=5
    TI_s=0
    r_spoil=2
    PE_grad_on=True
    RO_grad_on=True
    seq, encoding = seqs.tse_sequence(pp=pp,fov=fov, slice_thickness=slice_thickness, base_resolution=base_resolution,
                    TE_ms=TE_ms, TI_s=TI_s, Ex_FA=Ex_FA, Ref_FA=Ref_FA_target, r_spoil=r_spoil,
                    PE_grad_on=PE_grad_on, RO_grad_on=RO_grad_on)

    seq.set_definition('FOV', [fov, fov, slice_thickness])
    seq.set_definition('Name', 'gre')
    seq.write('external.seq')
    sz = [64, 64]  # image size
    obj_p = make_obj_simulation(sz=sz,fov=fov,slice_thickness=slice_thickness)
    
    # Read in the sequence 
    seq0 = mr0.Sequence.import_file("external.seq")
    
    #seq0.plot_kspace_trajectory()
    # Simulate the sequence
    graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
    signal = mr0.execute_graph(graph, seq0, obj_p)

    # PLOT sequence with signal in the ADC subplot
    mr0.util.pulseq_plot(seq, signal=signal.numpy())

    #  RECONSTRUCT the simulated data
    # additional noise as simulation is perfect
    #signal += 1e-4 * np.random.randn(signal.shape[0], 2).view(np.complex128)
    reconstruction.reconstruct_signal(Nread=base_resolution,Nphase=base_resolution,signal=signal,encoding=encoding,obj_p=obj_p)

    # %% OPTIMIZE the refocusing flip angles
    obj_p = make_obj_simulation(sz=sz,fov=fov,slice_thickness=slice_thickness,build=False,inhomogeneity=True)
    #target is done without any inhomogeneity
    target,Ref_FA_target = opt.generate_target_space(pp,sz)
    iterations= 100
    opt.perform_optimazation(obj_p, target, pp, Ref_FA_target,iterations,base_resolution,Ex_FA)

# %%
