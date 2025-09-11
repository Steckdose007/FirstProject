import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import MRzeroCore as mr0
import pulseqzero
pp = pulseqzero.pp_impl

# %% S1. Define sequence function
def tse_sequence(fov=200e-3, slice_thickness=8e-3, base_resolution=42,
                 TE_ms=5, TI_s=0, Ex_FA=90, Ref_FA=180, r_spoil=2,
                 PE_grad_on=True, RO_grad_on=True):
    """
    Generates a TSE sequence using PyPulseq.

    Args:
        system: PyPulseq system object.
        fov (float): Field of view in meters (default: 200e-3).
        slice_thickness (float): Slice thickness in meters (default: 8e-3).
        base_resolution (int): Base resolution for frequency and phase encoding (default: 42).
        TE_ms (float): Echo time in milliseconds (default: 5).
        TI_s (float): Inversion time in seconds (default: 0).
        Ex_FA (float): Excitation flip angle in degrees (default: 90).
        Ref_FA (float or array): Refocusing flip angle in degrees (default: 180).
        r_spoil (float): Spoil gradient factor (default: 2).
        PE_grad_on (bool): Enable/disable phase encoding gradients (default: True).
        RO_grad_on (bool): Enable/disable readout gradients (default: True).

    Returns:
        PyPulseq sequence object.
    """
    # Define resolution
    Nread = base_resolution  # frequency encoding steps/samples
    Nphase = base_resolution  # phase encoding steps/samples
    TE = TE_ms * 1e-3

    
    system = pp.Opts(
        max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
        rf_ringdown_time=20e-6, rf_dead_time=100e-6,
        adc_dead_time=20e-6, grad_raster_time=10e-6)

    seq = pp.Sequence(system)

    # Define rf events
    rf1, gz1, gzr1 = pp.make_sinc_pulse(
        flip_angle=Ex_FA * np.pi / 180, phase_offset=90 * np.pi / 180, duration=1e-3,
        slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
        system=system, return_gz=True)

    rf2, gz2, _ = pp.make_sinc_pulse(
        flip_angle=180*np.pi / 180, duration=1e-3,
        slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,
        system=system, return_gz=True)

    dwell=50e-6*2

    G_flag=(int(RO_grad_on),int(PE_grad_on))  # gradient flag (read,PE), if (0,0) all gradients are 0, for (1,0) PE is off

    # Define other gradients and ADC events
    gx = pp.make_trapezoid(channel='x', rise_time = 0.5*dwell, flat_area=Nread / fov*G_flag[0], flat_time=Nread*dwell, system=system)
    adc = pp.make_adc(num_samples=Nread, duration=Nread*dwell, phase_offset=90 * np.pi / 180, delay=0*gx.rise_time, system=system)
    gx_pre0 = pp.make_trapezoid(channel='x', area=+((1.0 + r_spoil) * gx.area / 2) , duration=1.5e-3, system=system)
    gx_prewinder = pp.make_trapezoid(channel='x', area=+(r_spoil * gx.area / 2), duration=1e-3, system=system)
    gp = pp.make_trapezoid(channel='y', area=0 / fov, duration=1e-3, system=system)
    rf_prep = pp.make_block_pulse(flip_angle=180 * np.pi / 180, duration=1e-3, system=system)
    if 0:
        gx = pp.make_delay(0)
        gx_pre0 = pp.make_delay(0)    
        gx_prewinder = pp.make_delay(0)
        gp = pp.make_delay(0)

    # the minimal TE is given by one full period form ref pulse to ref pulse, thus gz2+gx+2*gp
    minTE2=(pp.calc_duration(rf2) +pp.calc_duration(gx) + 2*pp.calc_duration(gp))/2

    minTE2=np.round(minTE2/10e-5)*10e-5
    # to realize longer TE,  we introduce a TEdelay that is added before and afetr the encoding period
    TEd=np.round(max(0, (TE/2-minTE2))/10e-5)*10e-5  # round to raster time
    if TEd==0:
      print('echo time set to minTE [ms]', 2*(minTE2 +TEd)*1000)
    else:
      print(' TE [ms]', 2*(minTE2 +TEd)*1000)
    # last timing step is to add TE/2 also between excitation and first ref pulse
    # from pulse top to pulse top we have already played out one full rf and gx_pre0, thus we substract these from TE/2
    delay_length = 0.010
    #some kind of delay
    ct_rf1=pp.calc_rf_center(rf1)
    ct_rf2=pp.calc_rf_center(rf2)
    TE_long = (rf1.ringdown_time + ct_rf1[0]+ delay_length + rf2.delay + ct_rf2[0])*2
    delay1 = pp.make_delay(TE_long/2 - rf1.ringdown_time - ct_rf1[0] - rf2.delay - ct_rf2[0] - pp.calc_duration(gzr1))
    delay2 = pp.make_delay(TE_long/2 - ct_rf2[0] - rf2.ringdown_time - pp.calc_duration(gx_pre0))
    delay3 = pp.make_delay((minTE2 +TEd ) - ct_rf2[0] - rf2.delay)
    # FLAIR
    if TI_s>0:
      seq.add_block(rf_prep)
      seq.add_block(pp.make_delay(TI_s))
      seq.add_block(gx_pre0)

    #BIG pulse
    seq.add_block(rf1,gz1)
    seq.add_block(gzr1)

    #Delay till first 180
    seq.add_block(delay1)
    
    seq.add_block(rf2,gz2)
    seq.add_block(gx_pre0)
    #Delay till BIG echo
    seq.add_block(delay2)
    #Delay from echo till pulse for echo train
    seq.add_block(delay3)
    #seq.add_block(adc, gx)

    encoding = []

    for ii in range(-Nphase // 2, Nphase // 2):  # e.g. -64:63
        gp  = pp.make_trapezoid(channel='y', area=+ii / fov*G_flag[1], duration=1e-3, system=system)
        gp_ = pp.make_trapezoid(channel='y', area=-ii / fov*G_flag[1], duration=1e-3, system=system)
        encoding.append(ii)

        # Try to index into a variable FA array, if it fails treat it as number
        try:
            flip_angle = Ref_FA[ii] * torch.pi / 180
        except:
            flip_angle = Ref_FA * torch.pi / 180
        rf2, gz2, _ = pp.make_sinc_pulse(flip_angle=flip_angle, duration=1e-3,slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4,system=system, return_gz=True)

        seq.add_block(rf2,gz2)
        seq.add_block(pp.make_delay(TEd)) # TE delay
        seq.add_block(gx_prewinder, gp)
        seq.add_block(adc, gx)
        seq.add_block(gx_prewinder, gp_)
        seq.add_block(pp.make_delay(TEd)) # TE delay

    
    # Check whether the timing of the sequence is correct
    ok, error_report = seq.check_timing()
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    return seq, encoding

# %% S2. CHECK, PLOT 
# Simulate the sequence
obj_p = mr0.util.load_phantom([96,96])
base_resolution=42
Ex_FA=90    # excitation flip angle
Ref_FA_target= torch.full((base_resolution,), 180)  # refocusing flip angle
fov=200e-3
slice_thickness=8e-3
# base_resolution=42, # This line was causing the error
TE_ms=5
TI_s=0
r_spoil=2
PE_grad_on=False
RO_grad_on=True
seq, encoding = tse_sequence(fov=fov, slice_thickness=slice_thickness, base_resolution=base_resolution,
                 TE_ms=TE_ms, TI_s=TI_s, Ex_FA=Ex_FA, Ref_FA=Ref_FA_target, r_spoil=r_spoil,
                 PE_grad_on=PE_grad_on, RO_grad_on=RO_grad_on)

seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')
seq.write('external.seq')
sz = [64, 64]  # image size
if 1:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)

# Manipulate loaded data
    obj_p.T2dash[:] = 30e-3
    obj_p.D *= 0 
    if 1:
        obj_p.B0 *= 2    # alter the B0 inhomogeneity
        #obj_p.B1[:] = 1
        obj_p.B1 *= 3  # alter the B1 inhomogeneity
    else:#no inhomogeneity
        obj_p.B0 *= 0   
        obj_p.B1[:] = 1  
    # Store PD for comparison
    PD = obj_p.PD.squeeze()
    B0 = obj_p.B0.squeeze()
#obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
# Convert Phantom into simulation data
obj_p = obj_p.build()
# %% S5:. SIMULATE  the external.seq file and add acquired signal to ADC plot

# Read in the sequence 
seq0 = mr0.Sequence.import_file("external.seq")
 
#seq0.plot_kspace_trajectory()
# Simulate the sequence
graph = mr0.compute_graph(seq0, obj_p, 200, 1e-3)
signal = mr0.execute_graph(graph, seq0, obj_p)

# PLOT sequence with signal in the ADC subplot
#mr0.util.pulseq_plot(seq, signal=signal.numpy())

# %% S6: RECONSTRUCT the simulated data
# additional noise as simulation is perfect
#signal += 1e-4 * np.random.randn(signal.shape[0], 2).view(np.complex128)

# %% S6: RECONSTRUCT the simulated data
Nread = base_resolution
Nphase = base_resolution

# reshape to k-space
kspace = torch.reshape(signal, (Nread, Nphase)).clone().t()
encoding = np.stack(encoding)
ipermvec = np.argsort(encoding)
kspace = kspace[:, ipermvec]

# fftshift FFT fftshift
spectrum = torch.fft.fftshift(kspace)
space = torch.fft.fft2(spectrum)
space = torch.fft.ifftshift(space)

# --- plotting ---
plt.figure(figsize=(12, 8))

plt.subplot(345)
plt.title('k-space')
mr0.util.imshow(np.abs(kspace.detach().cpu().numpy()))

plt.subplot(349)
plt.title('k-space_r')
mr0.util.imshow(np.log(np.abs(kspace.detach().cpu().numpy()) + 1e-6))  # log avoids log(0)

plt.subplot(346)
plt.title('FFT-magnitude')
mr0.util.imshow(np.abs(space.detach().cpu().numpy()))
plt.colorbar()

plt.subplot(3, 4, 10)
plt.title('FFT-phase')
mr0.util.imshow(np.angle(space.detach().cpu().numpy()), vmin=-np.pi, vmax=np.pi)
plt.colorbar()

# compare with phantom
plt.subplot(348)
plt.title('phantom PD')
mr0.util.imshow(obj_p.recover().PD.squeeze())

plt.subplot(3, 4, 12)
plt.title('phantom B0')
mr0.util.imshow(obj_p.recover().B0.squeeze())

# now actually show everything
plt.tight_layout()
plt.show()


