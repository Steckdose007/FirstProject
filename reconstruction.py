import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import MRzeroCore as mr0
import pulseqzero

def reconstruct_signal(Nread,Nphase,signal,encoding,obj_p):
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
    mr0.util.imshow(np.abs(kspace.detach().cpu().numpy()), cmap="gray")

    plt.subplot(349)
    plt.title('k-space_r')
    mr0.util.imshow(np.log(np.abs(kspace.detach().cpu().numpy()) + 1e-6), cmap="gray")  # log avoids log(0)

    plt.subplot(346)
    plt.title('FFT-magnitude')
    mr0.util.imshow(np.abs(space.detach().cpu().numpy()), cmap="gray")
    plt.colorbar()

    plt.subplot(3, 4, 10)
    plt.title('FFT-phase')
    mr0.util.imshow(np.angle(space.detach().cpu().numpy()), cmap="gray", vmin=-np.pi, vmax=np.pi)
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