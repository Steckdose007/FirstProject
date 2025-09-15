import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
import MRzeroCore as mr0
import pulseqzero
import reconstruction 
import sequence_script as seqs

def generate_target_space(pp,sz):
    obj_p = mr0.util.load_phantom(sz)
    base_resolution=42
    Ex_FA=90    # excitation flip angle
    Ref_FA_target= torch.full((base_resolution,), 180)  # refocusing flip angle

    with pulseqzero.mr0_mode():
        seq, encoding = seqs.tse_sequence(pp,base_resolution=base_resolution, Ex_FA=Ex_FA, Ref_FA=Ref_FA_target)

        seq0 = seq.to_mr0()
        signal,_ = mr0.util.simulate(seq0,obj_p,accuracy=1e-4)

    # reconstruct image
    space = reconstruction.reconstruction_for_optimazation(signal, encoding, base_resolution, base_resolution)

    # plot result
    plt.subplot(121)
    plt.title('FFT-magnitude')
    mr0.util.imshow(np.abs(space.numpy()), cmap=cm.gray)
    plt.colorbar()
    plt.show()

    # store target for optimization
    target = torch.abs(space)
    data_range = target.detach().max() - target.detach().min()
    print(data_range.item())
    return target, Ref_FA_target

def plot_results_images(target, result, finished=False, colorbars=False):

  # show target, initial and optimized image on common colorscale
  vmin = min(target.min(), result.min())
  vmax = max(target.max(), result.max())

  plt.subplot(121)
  plt.title("optimizer target")
  plt.axis('off')
  mr0.util.imshow(target, vmin=vmin, vmax=vmax, cmap=cm.gray)
  if colorbars: plt.colorbar(cmap='gray')

  plt.subplot(122)
  if finished: plt.title("optimizer result")
  else: plt.title("optimizer step")
  plt.axis('off')
  mr0.util.imshow(result, vmin=vmin, vmax=vmax, cmap=cm.gray)
  if colorbars: plt.colorbar(cmap='gray')

  plt.show()

def plot_optimizer_history(loss_hist, param_hist, rSAR_hist, finished=False):
    #plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.title("Loss and rSAR")
    plt.xlabel("itertation")
    plt.plot([l / loss_hist[0] for l in loss_hist], label="loss")
    plt.plot(rSAR_hist, label="rSAR")
    if finished:
      plt.plot([np.argmin(loss_hist), np.argmin(loss_hist)], [[l / loss_hist[0] for l in loss_hist][np.argmin(loss_hist)], rSAR_hist[np.argmin(loss_hist)]], "rx", label="optimum")

    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.xlabel("repetition")
    plt.ylabel("FA")
    plt.title("Optim. param")
    if finished: plt.plot(np.array(param_hist).T)
    else: plt.plot(np.array(param_hist[-2:]).T) # only plot current and last flip angle configuration
    plt.gca().yaxis.tick_right()
    plt.grid()

    plt.show()

def plot_optimized_flipangles(fa_optimized):
  plt.plot(fa_optimized, "o--")
  plt.xlabel("repetition")
  plt.ylabel("FA [deg]")
  plt.title("Optimized refocusing flip angles")
  plt.gca().yaxis.tick_right()
  plt.grid()
  plt.show()

def visualize_results(pp,target,obj_p, loss_hist, param_hist, rSAR_hist, Ref_FA_hist,base_resolution = 42,Ex_FA=90):
    # simute optimizer result: optimal flip angle configuration
    with pulseqzero.mr0_mode():
        seq, encoding = seqs.tse_sequence(pp,Ex_FA=Ex_FA, Ref_FA=Ref_FA_hist[np.argmin(loss_hist)])
        seq0 = seq.to_mr0()
        graph = mr0.compute_graph(seq0, obj_p.build(), 100000, 1e-8)
        signal = mr0.execute_graph(graph, seq0, obj_p.build(), 1e-8, 1e-8)  # high accuracy to check if more states are neccesary

    # reconstruct image
    space = reconstruction.reconstruction_for_optimazation(signal = signal, encoding= encoding,Nread=base_resolution,Nphase=base_resolution,)
    result = torch.abs(space) # current optiumizer step image

    # ====
    # plot results
    # ====

    print(f"Optimized Weights: {Ref_FA_hist[np.argmin(loss_hist)]}")

    # images
    plot_results_images(target, result, finished=True)

    # optimization timeline
    plot_optimizer_history(loss_hist, param_hist,rSAR_hist, finished=True)

    # optimized flip angle configuration
    plot_optimized_flipangles(Ref_FA_hist[np.argmin(loss_hist)])

def perform_optimazation(obj_p, target, pp, Ref_FA_target,iterations=5,base_resolution = 42,Ex_FA=90):
    # sequence parametes
    #base_resolution=42
    #Ex_FA = 90    # excitation flip angle
    #Start parameter for Refocusing flip angles
    Ref_FA = torch.full((base_resolution,), 180.0, requires_grad=True)  # refocusing flip angles

    # initalize optimizer
    params = [{"params": Ref_FA, "lr": 5.0}]  # adjust learning rate as needed
    optimizer = torch.optim.Adam(params)

    lambda_SAR = 0.15
    lambda_image = 1-lambda_SAR

    loss_hist = []
    rSAR_hist = []
    Ref_FA_hist = []

    # optimization loop
    for i in range(iterations):

        optimizer.zero_grad()

        # ====
        # simulate
        # ====

        with pulseqzero.mr0_mode():
            seq, encoding = seqs.tse_sequence(pp,base_resolution=base_resolution, Ex_FA=Ex_FA, Ref_FA=Ref_FA)

            seq0 = seq.to_mr0()

            if i%5 == 0:
                graph = mr0.compute_graph(seq0, obj_p.build(), 100000, 1e-4)

            signal = mr0.execute_graph(graph, seq0, obj_p.build(), 1e-4, 1e-4)
        # reconstruct image
        space = reconstruction.reconstruction_for_optimazation(signal, encoding, base_resolution, base_resolution)
        image = torch.abs(space) # current optimizer step image


        # ====
        # loss computation
        # ====

        MSE_image = ((image - target)**2).mean()/((target**2).mean())  # MSE of images
        rSAR = torch.sum(Ref_FA**2)/torch.sum(Ref_FA_target**2)     # relative SAR

        loss = lambda_SAR * rSAR + lambda_image * MSE_image

        print(f"{i+1} / {iterations}: loss={loss.item()}, rSAR={rSAR}, Ref_FA={Ref_FA.detach().numpy()}")

        loss_hist.append(loss.item())
        rSAR_hist.append(rSAR.item())
        Ref_FA_hist.append(Ref_FA.detach().numpy().copy())

        # ====
        # perform optimizer step
        # ====

        loss.backward()
        optimizer.step()

        # plot images
        plot_results_images(target, image)

        # optimization timeline
        plot_optimizer_history(loss_hist, Ref_FA_hist, rSAR_hist)

    visualize_results(pp,target,obj_p, loss_hist, Ref_FA_hist, rSAR_hist, Ref_FA_hist,base_resolution = 42,Ex_FA=90)