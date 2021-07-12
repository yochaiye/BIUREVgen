import numpy as np
import gpuRIR


def generate_rirs(room_sz, pos_src, pos_rcv, T60, fs):
    # print(room_sz, pos_src, pos_rcv, sep='\n')
    mic_pattern = "omni"  # Receiver polar pattern
    beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # Reflection coefficients
    Tmax = T60 * 0.8  # Time to stop the simulation [s]
    Tdiff = 0.1 # ISM stops here and the gaussian noise with an exponential envelope starts
    nb_img = gpuRIR.t2n(T60, room_sz)  # Number of image sources in each dimension
    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, mic_pattern=mic_pattern)
    # RIRs = np.random.randn(1, len(pos_rcv), int(np.round(Tmax*fs)))
    return RIRs