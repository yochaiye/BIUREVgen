from scene_gen import generate_scenes
from rir_gen import generate_rirs
import numpy as np
import pathlib
from scipy.signal import lfilter
import soundfile as sf
import argparse


T60 = [0.2, 0.4, 0.7, 1]	    # Time for the RIR to reach 60dB of attenuation [s]
mics_num = 8


def save_wavs(rev_signals, file, speaker_out_dir, fs):
    for j, sig in enumerate(rev_signals, 1):
        out_file_name = file.stem + '_ch{}'.format(j) + file.suffix
        out_file_path = speaker_out_dir / out_file_name
        sf.write(out_file_path, sig, fs)


def round_numbers(list_in):
    return list(map(lambda x: round(x, 4), list_in))


def write_scene_to_file(scenes, file_name):
    with open (file_name, 'w') as f:
        for scene in scenes:
            f.write(f"Room size: {round_numbers(scene['room_dim'])}\n")
            f.write(f"Critical distance: {round(scene['critic_dist'], 4)}\n")
            pos = round_numbers(scene['src_pos'][0])
            f.write(f"Source position: {pos}\n")
            mics_num = len(scene['mic_pos'])
            for i in range(mics_num):
                pos = round_numbers(scene['mic_pos'][i])
                dist = round(scene['dists'][i], 4)
                f.write(f"Mic{i} pos\t: {pos}, dist:{dist}\n")
            f.write('\n\n\n')
            f.flush()
    return


def generate_rev_speech(args, scene_type, clean_speech_dir, save_rev_speech_dir, snr=20):
    """
    :param args: From Parser
    :param scene_type: 'near' / 'far' / 'random' / 'winning_ticket'
    :param clean_speech_dir: Directory where the clean speech files are stores
    :param save_rev_speech_dir: Directory to save the generated files
    :param snr: SNR for BIUREV-N
    :return:
    """
    # Get all sub directories (speakers)
    speakers_dir_clean = sorted([e for e in clean_speech_dir.iterdir() if e.is_dir()])

    # Generate reverberant speech files
    scene_agg = []
    for i, speaker in enumerate(speakers_dir_clean, 1):

        # Get all clean WAV files from the speaker's dir
        speaker_files_clean = sorted([e for e in speaker.iterdir() if e.is_file()])
        files_num = len(speaker_files_clean)
        for k, file in enumerate(speaker_files_clean, 1):
            print('Processing %s/%s. Speaker: %s (%d/%d), file %d/%d' % (args.split, scene_type, speaker.name, i, len(speakers_dir_clean), k, files_num))

            # Read a clean file
            s, fs = sf.read(file)

            # Pick a random T60 value
            np.random.shuffle(T60)
            T60_scene = T60[0]

            # Generate a scene
            scene = generate_scenes(1, scene_type, T60_scene)[0]
            if scene_type == 'test':
                scene_agg.append(scene)

            # Generate RIRs
            RIRs = generate_rirs(scene['room_dim'], scene['src_pos'], scene['mic_pos'], T60_scene, fs)

            # Generate reverberant speech
            rev_signals = np.zeros([mics_num, len(s)])
            for j, rir in enumerate(RIRs[0]):
                rev_signals[j] = lfilter(rir, 1, s)
            if args.dataset == 'BIUREV' or args.dataset == 'both':
                rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1   # Normalise so amplitude doesn't clip when saved to disk

                # Create a directory in which the reverberant speech will be saved
                speaker_out_dir = save_rev_speech_dir / 'BIUREV' / args.split / scene_type / speaker.name
                speaker_out_dir.mkdir(parents=True, exist_ok=True)

                save_wavs(rev_signals, file, speaker_out_dir, fs)               # Save signals to disk

            # BIUREV-N -- Add noise
            if args.dataset == 'BIUREV-N' or args.dataset == 'both':
                furthest_mic = np.argmax(scene['dists'])
                var_furthest = np.var(rev_signals[furthest_mic])
                g = np.sqrt(10 ** (-snr / 10) * var_furthest)
                noise = g*np.random.randn(mics_num, len(s))

                noise = lfilter([1], np.array([1, -0.9]), noise, axis=-1)   # Low-pass filter the noise, so it overlaps speech frequncies --> more accurate SNR definition
                rev_signals = rev_signals + noise
                rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1   # Normalise so amplitude doesn't clip when saved to disk

                # Create a directory in which the reverberant speech will be saved
                speaker_out_dir = save_rev_speech_dir / 'BIUREV-N' / args.split / scene_type / speaker.name
                speaker_out_dir.mkdir(parents=True, exist_ok=True)
                save_wavs(rev_signals, file, speaker_out_dir, fs)               # Save signals to disk

    return scene_agg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample."
    )
    parser.add_argument("--split", choices=['train', 'val', 'test'], default='val', help="Generate training, val or test")
    parser.add_argument("--dataset", choices=['BIUREV', 'BIUREV-N', 'both'], default='both', help="Generate BIUREV/BIUREV-N/both")
    args = parser.parse_args()

    #################################### Generate training data ####################################

    save_rev_speech_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/')
    if args.split == 'train':
        clean_speech_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_tr/data/cln_train/')
        scene_type = 'random'
        generate_rev_speech(args, scene_type, clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'train_random.txt')

    #################################### Generate validation data ####################################

    elif args.split == 'val':
        clean_speech_dir = pathlib.Path('/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_dt/data/cln_test/')

        scenes = generate_rev_speech(args, 'near', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'val_near.txt')

        generate_rev_speech(args, 'far', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'val_far.txt')

    #################################### Generate test data ####################################
    elif args.split == 'test':
        clean_speech_dir = pathlib.Path(
            '/mnt/dsi_vol1/users/yochai_yemini/REVERB/SimData/REVERB_WSJCAM0_et/data/cln_test/')

        # scenes = generate_rev_speech(args, 'near', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_near.txt')

        scenes = generate_rev_speech(args, 'far', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_far.txt')

        scenes = generate_rev_speech(args, 'random', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_random.txt')

        scenes = generate_rev_speech(args, 'winning_ticket', clean_speech_dir, save_rev_speech_dir)
        # write_scene_to_file(scenes, 'test_winning_ticket.txt')