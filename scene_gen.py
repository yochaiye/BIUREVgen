#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 17:32:14 2021

@author: yochai_yemini
"""

import numpy as np
# from plot_room import plot_room


def critical_distance(V, T60):
    return 0.057*np.sqrt(V/T60)


def mic_source_dist_range(room_size, T60, scene_type):
    if scene_type == 'near':
        min_dist = 0.2
        max_dist = critical_distance(np.prod(room_size), T60)
    elif (scene_type == 'far') or (scene_type == 'winning_ticket'):
        min_dist = 2*critical_distance(np.prod(room_size), T60)
        max_dist = 3
        if (min_dist >= max_dist):      # make sure that min_dist <= max_dist
            min_dist = 1.5*critical_distance(np.prod(room_size), T60)
    elif scene_type == 'random':
        min_dist = 0.2
        max_dist = 3
    else:
        raise ValueError("scene_type must be one of {'near', 'far', 'random', 'winning_ticket'}")
    return min_dist, max_dist

# General settings
# train_num = 7861
# val_num = 742
# test_num = 1088
mics_num = 8

# Room's size
room_len_x_min = 4
room_len_x_max = 7
aspect_ratio_min = 1
aspect_ratio_max = 1.5
room_len_z = 2.7

# Margin distance between the source/mics to the walls
margin = 0.5


def generate_scenes(scenes_num, scene_type, T60):
    """
    Generates random rooms with random source and microphones positions
    :param scenes_num: How many scenes (=rooms with source-microphones setups) to generate.
    :param scene_type: 'near', 'far', 'winning_ticket' or 'random'.
    :return: A dictionary with the room size and source and microphone positions.
    """
    src_mics_pos = []
    for i in range(scenes_num):

        # Draw the room's dimensions
        room_len_x = np.random.uniform(room_len_x_min, room_len_x_max)
        aspect_ratio = np.random.uniform(aspect_ratio_min, aspect_ratio_max)
        room_len_y = room_len_x * aspect_ratio
        room_dim = [*np.random.permutation([room_len_x, room_len_y]), room_len_z]

        # Desired range of distance between the source and each microphone
        src_to_mic_dist_min, src_to_mic_dist_max = mic_source_dist_range(room_dim, T60, scene_type)

        # Draw a source position
        src_x = np.random.uniform(margin, room_dim[0]-margin)
        src_y = np.random.uniform(margin, room_dim[1]-margin)
        src_z = 1.75
        src_pos = [src_x, src_y, src_z]

        # Microphones position
        mics_pos_agg = []
        dists = []
        critic_dist = critical_distance(np.prod(room_dim), T60)
        for mic in range(mics_num):
            while True:
                # Draw a microphone position
                mic_x = np.random.uniform(margin, room_dim[0]-margin)
                mic_y = np.random.uniform(margin, room_dim[1]-margin)
                mic_z = 1.6
                mic_pos = [mic_x, mic_y, mic_z]

                # Check if the distance between the mic and the source is in the desired range
                src_mic_dist = np.linalg.norm(np.array(mic_pos)-np.array(src_pos), 2)
                if scene_type == 'winning_ticket':
                    if (mic == 0) and (0.2 <= src_mic_dist <= critic_dist):
                        break
                    elif (mic != 0) and (src_to_mic_dist_min <= src_mic_dist <= src_to_mic_dist_max):
                        break
                elif src_to_mic_dist_min <= src_mic_dist <= src_to_mic_dist_max:
                    break
            mics_pos_agg.append(mic_pos)
            dists.append(src_mic_dist)
        # print(dists)
        # plot_room(room_dim, [src_pos], mics_pos_agg)
        src_mics_pos.append({'room_dim': room_dim, 'src_pos': np.array([src_pos]), 'mic_pos': np.array(mics_pos_agg),
                             'critic_dist': critic_dist, 'dists': dists})
        return src_mics_pos
        