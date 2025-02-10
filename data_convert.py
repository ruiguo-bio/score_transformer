import pretty_midi

import itertools
from preprocessing_0611 import event_2midi,midi_2event

import re

import os
from vocab_0611 import *
import sys

V0=120
V1=100
V2=60

step_token = [f'e_{num}' for num in range(16)]

duration_single = [f'n_{num}' for num in range(1,33)]

duration_to_time = {'whole':4,'half':2,'quarter':1,'eighth':0.5,'sixteenth':0.25}


def get_note_duration_dict(beat_duration,curr_time_signature):
    duration_name_to_time = {}
    if curr_time_signature[1] == 4:
        # 4/4, 2/4, 3/4
        quarter_note_duration = beat_duration
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3
        if curr_time_signature[0] >= 4:
            whole_note_duration = 4 * quarter_note_duration
        bar_duration = curr_time_signature[0] * quarter_note_duration

    else:
        # 6/8

        quarter_note_duration = beat_duration / 3 * 2
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3

        bar_duration = curr_time_signature[0] * eighth_note_duration

    duration_name_to_time['half'] = half_note_duration
    duration_name_to_time['quarter'] = quarter_note_duration
    duration_name_to_time['eighth'] = eighth_note_duration
    duration_name_to_time['sixteenth'] = sixteenth_note_duration

    basic_names = duration_name_to_time.keys()
    name_pairs = itertools.combinations(basic_names, 2)
    name_triple = itertools.combinations(basic_names, 3)
    name_quadruple = itertools.combinations(basic_names, 4)

    for name1,name2 in name_pairs:
        duration_name_to_time[name1+'_'+name2] = duration_name_to_time[name1] + duration_name_to_time[name2]

    for name1, name2,name3 in name_triple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3] = duration_name_to_time[name1] + duration_name_to_time[name2] + duration_name_to_time[name3]

    for name1, name2, name3, name4 in name_quadruple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3 + '_' + name4] = duration_name_to_time[name1] + duration_name_to_time[
            name2] + duration_name_to_time[name3] + duration_name_to_time[name4]


    duration_name_to_time['zero'] = 0

    # duration_name_to_time['quarter_triplets'] = quarter_triplets_duration
    # duration_name_to_time['eighth_triplets'] = eighth_triplets_duration
    # duration_name_to_time['sixteenth_triplets'] = sixteenth_triplets_duration

    if curr_time_signature[0] >= 4 and curr_time_signature[1] == 4:
        duration_name_to_time['whole'] = whole_note_duration

    duration_time_to_name = {v: k for k, v in duration_name_to_time.items()}

    duration_times = np.sort(np.array(list(duration_time_to_name.keys())))
    return duration_name_to_time,duration_time_to_name,duration_times,bar_duration


def get_duration(beat_duration,curr_time_signature):
    duration_name_to_time = {}
    if curr_time_signature[1] == 4:
        # 4/4, 2/4, 3/4
        quarter_note_duration = beat_duration
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3
        if curr_time_signature[0] >= 4:
            whole_note_duration = 4 * quarter_note_duration
        bar_duration = curr_time_signature[0] * quarter_note_duration

    else:
        # 6/8

        quarter_note_duration = beat_duration
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4

        bar_duration = curr_time_signature[0] * eighth_note_duration

    duration_name_to_time['half'] = half_note_duration
    duration_name_to_time['quarter'] = quarter_note_duration
    duration_name_to_time['eighth'] = eighth_note_duration
    duration_name_to_time['sixteenth'] = sixteenth_note_duration

    basic_names = duration_name_to_time.keys()
    name_pairs = itertools.combinations(basic_names, 2)
    name_triple = itertools.combinations(basic_names, 3)
    name_quadruple = itertools.combinations(basic_names, 4)

    for name1,name2 in name_pairs:
        duration_name_to_time[name1+'_'+name2] = duration_name_to_time[name1] + duration_name_to_time[name2]

    for name1, name2,name3 in name_triple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3] = duration_name_to_time[name1] + duration_name_to_time[name2] + duration_name_to_time[name3]

    for name1, name2, name3, name4 in name_quadruple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3 + '_' + name4] = duration_name_to_time[name1] + duration_name_to_time[
            name2] + duration_name_to_time[name3] + duration_name_to_time[name4]


    duration_name_to_time['zero'] = 0

    # duration_name_to_time['quarter_triplets'] = quarter_triplets_duration
    # duration_name_to_time['eighth_triplets'] = eighth_triplets_duration
    # duration_name_to_time['sixteenth_triplets'] = sixteenth_triplets_duration

    if curr_time_signature[0] >= 4 and curr_time_signature[1] == 4:
        duration_name_to_time['whole'] = whole_note_duration

    duration_time_to_name = {v: k for k, v in duration_name_to_time.items()}

    duration_times = np.sort(np.array(list(duration_time_to_name.keys())))
    return duration_name_to_time,duration_time_to_name,duration_times,bar_duration



def time2durations(note_duration, duration_time_to_name, duration_times):

    duration_index = np.argmin(np.abs(note_duration - duration_times))
    duration_name = duration_time_to_name[duration_times[duration_index]]
    if duration_name == 'zero':
        return []

    duration_elements = duration_name.split('_')
    return duration_elements


def add_duration(duration_list,current_step):
    total_duration = 0
    for duration in duration_list:
        total_duration += duration_to_time[duration]
    total_duration = int(total_duration*4)
    last_time = int(current_step[2:])
    current_step = f'e_{last_time + total_duration}'
    if total_duration > 32:
        print(f'total duration {total_duration}')

    total_duration = f'n_{total_duration}'
    return total_duration, current_step


def rest_multi_step_single(events, remove_rest=True,remove_continue=True):

    in_duration = False
    in_pitch = False
    is_rest_s = False

    new_event = []
    continue_list = []
    is_continue = False
    current_step = 'e_0'
    pitch_list = []
    curr_track = ''

    r = re.compile('track_\d')

    track_program = list(set(filter(r.match, events)))
    track_nums = len(track_program)
    bar_num = 0


    duration_list = []

    previous_step = 'e_0'
    for idx,event in enumerate(events):

        # print(idx,event)
        if event == 'bar':
            bar_num += 1
        # if idx == 228:
        #     print('here')

        if event not in durations and in_duration:

            if is_rest_s and new_event[-1] in pitch_tokens:
                total_duration, _ = add_duration(duration_list, current_step)
            elif is_rest_s and is_continue:
                total_duration, current_step = add_duration(duration_list, previous_step)
            elif is_rest_s and new_event[-1] in duration_single + track_num:
                current_step = previous_step
                total_duration, current_step = add_duration(duration_list, current_step)
            else:
                previous_step = current_step
                total_duration, current_step = add_duration(duration_list, current_step)

            is_rest_s = False

            in_duration = False
            duration_list = []
            if in_pitch:
                if int(total_duration[2:]) > 32:
                    print(f'total duration is {total_duration}')
                new_event.append(total_duration)
                in_pitch = False
            if is_continue:
                track_pos = np.where(np.array(new_event) == curr_track)[0][-2]
                bar_pos = np.where(np.array(new_event) == 'bar')[0][-1]
                if curr_track == 'track_0':
                    if track_nums > 1:
                        next_track_pos = np.where(np.array(new_event) == 'track_1')[0][-1]
                    else:
                        next_track_pos = bar_pos
                elif curr_track == 'track_1':
                    if track_nums > 2:
                        next_track_pos = np.where(np.array(new_event) == 'track_2')[0][-1]
                    else:
                        next_track_pos = bar_pos
                else:
                    next_track_pos = bar_pos

                for pitch in pitch_list:
                    if len(np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0]) > 0:

                        pitch_pos = track_pos + np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0][-1]

                        for token in new_event[pitch_pos + 1:]:
                            if token in duration_single:
                                break
                        old_duration = token


                        for token in new_event[pitch_pos - 1:track_pos:-1]:
                            if token in step_token:
                                break
                        old_step = token


                        if new_event[pitch_pos-1] in step_token:
                            if new_event[pitch_pos+1] in duration_single:
                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event[pitch_pos+1] = new_duration
                            else:

                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                new_event.insert(pitch_pos + 1, old_step)
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event.insert(pitch_pos+1,new_duration)
                                next_track_pos += 2

                        else:
                            new_event.insert(pitch_pos, old_step)
                            new_event.insert(pitch_pos, old_duration)
                            next_track_pos += 2
                            if new_event[pitch_pos + 3] in duration_single:
                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event[pitch_pos + 3] = new_duration
                            else:

                                new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                                if int(new_duration[2:]) > 32:
                                     print(f'new duration is {new_duration}')
                                new_event.insert(pitch_pos + 3, old_step)
                                new_event.insert(pitch_pos + 3, new_duration)
                                next_track_pos += 2

                        pop_list = []
                        total_break = False
                        for pos in range(track_pos,next_track_pos):
                            if total_break:
                                break
                            if new_event[pos] in step_token:
                                for duration_pos in range(pos+1,next_track_pos):
                                    if new_event[duration_pos] in duration_single:
                                        this_duration = new_event[duration_pos]
                                        break

                                for next_pos in range(pos+1,next_track_pos):
                                    if total_break:
                                        break
                                    if new_event[next_pos] in step_token:
                                        if new_event[next_pos] == new_event[pos]:
                                            for next_duration_pos in range(next_pos + 1, next_track_pos):
                                                if new_event[next_duration_pos] in duration_single:
                                                    next_duration = new_event[next_duration_pos]
                                                    if next_duration == this_duration:
                                                        if next_pos-1 != duration_pos:
                                                            continue

                                                            for move_pitch_pos in range(next_duration_pos-1,next_pos,-1):
                                                                new_event.insert(duration_pos, new_event[move_pitch_pos])
                                                                del new_event[move_pitch_pos+1]
                                                            pop_list.append(next_pos+1)
                                                            pop_list.append(next_duration_pos)
                                                            total_break = True
                                                            break


                                                        else:
                                                            pop_list.append(duration_pos)
                                                            pop_list.append(next_pos)
                                                    break
                        if len(pop_list):
                            for pop_pos in pop_list[::-1]:
                                del new_event[pop_pos]
                            next_track_pos -= len(pop_list)
                            pop_list = []




                is_continue = False
                pitch_list = []

        if event in rests:
            if event == 'rest_s':
                # if in_duration:
                # #clear duration if already in rest_s
                #
                #     if is_rest_s:
                #         total_duration, _ = add_duration(duration_list, current_step)
                #         in_duration = False
                #         duration_list = []
                #         if in_pitch:
                #             new_event.append(total_duration)
                #             in_pitch = False
                #         if is_continue:
                #             track_pos = np.where(np.array(new_event) == curr_track)[0][-2]
                #
                #             for pitch in pitch_list:
                #                 pitch_pos = track_pos + np.where(np.array(new_event[track_pos:]) == pitch)[0][0]
                #                 if new_event[pitch_pos - 1] in step_token:
                #                     if new_event[pitch_pos + 1] in duration_single:
                #                         new_event[pitch_pos + 1] = 'n_' + str(
                #                             int(new_event[pitch_pos + 1][2:]) + int(total_duration[2:]))

                is_rest_s = True

            if remove_rest:
                pass
            else:
                new_event.append(event)
            continue

        if event in track_num:
            current_step = 'e_0'
            previous_step = 'e_0'
            duration_list = []
            pitch_list = []
            in_duration = False
            in_pitch = False
            is_rest_s = False
            is_continue = False
            new_event.append(event)
            curr_track = event
            continue

        if event in pitch_tokens:
            if is_continue:
                pitch_list.append(event)
            else:
                if not in_pitch:
                    if is_rest_s:
                        if int(previous_step[2:]) > 15:
                            print(f'previous step is {previous_step}')
                        new_event.append(previous_step)
                        current_step = previous_step
                        is_rest_s = False
                    else:
                        if int(current_step[2:]) > 15:
                            print(f'current step is {current_step}')
                        new_event.append(current_step)
                    in_pitch = True
                new_event.append(event)

            continue

        if event in durations:
            duration_list.append(event)
            in_duration = True
            continue

        if event == 'continue':
            if bar_num > 1:
                is_continue = True
            continue



            # print(bar_num)
            # if bar_num == 90:
            #     print("here")

        new_event.append(event)


    else:
        if is_rest_s and new_event[-1] in pitch_tokens:
            total_duration, _ = add_duration(duration_list, current_step)
        elif is_rest_s and is_continue:
            total_duration, current_step = add_duration(duration_list, previous_step)
        elif is_rest_s and new_event[-1] in duration_single + track_num:
            current_step = previous_step
            total_duration, current_step = add_duration(duration_list, current_step)
        else:
            previous_step = current_step
            total_duration, current_step = add_duration(duration_list, current_step)
        is_rest_s = False

        in_duration = False
        duration_list = []
        if in_pitch:
            new_event.append(total_duration)
            in_pitch = False
        if is_continue:
            track_pos = np.where(np.array(new_event) == curr_track)[0][-2]
            bar_pos = np.where(np.array(new_event) == 'bar')[0][-1]
            if curr_track == 'track_0':
                if track_nums > 1:
                    next_track_pos = np.where(np.array(new_event) == 'track_1')[0][-1]
                else:
                    next_track_pos = bar_pos
            elif curr_track == 'track_1':
                if track_nums > 2:
                    next_track_pos = np.where(np.array(new_event) == 'track_2')[0][-1]
                else:
                    next_track_pos = bar_pos
            else:
                next_track_pos = bar_pos

            for pitch in pitch_list:
                if len(np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0]) > 0:
                    pitch_pos = track_pos + np.where(np.array(new_event[track_pos:next_track_pos]) == pitch)[0][-1]

                    for token in new_event[pitch_pos + 1:]:
                        if token in duration_single:
                            break
                    old_duration = token


                    for token in new_event[pitch_pos - 1:track_pos:-1]:
                        if token in step_token:
                            break
                    old_step = token

                    if new_event[pitch_pos - 1] in step_token:
                        if new_event[pitch_pos + 1] in duration_single:
                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event[pitch_pos + 1] = new_duration
                        else:

                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event.insert(pitch_pos + 1, old_step)
                            new_event.insert(pitch_pos + 1, new_duration)
                            next_track_pos += 2

                    else:
                        new_event.insert(pitch_pos, old_step)
                        new_event.insert(pitch_pos, old_duration)
                        next_track_pos += 2
                        if new_event[pitch_pos + 3] in duration_single:
                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event[pitch_pos + 3] = new_duration
                        else:

                            new_duration = 'n_' + str(int(old_duration[2:]) + int(total_duration[2:]))
                            if int(new_duration[2:]) > 32:
                                print(f'new duration is {new_duration}')
                            new_event.insert(pitch_pos + 3, old_step)
                            new_event.insert(pitch_pos + 3, new_duration)
                            next_track_pos += 2

                    pop_list = []
                    total_break = False
                    for pos in range(track_pos, next_track_pos):
                        if total_break:
                            break
                        if new_event[pos] in step_token:
                            for duration_pos in range(pos + 1, next_track_pos):
                                if new_event[duration_pos] in duration_single:
                                    this_duration = new_event[duration_pos]
                                    break

                            for next_pos in range(pos + 1, next_track_pos):
                                if total_break:
                                    break
                                if new_event[next_pos] in step_token:
                                    if new_event[next_pos] == new_event[pos]:
                                        for next_duration_pos in range(next_pos + 1, next_track_pos):
                                            if new_event[next_duration_pos] in duration_single:
                                                next_duration = new_event[next_duration_pos]
                                                if next_duration == this_duration:
                                                    if next_pos - 1 != duration_pos:
                                                        continue

                                                        for move_pitch_pos in range(next_duration_pos - 1, next_pos, -1):
                                                            new_event.insert(duration_pos, new_event[move_pitch_pos])
                                                            del new_event[move_pitch_pos + 1]

                                                        pop_list.append(next_pos + 1)
                                                        pop_list.append(next_duration_pos)
                                                        total_break = True
                                                        break


                                                    else:
                                                        pop_list.append(duration_pos)
                                                        pop_list.append(next_pos)
                                                break
                    if len(pop_list):
                        for pop_pos in pop_list[::-1]:
                            del new_event[pop_pos]
                        next_track_pos -= len(pop_list)
                        pop_list = []

    # print(events)
    # print(new_event)
    return new_event


def step_multi_rest_multi(events):
    current_time = 0
    new_events = []
    current_step = 0
    bar_time = 0
    track_started = False
    previous_time = 0
    is_rest_s = False
    duration_list = []
    in_duration = False


    for idx,event in enumerate(events):
        # print(idx,event)
        # if idx == 488:
        #     print("here")
        # if len(new_events) == 162:
        #     print('here')


        if event in time_signature_token:
            if event == '4/4':
                bar_time = 16
            elif event == '3/4':
                bar_time = 12
            elif event == '2/4':
                bar_time = 8
            else:
                bar_time = 12

            duration_name_to_time, duration_time_to_name, duration_times, bar_duration = get_duration(
                1, (int(event[0]),int(event[2])))


        if event not in durations and in_duration:
            # if not is_rest_s:
            previous_time = current_time

            total_duration = 0
            for duration in duration_list:
                total_duration += duration_to_time[duration]
            total_duration = int(total_duration * 4)


            current_time = int(current_step[2:]) + total_duration

            is_rest_s = False
            in_duration = False
            duration_list = []


        if event in track_num[1:] + ['bar']:
            if track_started:
                if current_time < bar_time:
                    new_events.append('rest_e')
                    difference = bar_time - current_time
                    duration_elements = time2durations(difference/4, duration_time_to_name, duration_times)
                    new_events.extend(duration_elements)

                current_time = 0
            else:
                track_started = True

        if event == 'track_0':
            current_time = 0
        if event in step_token:
            current_step = event
            if int(event[2:]) < current_time:
                difference = int(event[2:]) - previous_time
                if new_events[-1] == 'continue':
                    new_events.insert(-1,'rest_s')
                else:
                    new_events.append( 'rest_s')
                if difference > 0:
                    duration_elements = time2durations(difference/4, duration_time_to_name, duration_times)
                    new_events.extend(duration_elements)
                current_time = int(event[2:])
                is_rest_s = True
            if int(event[2:]) > current_time:
                difference = int(event[2:]) - current_time
                new_events.append('rest_e')
                duration_elements = time2durations(difference/4, duration_time_to_name, duration_times)
                new_events.extend(duration_elements)
                current_time = int(event[2:])


        if event not in step_token:
            new_events.append(event)

        if event in durations:
            duration_list.append(event)
            in_duration = True

    else:

        if in_duration:

            total_duration = 0
            for duration in duration_list:
                total_duration += duration_to_time[duration]
            total_duration = int(total_duration * 4)

            current_time = int(current_step[2:]) + total_duration
        if current_time < bar_time:
            new_events.append('rest_e')
            difference = bar_time - current_time
            duration_elements = time2durations(difference / 4, duration_time_to_name, duration_times)
            new_events.extend(duration_elements)

    return new_events


def rest_multi_step_multi(events):
    in_duration = False
    in_pitch = False
    is_rest_s = False
    is_rest_e = False

    new_event = []
    current_step = 'e_0'
    bar_num = 0
    r = re.compile('track_\d')

    duration_list = []

    previous_step = 'e_0'
    for idx,event in enumerate(events):
        # print(idx,event)
        # if idx == 162:
        #     print('here')
        # print(len(new_event))

        if event not in durations and in_duration:

            if is_rest_s:
                current_step = previous_step
                _, current_step = add_duration(duration_list, current_step)
            else:
                previous_step = current_step
                _, current_step = add_duration(duration_list, current_step)


            in_duration = False
            duration_list = []

            if is_rest_s:
                is_rest_s = False



        if event in rests:
            if event == 'rest_s':

                is_rest_s = True
            if event == 'rest_e':
                is_rest_e = True

            continue

        if event in track_num:
            current_step = 'e_0'
            previous_step = 'e_0'
            duration_list = []
            in_duration = False
            in_pitch = False
            is_rest_s = False

            new_event.append(event)

            continue

        if event in pitch_tokens:
            is_rest_e = False

            if not in_pitch:
                if is_rest_s:
                    if int(previous_step[2:]) > 15:
                        print(f'previous step is {previous_step}')
                    new_event.append(previous_step)
                    current_step = previous_step
                    is_rest_s = False
                else:
                    if int(current_step[2:]) > 15:
                        print(f'current step is {current_step}')
                    new_event.append(current_step)
                in_pitch = True
            new_event.append(event)
            continue

        if event in durations:

            duration_list.append(event)
            if not is_rest_e and not is_rest_s:
                new_event.append(event)
            in_duration = True
            in_pitch = False
            continue

        if event == 'bar':
            bar_num += 1

            # print(bar_num)
            # if bar_num == 90:
            #     print("here")

        new_event.append(event)

    return new_event


# rest single -> step single
def rest_single_step_single(events):
    in_pitch = False
    is_rest_s = False
    is_rest_e = False

    new_event = []
    current_step = 'e_0'

    bar_num = 0
    r = re.compile('track_\d')

    previous_step = 'e_0'
    for idx, event in enumerate(events):
        # print(idx, event)
        # if idx == 288:
        #     print('here')

        if event in rests:
            if event == 'rest_s':
                is_rest_s = True
            if event == 'rest_e':
                is_rest_e = True
            continue

        if event in track_num:
            current_step = 'e_0'
            previous_step = 'e_0'

            in_pitch = False
            is_rest_s = False
            new_event.append(event)
            continue

        if event in pitch_tokens:
            if not in_pitch:
                if is_rest_s:
                    if int(previous_step[2:]) > 15:
                        print(f'previous step is {previous_step}')
                    new_event.append(previous_step)
                    current_step = previous_step
                    is_rest_s = False
                else:
                    if int(current_step[2:]) > 15:
                        print(f'current step is {current_step}')
                    new_event.append(current_step)
                in_pitch = True

        if event in duration_single:
            if is_rest_s:
                current_step = f'e_{int(previous_step[2:]) + int(event[2:])}'
                in_pitch = False
                is_rest_s = False
                continue
            else:
                previous_step = current_step
                current_step = f'e_{int(event[2:]) + int(current_step[2:])}'
                in_pitch = False

                if is_rest_e:
                    is_rest_e = False
                    continue

        if event == 'bar':
            bar_num += 1

            # print(bar_num)
            # if bar_num == 90:
            #     print("here")

        new_event.append(event)

    return new_event

# step single -> rest single
def step_single_rest_single(events):
    current_time = 0
    new_events = []
    current_step = 0
    bar_time = 0
    track_started = False
    previous_time = 0
    is_rest_s = False

    for idx,event in enumerate(events):
        # print(idx,event)
        # if idx == 32:
        #     print("here")


        if event in time_signature_token:
            if event == '4/4':
                bar_time = 16
            elif event == '3/4':
                bar_time = 12
            elif event == '2/4':
                bar_time = 8
            else:
                bar_time = 12

        if event in track_num[1:] + ['bar']:
            if track_started:
                if current_time < bar_time:
                    new_events.append('rest_e')
                    difference = bar_time - current_time
                    new_events.append(f'n_{difference}')
                current_time = 0
            else:
                track_started = True

        if event == 'track_0':
            current_time = 0
        if event in step_token:
            current_step = event
            if int(event[2:]) < current_time:
                difference = int(event[2:]) - previous_time
                new_events.append('rest_s')
                if difference > 0:
                    new_events.append(f'n_{difference}')
                previous_time = int(event[2:])
                is_rest_s = True
            if int(event[2:]) > current_time:
                difference = int(event[2:]) - current_time
                new_events.append('rest_e')
                new_events.append(f'n_{difference}')
                current_time = int(event[2:])

        if event in duration_single:
            if not is_rest_s:
                previous_time = current_time
            current_time = int(current_step[2:]) + int(event[2:])

            is_rest_s = False
        if event not in step_token:
            new_events.append(event)


    else:
        if current_time < bar_time:
            new_events.append('rest_e')
            difference = bar_time - current_time
            new_events.append(f'n_{difference}')

    return new_events


def get_note_duration_dict(beat_duration,curr_time_signature):
    duration_name_to_time = {}
    if curr_time_signature[1] == 4:
        # 4/4, 2/4, 3/4
        quarter_note_duration = beat_duration
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3
        if curr_time_signature[0] >= 4:
            whole_note_duration = 4 * quarter_note_duration
        bar_duration = curr_time_signature[0] * quarter_note_duration

    else:
        # 6/8

        quarter_note_duration = beat_duration / 3 * 2
        half_note_duration = quarter_note_duration * 2
        eighth_note_duration = quarter_note_duration / 2
        sixteenth_note_duration = quarter_note_duration / 4
        # quarter_triplets_duration = half_note_duration / 3
        # eighth_triplets_duration = quarter_note_duration / 3
        # sixteenth_triplets_duration = eighth_note_duration / 3

        bar_duration = curr_time_signature[0] * eighth_note_duration

    duration_name_to_time['half'] = half_note_duration
    duration_name_to_time['quarter'] = quarter_note_duration
    duration_name_to_time['eighth'] = eighth_note_duration
    duration_name_to_time['sixteenth'] = sixteenth_note_duration

    basic_names = duration_name_to_time.keys()
    name_pairs = itertools.combinations(basic_names, 2)
    name_triple = itertools.combinations(basic_names, 3)
    name_quadruple = itertools.combinations(basic_names, 4)

    for name1,name2 in name_pairs:
        duration_name_to_time[name1+'_'+name2] = duration_name_to_time[name1] + duration_name_to_time[name2]

    for name1, name2,name3 in name_triple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3] = duration_name_to_time[name1] + duration_name_to_time[name2] + duration_name_to_time[name3]

    for name1, name2, name3, name4 in name_quadruple:
        duration_name_to_time[name1 + '_' + name2 + '_' + name3 + '_' + name4] = duration_name_to_time[name1] + duration_name_to_time[
            name2] + duration_name_to_time[name3] + duration_name_to_time[name4]


    duration_name_to_time['zero'] = 0

    # duration_name_to_time['quarter_triplets'] = quarter_triplets_duration
    # duration_name_to_time['eighth_triplets'] = eighth_triplets_duration
    # duration_name_to_time['sixteenth_triplets'] = sixteenth_triplets_duration

    if curr_time_signature[0] >= 4 and curr_time_signature[1] == 4:
        duration_name_to_time['whole'] = whole_note_duration

    duration_time_to_name = {v: k for k, v in duration_name_to_time.items()}

    duration_times = np.sort(np.array(list(duration_time_to_name.keys())))
    return duration_name_to_time,duration_time_to_name,duration_times,bar_duration


def remi_2midi(events):
    
    if events[1][0] == 't':
        # print(events)
        tempo_category = int(events[1][2])
        if tempo_category == len(tempo_bins) - 1:
            tempo = tempo_bins[tempo_category]
        else:
            tempo = (tempo_bins[tempo_category] + tempo_bins[tempo_category + 1]) / 2
    else:
        tempo = float(events[1])
    pm_new = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    numerator = int(events[0].split('/')[0])
    denominator = int(events[0].split('/')[1])
    time_signature = pretty_midi.TimeSignature(numerator, denominator, 0)
    pm_new.time_signature_changes = [time_signature]

    r = re.compile('i_\d')

    programs = list(filter(r.match, events))

    # program_start_pos = np.where(track_0_program == np.array(events))[0][0]
    #
    bar_start_pos = np.where('bar' == np.array(events))[0][0]
    # programs = events[program_start_pos:start_track_pos]

    for track in programs:
        track = pretty_midi.Instrument(program=int(track.split('_')[-1]))
        pm_new.instruments.append(track)

    # add a fake note for duration dict calculation
    pm_new.instruments[0].notes.append(pretty_midi.Note(
        velocity=100, pitch=30, start=0, end=10))
    beats = pm_new.get_beats()
    pm_new.instruments[0].notes.pop()
    duration_name_to_time,duration_time_to_name,duration_times,bar_duration = get_note_duration_dict(
        beats[1] - beats[0], (time_signature.numerator, time_signature.denominator))
    sixteenth_duration = duration_name_to_time['sixteenth']
    curr_time = 0
    bar_num = 0
    bar_start_time = 0
    pitch_list = []
    for idx,event in enumerate(events):
        # print(idx,event)
        if event == 'bar':
            curr_time = bar_num * bar_duration
            bar_start_time = curr_time
            bar_num += 1
        if event in track_num:
            curr_time = bar_start_time
            current_track = event

        if event in step_token:
            curr_time = bar_start_time + int(event[2:]) * sixteenth_duration
        if event in pitch_tokens:
            pitch_list.append(int(event[2:]))


        if event in duration_single:
            end_time = curr_time + (int(event[2:])) * sixteenth_duration
            start_time = curr_time
            for pitch in pitch_list:
                if current_track == 'track_0':
                    vel = V0
                elif current_track == 'track_1':
                    vel = V1
                else:
                    vel = V2
                note = pretty_midi.Note(velocity=vel, pitch=pitch,
                                        start=start_time, end=end_time)
                pm_new.instruments[int(current_track[6])].notes.append(note)
            pitch_list = []


    return pm_new



def remove_first_continue(events):
    in_first_bar = False
    bar_count = 0
    return_events = []
    for event in events:
        if event == 'bar':
            if in_first_bar is False and bar_count == 0:
                in_first_bar = True
                bar_count += 1
            else:
                in_first_bar = False
        if event == 'continue':
            if in_first_bar:
                continue
        return_events.append(event)
    return return_events

def walk(folder_name,suffix):
    files = []
    for p, d, f in os.walk(folder_name):
        for file_name in f:

            if file_name[-len(suffix):].lower() == suffix:
                files.append(os.path.join(p, file_name))
    return files
#

#
# midi_folder = "/home/ruiguo/dataset/chinese"
# event_folder = "/home/data/guorui/dataset/lmd/only_melody_bass_event"


if __name__== "__main__":

    # change this folder
    event_folder = "/home/ruiguo/dataset/pop909_old"
    # # # # #
    file_names = walk(event_folder, 'event')
    total_files = 0
    for file_name in file_names:
        print(file_name)
        rest_multi = pickle.load(open(file_name, 'rb'))
        # rest_multi_midi = event_2midi(rest_multi)[0]
        # rest_multi_midi.write(file_name[:-6] + '.mid')

        # result = midi_2event(file_name,logger)

        # if result:
        #     rest_multi,pm0 = result
        # else:
        #     continue
        # base_name = os.path.basename(file_name)[:-4]
        # pickle.dump(rest_multi,open('/home/ruiguo/dataset/chinese_event' + base_name + '_event','wb'))
        # rest, multi duration (sheet)
        # rest_multi = pickle.load(open(file_name,'rb'))
        #     # new_event = remove_first_continue(events)
        #
        #     # step, single duration, (remi)
        #     np.where(np.array(rest_multi) == 'bar')
        #     rest_multi[203:]
        step_single = rest_multi_step_single(rest_multi)
        remi_midi = remi_2midi(step_single)
        # np.where(np.array(remi_midi) == 'bar')
        # step_single[139:]
        remi_midi.write(os.path.join(event_folder, file_name[:-5] + 'step_single.mid'))

        rest_multi_converted = midi_2event(os.path.join(event_folder, file_name[:-5] + 'step_single.mid'))[0]
        bar_num = 0
        idx = 3
        # while idx < len(rest_multi_converted):
        #     event = rest_multi_converted[idx]
        #     if event == 'bar':
        #         bar_num += 1
        #     if event != rest_multi[idx]:
        #         print(rest_multi[idx-20:idx+100])
        #         print(rest_multi_converted[idx-20:idx + 100])
        #         print(file_name)
        #         idx += 20
        #         break
        #
        #     else:
        #         idx += 1

        # for idx,event in enumerate(step_single[3:]):
        #     if event not in vocab.all_tokens:
        #         logger.info('not in vocab')
        #         logger.info(idx,event)
        # print(step_single)
        # #
        # # rest, single duration
        # rest_single = step_single_rest_single(step_single)
        #
        # # step, single duration
        # step_single_back = rest_single_step_single(rest_single)
        #
        # step_multi = rest_multi_step_multi(rest_multi)
        # rest_multi_back = step_multi_rest_multi(step_multi)
        #
        # assert rest_multi_back == rest_multi
        # assert step_single_back == step_single

        # step, multi duration

        #
        #     # rest_multi = step_multi_rest_multi(step_multi)
        #     # print(events)
        #     # print(step_multi)
        #     # print(rest_multi)
        #     # print('')
        #     # event_folder = os.path.dirname(file_name)
        #     pickle.dump(rest_multi, open(os.path.join(event_folder, os.path.basename(file_name)[:-4] + '_rest_multi'), 'wb'))
        # #     #
        pickle.dump(step_single, open(os.path.join(event_folder, file_name[:-5] + 'step_single'), 'wb'))
        total_files += 1
    print(total_files)
    sys.exit()
    # pickle.dump(rest_single, open(os.path.join(event_folder, file_name[:-5] + 'rest_single'), 'wb'))
    # pickle.dump(step_multi, open(os.path.join(event_folder, file_name[:-5] + 'step_multi'), 'wb'))

    #     #
    #
    #

    #
    #
    #
    # folder_prefix = '/home/data/guorui/'

    # for i in range(3):
    #     original_dataset = pickle.load(open(folder_prefix  + 'score_transformer/sync/' + 'new_training_all_batches_' + str(i), 'rb'))

    # test_dataset = pickle.load(open(folder_prefix + 'score_transformer/sync/' + 'cleaned_continue_test_batches', 'rb'))
    # new_event_list = []
    # for item_list in test_dataset[1000:]:
    #     this_list = []
    #     for events in item_list:
    # new_event = remove_first_continue(events)
    # new_event = event_processing(events)
    # this_list.append(new_event)
    # remi_2midi(new_event,'test1')
    # pm, _ = preprocessing_0611.event_2midi(events)
    # pm.write('test0.mid')

    #     new_event_list.append(this_list)
    # pickle.dump(new_event_list,open(folder_prefix  + 'score_transformer/sync/' + 'remi_test_batches', 'wb'))

    #
    # rest_multi,pm = preprocessing_0611.midi_2event('score.mid')
    # step_single = rest_multi_step_single(rest_multi)
    # rest_single = step_single_rest_single(step_single)
    # step_multi = rest_multi_step_multi(rest_multi)
    # print(rest_multi)
    # print(step_single)
    # print(rest_single)
    # print(step_multi)
    # print('')
    #
