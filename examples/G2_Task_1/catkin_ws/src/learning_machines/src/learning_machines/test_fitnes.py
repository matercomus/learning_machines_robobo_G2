irs = [6.436337702191542, 6.437739922634757, 53.069459218725825, 54.55217626678169, 5.844646195122805, 42.43891238252091, 57.773598058701836, 16.92019561993679]
speed_L, speed_R = (15.192315874138586, -78.8872176584988)

def reward_function_2(speed_L, speed_R, irs: list, sensor_max=200):
    s_trans = abs(speed_L) + abs(speed_R)
    if speed_L * speed_R < 0:
        s_rot = abs(abs(speed_L) - abs(speed_R)) / max(abs(speed_L), abs(speed_R))
    else:
        s_rot = 0

    if max(irs) >= sensor_max:
        v_sens = 1
    else:
        v_sens = (max(irs) - min(irs)) / (sensor_max - min(irs))
    
    print(s_trans, s_rot, v_sens)

    return s_trans * (1 - s_rot) * (1 - v_sens)

print(reward_function_2(speed_L, speed_R, irs))