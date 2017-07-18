import numpy as np
import matplotlib.pyplot as plt

def extract_int(line):
    numbers = [int(s) for s in line.rstrip().split("=") if s.isdigit()]
    return numbers[0]

def parse_siddon_file(path):
    start_line = "start_matrix_chunk"
    stop_line = "end_matrix_chunk"
    time_line = "total"
    matrix_line = "matrix_element_points"
    bp_time_line = "backprojection"
    viewgram_time_line = "viewgram"
    total_bw_time_line = "bw_total"
    total_fw_time_line = "fw_total"

    active = False
    point_count = []
    matrix_count = []
    time = []
    bp_time = []
    viewgram_time = []
    total_fw_time = []
    total_bw_time = []
    
    point_count_cur = 0
    matrix_count_cur = 0

    with open(path, "r") as incoming:
        for line in incoming:
            if(line.startswith(start_line) and not(active)):
                active = True
                point_count_cur = 0
                matrix_count_cur = 0

            if(active and line.startswith(matrix_line)):
                point_count_cur += extract_int(line)
                matrix_count_cur += 1

            if(active and line.startswith(stop_line)):
                active = False;
                point_count.append(point_count_cur)
                matrix_count.append(matrix_count_cur)

            if(line.startswith(bp_time_line)):
                bp_time.append(extract_int(line))

            if(line.startswith(time_line)):
                time.append(extract_int(line))

            if(line.startswith(viewgram_time_line)):
                viewgram_time.append(extract_int(line))

            if(line.startswith(total_fw_time_line)):
                total_fw_time.append(extract_int(line))

            if(line.startswith(total_bw_time_line)):
                total_bw_time.append(extract_int(line))
            
    return point_count, matrix_count, time, bp_time, viewgram_time, total_fw_time, total_bw_time

def parse_tf_file(path):
    start_line = "TF available in forward projector"
    stop_line = "end of forward projection"
    
    total_time_line = "total"
    GPU_time_line = "GPU"
    copyHL_time_line = "copyHL"
    copyLL_time_line = "copyLL"
    scheduling_time_line = "scheduling"
    ray_tracing_time_line = "ray_tracing"
    fp_time_line = "forwardprojection"
    bp_time_line = "backprojection"
    viewgram_time_line = "viewgram"

    total_bw_time_line = "bw_total"
    total_fw_time_line = "fw_total"

    total_points_line = "number points"
    matrix_elements_line = "matrix_element_count"

    active = False

    total_time = []
    GPU_time = []
    copyHL_time = []
    copyLL_time = []
    scheduling_time = []
    rt_time = []
    fp_time = []
    bp_time = []
    viewgram_time = []
    total_points = []
    matrix_elements = []

    total_fw_time = []
    total_bw_time = []

    with open(path, "r") as incoming:
        for line in incoming:
            if(line.startswith(start_line)):
                active = True

            if(line.startswith(stop_line)):
                active = False

            if(active and line.startswith(total_time_line)):
                total_time.append(extract_int(line))

            if(active and line.startswith(GPU_time_line)):
                GPU_time.append(extract_int(line))

            if(active and line.startswith(copyHL_time_line)):
                copyHL_time.append(extract_int(line))

            if(active and line.startswith(copyLL_time_line)):
                copyLL_time.append(extract_int(line))

            if(active and line.startswith(scheduling_time_line)):
                scheduling_time.append(extract_int(line))

            if(active and line.startswith(ray_tracing_time_line)):
                rt_time.append(extract_int(line))

            if(active and line.startswith(fp_time_line)):
                fp_time.append(extract_int(line))

            if(active and line.startswith(total_points_line)):
                total_points.append(extract_int(line))

            if(active and line.startswith(matrix_elements_line)):
                matrix_elements.append(extract_int(line))

            if(line.startswith(bp_time_line)):
                bp_time.append(extract_int(line))

            if(line.startswith(viewgram_time_line)):
                viewgram_time.append(extract_int(line))

            if(line.startswith(total_fw_time_line)):
                total_fw_time.append(extract_int(line))

            if(line.startswith(total_bw_time_line)):
                total_bw_time.append(extract_int(line))

    return total_time, GPU_time, copyHL_time, copyLL_time, scheduling_time, rt_time, fp_time, total_points, matrix_elements, bp_time, viewgram_time, total_fw_time, total_bw_time

print("parsing Siddon output")
# main program
point_count, matrix_count, time, bp_time, viewgram_time, total_fw_time, total_bw_time = parse_siddon_file("/home/pwindisc/stir_examples/examples_derenzo/OSMAPOSL/matrix_siddon_6_iterations.log")

print("Siddon statistics")
mpps_siddon = np.divide(point_count, time)
print("mean # points per millisecond = {0}".format(np.mean(mpps_siddon)))
print("most recent # points per millisecond = {0}".format(mpps_siddon[1]))

print("mean time per viewgram = {0}".format(np.mean(viewgram_time)))
print("percentage spent in FW+BW projection = {0}".format(float(total_fw_time[-1] + total_bw_time[-1]) / viewgram_time[-1]))
print("percentage spent in FW projection = {0}".format(float(total_fw_time[-1]) / viewgram_time[-1]))


"""
print(point_count)
print(matrix_count)
print(time)
"""
print("----------------------------------------")

print("parsing TF output")
total_time, GPU_time, copyHL_time, copyLL_time, scheduling_time, rt_time, fp_time, total_points, matrix_elements, bp_time, viewgram_time, total_fw_time, total_bw_time = parse_tf_file("/home/pwindisc/stir_examples/examples_derenzo/OSMAPOSL/matrix_tf_6_iterations.log")

print("TF statistics")
mpps_tf = np.divide(total_points, total_time)
print("mean # points per millisecond = {0}".format(np.mean(mpps_tf)))
print("most recent # points per millisecond = {0}".format(mpps_tf[-1]))

print("mean time per viewgram = {0}".format(np.mean(viewgram_time)))
print("percentage spent in FW+BW projection = {0}".format(float(total_fw_time[-1] + total_bw_time[-1]) / viewgram_time[-1]))
print("mean percentage spent in FW projection = {0}".format(float(total_fw_time[-1]) / viewgram_time[-1]))


print("----------------------------------------")

print("mean speedup = {0}".format(np.mean(mpps_tf) / np.mean(mpps_siddon)))
print("momentary speedup = {0}".format(float(mpps_tf[-1]) / mpps_siddon[-1]))

# draw a pie chart that shows the time distribution within the forward projection step 
total_time_mean = np.mean(total_time)
GPU_time_mean = np.mean(GPU_time)
IO_time_mean = np.mean(copyHL_time) + np.mean(copyLL_time)
scheduling_time_mean = np.mean(scheduling_time)
fp_time_mean = np.mean(fp_time)
rest_time_mean = total_time_mean - GPU_time_mean - IO_time_mean - scheduling_time_mean - fp_time_mean

fig, ax = plt.subplots()
labels = 'GPU', 'I/O', 'point scheduling', 'forward projection', 'rest'
sizes = [GPU_time_mean, IO_time_mean, scheduling_time_mean, fp_time_mean, rest_time_mean]
explode = (0.1, 0, 0, 0, 0)
ax.pie(sizes, explode = explode, labels = labels, shadow = True, startangle = 90)
ax.axis('equal')
# plt.show()
fig.savefig('forward_projection_time_distribution.pdf')

# draw the time evolution of the speedup and the number of points per chunk
mpps_tf = mpps_tf.astype(float)
mpps_siddon = mpps_siddon.astype(float)
clength = np.minimum(np.size(mpps_tf), np.size(mpps_siddon)) - 1

fig2, ax2 = plt.subplots()
ax2.plot(np.divide(mpps_tf[0:clength], mpps_siddon[0:clength]), 'b')
ax2.set_xlabel("chunk")
ax2.tick_params('y', colors = 'b')
ax2.set_ylabel("speedup", color = 'b')

ax3 = ax2.twinx()
ax3.plot(point_count, 'r')
ax3.set_ylabel("# points per chunk", color = 'r')
ax3.tick_params('y', colors = 'r')
fig2.tight_layout()

ax2.set_xlim([0, 2000])
ax3.set_xlim([0, 2000])

plt.show()
fig2.savefig('speedup.pdf')

"""
# draw the number of points
fig3, ax3 = plt.subplots()
#ax3.plot(point_count)
ax3.plot(matrix_elements)
ax3.set_xlabel("chunk")
ax3.set_ylabel("matrix elements per chunk")
plt.show()
#fig3.savefig('speedup.pdf')
"""

"""
print(len(total_time))
print(len(GPU_time))
print(len(copyHL_time))
print(len(copyLL_time))
print(len(scheduling_time))
print(len(rt_time))
print(len(fp_time))
print(len(total_points))
print(len(matrix_elements))
"""
