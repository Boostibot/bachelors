import matplotlib.pyplot as plt 
import matplotlib.image
import struct
import numpy as np
import csv 
import math
import sys
import os 

class Map_Set:
    nx = 0
    ny = 0
    dx = 0
    dy = 0
    dt = 0.0
    N = 0
    iter = 0
    maps = {}

def load_bin_map_file(path):
    out = Map_Set()
    with open(path, 'rb') as f:
        data = f.read()
        pos = 0
        def read_i32(pos):
            pos += 4
            return int.from_bytes(data[pos-4:pos], byteorder='little', signed=True), pos

        def read_i64(pos):
            pos += 8
            return int.from_bytes(data[pos-8:pos], byteorder='little', signed=True), pos
        
        def read_f64(pos):
            pos += 8
            return struct.unpack('d', data[pos-8:pos])[0], pos

        def read_2d_f64_arr(nx, ny, pos):
            before = pos
            pos += nx*ny*8
            return np.frombuffer(data[before:pos], dtype=np.float64).reshape(ny, nx), pos

        def read_name(pos):
            pos += 32
            return data[pos-32:pos].split(b'\0')[0].decode("utf-8"), pos

        magic, pos = read_i32(pos)
        map_count, pos = read_i32(pos)
        out.nx, pos = read_i32(pos)
        out.ny, pos = read_i32(pos)
        out.dx, pos = read_f64(pos)
        out.dy, pos = read_f64(pos)
        out.dt, pos = read_f64(pos)
        out.iter, pos = read_i64(pos)
        out.N = out.nx*out.ny
        names = []
        maps = []
        for i in range(map_count):
            name, pos = read_name(pos) 
            names += [name]
        
        for i in range(map_count):
            map, pos = read_2d_f64_arr(out.nx, out.ny, pos) 
            maps += [map]

        print(f"Map file '{path}' read. dims:{out.nx}x{out.ny} size:{map_count*out.N*8/1024/1024}MB maps:{map_count} {names}")
        out.maps = dict(zip(names, maps))
    return out
       
def load_dir_bin_map_file(path, index):
    return load_bin_map_file(path + "/maps_{:04d}.bin".format(index))

class Stats:
    nx = 0
    ny = 0
    N = 0
    dt = 0
    step_res_count = 0
    time = []
    iter = []
    Phi_iters = []
    T_iters = []
    T_delta_L1 = []
    T_delta_L2 = []
    T_delta_Lmax = []
    T_delta_max = []
    T_delta_min = []
    Phi_delta_L1 = []
    Phi_delta_L2 = []
    Phi_delta_Lmax = []
    Phi_delta_max = []
    Phi_delta_min = []
    step_res_L1 = []
    step_res_L2 = []
    step_res_Lmax = []
    step_res_max = []
    step_res_min = []

def load_stat_file(path):
    def sa():
        return [[] for i in range(20)]

    def arr_to_f64_and_pad(arr, to_size):
        out = []
        for a in arr:
            try:
                out += [float(a)]
            except (ValueError, IndexError):
                out += [0.0]
        for i in range(to_size - len(out)):
            out += [0.0]
        return out

    print(path)
    stats = Stats()
    stats.step_res_L1 = sa()
    stats.step_res_L2 = sa()
    stats.step_res_Lmax = sa()
    stats.step_res_max = sa()
    stats.step_res_min = sa()
    with open(path,'r') as csvfile: 
        base_fields = 12
        rows = csv.reader(csvfile, delimiter = ',') 
        for i, row in enumerate(rows): 
            if i == 0:
                row_f64 = arr_to_f64_and_pad(row, 3)
                stats.nx = row_f64[0]
                stats.ny = row_f64[1]
                stats.dt = row_f64[2]
                stats.N = stats.nx*stats.ny
            if i > 1:
                step_residuals = max(math.ceil((len(row) - base_fields) / 4), 0)
                fields = base_fields + step_residuals*4
                stats.step_res_count = max(stats.step_res_count, step_residuals)

                row_f64 = arr_to_f64_and_pad(row, fields)
                stats.time += [row_f64[0]]
                stats.iter += [row_f64[1]]
                stats.Phi_iters += [row_f64[2]]
                stats.T_iters += [row_f64[3]]

                stats.T_delta_L1 += [row_f64[4]]
                stats.T_delta_L2 += [row_f64[5]]
                stats.T_delta_max += [row_f64[6]]
                stats.T_delta_min += [row_f64[7]]

                stats.Phi_delta_L1 += [row_f64[8]]
                stats.Phi_delta_L2 += [row_f64[9]]
                stats.Phi_delta_max += [row_f64[10]]
                stats.Phi_delta_min += [row_f64[11]]

                step_res_fields = row_f64[12:]
                for i in range(step_residuals):
                    stats.step_res_L1[i] += [step_res_fields[i*4 + 0]]
                    stats.step_res_L2[i] += [step_res_fields[i*4 + 1]]
                    stats.step_res_max[i] += [step_res_fields[i*4 + 2]]
                    stats.step_res_min[i] += [step_res_fields[i*4 + 3]]

    stats.time = np.array(stats.time)
    stats.iter = np.array(stats.iter)
    stats.Phi_iters = np.array(stats.Phi_iters)
    stats.T_iters = np.array(stats.T_iters)
    stats.T_delta_L1 = np.array(stats.T_delta_L1)
    stats.T_delta_L2 = np.array(stats.T_delta_L2)
    stats.T_delta_max = np.array(stats.T_delta_max)
    stats.T_delta_min = np.array(stats.T_delta_min)
    stats.Phi_delta_L1 = np.array(stats.Phi_delta_L1)
    stats.Phi_delta_L2 = np.array(stats.Phi_delta_L2)
    stats.Phi_delta_max = np.array(stats.Phi_delta_max)
    stats.Phi_delta_min = np.array(stats.Phi_delta_min)

    stats.T_delta_Lmax = np.maximum(np.abs(stats.T_delta_max), np.abs(stats.T_delta_min))
    stats.Phi_delta_Lmax = np.maximum(np.abs(stats.Phi_delta_max), np.abs(stats.Phi_delta_min))
    for i in range(len(stats.step_res_Lmax)):
        stats.step_res_L1[i] = np.array(stats.step_res_L1[i])
        stats.step_res_L2[i] = np.array(stats.step_res_L2[i])
        stats.step_res_max[i] = np.array(stats.step_res_max[i])
        stats.step_res_min[i] = np.array(stats.step_res_min[i])
        stats.step_res_Lmax[i] = np.maximum(np.abs(stats.step_res_max[i]), np.abs(stats.step_res_min[i]))
    return stats
    
def load_dir_stat_file(path):
    return load_stat_file(path + "/stats.csv")

def list_subdirs(dirname, recursive=False):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()].sort()
    if recursive:
        for dirname in list(subfolders):
            subfolders.extend(list_subdirs(dirname))
    return subfolders

def plot_stats_lmax(stats):
    plt.plot(stats.time, stats.T_delta_Lmax, color='red', label='T delta Lmax')
    plt.plot(stats.time, stats.Phi_delta_Lmax, color='cyan', label='Phi delta Lmax')
    colors = ['#f27f27', '#c95c1c', '#b54414', '#a12f0d']
    for i in range(stats.step_res_count):
        plt.plot(stats.time, stats.step_res_Lmax[i], color=colors[min(i, len(colors) - 1)], label=f'Step res Lmax #{i+1}')
    plt.xlabel('timestep') 
    plt.ylabel('delta') 
    plt.legend() 
    plt.yscale('log')
    plt.grid()
    plt.show() 

def plot_stats_l2(stats):
    plt.plot(stats.time, stats.T_delta_L2, color='red', label='T delta L2')
    plt.plot(stats.time, stats.Phi_delta_L2, color='cyan', label='Phi delta L2')
    colors = ['#f27f27', '#c95c1c', '#b54414', '#a12f0d']
    for i in range(stats.step_res_count):
        plt.plot(stats.time, stats.step_res_L2[i], color=colors[min(i, len(colors) - 1)], label=f'Step res L2 #{i+1}')
    plt.xlabel('timestep') 
    plt.ylabel('delta') 
    plt.legend() 
    plt.yscale('log')
    plt.grid()
    plt.show() 

def plot_map(map, name):
    linx = np.linspace(0, map.dx*map.nx, map.nx+1)
    liny = np.linspace(0, map.dy*map.ny, map.ny+1)
    X, Y = np.meshgrid(linx, liny)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.pcolormesh(X, Y, map.maps[name], cmap='RdBu_r', shading='flat', vmin=0, vmax=1)
    plt.colorbar(img, ax=ax)
    plt.show() 

def path_rel(base, path):
    return os.path.join(os.path.dirname(__file__), base, path)

def phi_map_discretize(map):
    return (map > 0.5).astype(int)

def show_comparison(base,path1, path2, name, i, filter=False):
    name1 = path1.split("__")[-1]
    name2 = path2.split("__")[-1]

    maps1 = load_dir_bin_map_file(path_rel(base, path1), i)
    maps2 = load_dir_bin_map_file(path_rel(base, path2), i)

    map1 = maps1.maps[name]
    map2 = maps2.maps[name]

    if filter:
        map1 = phi_map_discretize(map1)
        map2 = phi_map_discretize(map2)
    diff = map1 - map2
    l1_norm = np.sum(np.abs(diff))
    print(f"l1:{l1_norm} avg:{l1_norm/maps1.N}")

    fig = plt.figure(figsize=(16, 6))

    gs  = fig.add_gridspec(1,3)
    ax1 = fig.add_subplot(gs[0, 0], label=name1)
    ax2 = fig.add_subplot(gs[0, 1], label=name2)
    ax3 = fig.add_subplot(gs[0, 2], label="diff")

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    ax1.set_title(name1)
    ax2.set_title(name2)
    ax3.set_title("diff")

    linx = np.linspace(0, maps1.dx*maps1.nx, maps1.nx+1)
    liny = np.linspace(0, maps1.dy*maps1.ny, maps1.ny+1)
    X, Y = np.meshgrid(linx, liny)
    img1 = ax1.pcolormesh(X, Y, map1, cmap='RdBu_r', shading='flat', vmin=0, vmax=1)
    img2 = ax2.pcolormesh(X, Y, map2, cmap='RdBu_r', shading='flat', vmin=0, vmax=1)
    img_diff = ax3.pcolormesh(X, Y, diff, cmap='RdBu_r', shading='flat')
    plt.colorbar(img1, ax=ax1,fraction=0.046, pad=0.04)
    plt.colorbar(img2, ax=ax2,fraction=0.046, pad=0.04)
    plt.colorbar(img_diff, ax=ax3,fraction=0.046, pad=0.04)

    plt.subplots_adjust(wspace=0.4)
    fig.show()
    plt.show()

#Basic comparison
if False:
    show_comparison(
        "showcase/first_comp",
        "2024-06-28__19-53-08__semi-implicit",
        "2024-06-28__19-55-31__explicit-rk4",
        "F", 20, filter=False)
    
else:
    show_comparison(
        "showcase/first_comp",
        "2024-06-28__19-53-08__semi-implicit",
        "2024-06-28__20-00-18__explicit-rk4-adaptive",
        "F", 20, filter=False)



# stats = load_dir_stat_file(abs_path)
# maps_set = load_dir_bin_map_file(abs_path, 1)
# plot_map(maps_set, "grad_Phi")
# plot_stats_l2(stats)
