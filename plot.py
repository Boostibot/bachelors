import matplotlib.pyplot as plt 
import matplotlib.image
import struct
import mpl_toolkits.axes_grid1
import mpl_toolkits.axes_grid1.inset_locator
import numpy as np
import csv 
import math
import sys
import os 
import scipy
import scipy.ndimage
import mpl_toolkits

# I ❤︎ C++
class Map_Set:
    nx = 0
    ny = 0
    dx = 0
    dy = 0
    time = 0.0
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
        out.time, pos = read_f64(pos)
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

        print(f"Map file '{path}' read. dims:{out.nx}x{out.ny} size:{map_count*out.N*8/1024/1024}MB maps:{map_count} {names} time:{out.time}")
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
    time = None
    iter = None
    Phi_iters = None
    T_iters = None
    T_delta_L1 = None
    T_delta_L2 = None
    T_delta_Lmax = None
    T_delta_max = None
    T_delta_min = None
    Phi_delta_L1 = None
    Phi_delta_L2 = None
    Phi_delta_Lmax = None
    Phi_delta_max = None
    Phi_delta_min = None
    step_res_L1 = None
    step_res_L2 = None
    step_res_Lmax = None
    step_res_max = None
    step_res_min = None

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
    stats.time = []
    stats.iter = []
    stats.Phi_iters = []
    stats.T_iters = []
    stats.T_delta_L1 = []
    stats.T_delta_L2 = []
    stats.T_delta_Lmax = []
    stats.T_delta_max = []
    stats.T_delta_min = []
    stats.Phi_delta_L1 = []
    stats.Phi_delta_L2 = []
    stats.Phi_delta_Lmax = []
    stats.Phi_delta_max = []
    stats.Phi_delta_min = []
    stats.step_res_L1 = sa()
    stats.step_res_L2 = sa()
    stats.step_res_Lmax = sa()
    stats.step_res_max = sa()
    stats.step_res_min = sa()

    num_row = 0
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
                num_row += 1
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

DPI = None
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

def plot_map(map, name, min=0, max=1):
    linx = np.linspace(0, map.dx*map.nx, map.nx+1)
    liny = np.linspace(0, map.dy*map.ny, map.ny+1)
    X, Y = np.meshgrid(linx, liny)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.pcolormesh(X, Y, map.maps[name], cmap='RdBu_r', shading='flat', vmin=min, vmax=max)
    plt.colorbar(img, ax=ax)
    plt.show() 

def path_rel(base, path):
    return os.path.join(os.path.dirname(__file__), base, path)

def phi_map_discretize(map, threshold=0.5):
    return (map > threshold).astype(int)

def plot_phase_comparison(base,path1, path2, name, i, filter=False):
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


def extract_outline(values, star=0, threshold=0.5, close_paths=False):
    (nx, ny) = values.shape
    F_levelset = phi_map_discretize(values, threshold)

    # 0 0 0 0
    # 0 0 0 0
    # 1 1 1 1
    # 1 1 1 1

    #  -1-1
    #   1 1
    #   0 0 

    kernel = [
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0],
    ]

    directions = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]

    convolved = scipy.ndimage.convolve(F_levelset, kernel, mode='constant')

    visited = np.zeros(shape=(nx, ny), dtype=np.uint32)

    def point_add(point, dir):
        return (point[0] + dir[0], point[1] + dir[1])

    lines = []
    edge = convolved > 0

    iter = 0
    while True:
        # Select nice start position. That is a posisiton that has both left and right
        # or top and bot neighbours thus the line can be joined up nicely
        # F_starts = (convolved - visited*20) == 1 
        F_starts = np.logical_and(convolved == 1, visited == 0) 
        iter += 1

        nonzero = np.nonzero(F_starts)
        if len(nonzero[0]) == 0:
            break
        
        # iterate around shape
        point = (nonzero[0][star], nonzero[1][star])
        line = []
        step = 1

        while True:
            assert edge[point] > 0
            line.append([point[0], point[1]])
            visited[point] = 1
            step += 1
            found = False
            dir = None
            last_i = 0
            for i in range(8):
                last_i += 1
                dir = directions[last_i % 8]
                off = point_add(point, dir)
                if 0 <= off[0] and off[0] < nx and 0 <= off[1] and off[1] < ny:
                    if edge[off] > 0 and visited[off] == 0:
                        found = True
                        break
            if found == False:
                break
            point = point_add(point, dir)

        if False:
            linx = np.linspace(0, 1, nx+1)
            liny = np.linspace(0, 1, ny+1)
            X, Y = np.meshgrid(linx, liny)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.pcolormesh(Y, X, edge + 3*visited, cmap='RdBu_r', shading='flat')
            start_circle = plt.Circle((line[0][0]/nx, line[0][1]/ny), 1/(2*nx), color='g')
            end_circle = plt.Circle((line[-1][0]/nx, line[-1][1]/ny), 1/(2*nx), color='r')
            ax.add_patch(start_circle)
            ax.add_patch(end_circle)
            plt.show()

        # close loop
        if close_paths:
            line.append(line[0])
        if len(line) > 3:
            # if not degenerate shape (line)
            points = np.array(line)
            pointsT = points.T
            allx = np.all(pointsT[0] == line[0][0])
            ally = np.all(pointsT[1] == line[0][1])
            if allx == False and ally == False:
                lines += [points]

    return lines


from collections import defaultdict
def join_path_segments(segments):
    def first_item(collection):
        if len(collection) == 0:
            return None
        return next(iter(collection))

    if len(segments) == 0:
        return []

    conectivity = defaultdict(set)
    for line in segments:
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])

        conectivity[p1].add(p2)
        conectivity[p2].add(p1)

    lines = []
    while len(conectivity) > 0:
        (p1, set1) = first_item(conectivity.items())
        if len(set1) == 0:
            break

        line = [p1]
        while True:
            p2 = first_item(set1)
            if p2 == None:
                break
            
            set2:set = conectivity[p2]

            line.append(p2)
            set1.remove(p2)
            set2.remove(p1)

            if len(set1) == 0:
                conectivity.pop(p1)
                
            if len(set2) == 0:
                conectivity.pop(p2)

            set1 = set2
            p1 = p2

        lines += [np.array(line)]
    return lines

def marching_squares(field, treshold):
    (nx, ny) = field.shape
    discretized = field > treshold
    states = (
        discretized[0:nx-1, 0:ny-1]     # bot left
        + 2*discretized[1:nx,   0:ny-1] # bot right
        + 4*discretized[1:nx,   1:ny]   # top right
        + 8*discretized[0:nx-1, 1:ny]   # top left
    )

    fxt_arr = (treshold - field[:nx-1, 1:ny]) /(field[1:nx,  1:ny] - field[:nx-1,  1:ny])
    fxb_arr = (treshold - field[:nx-1, :ny-1])/(field[1:nx, :ny-1] - field[:nx-1, :ny-1])
    fyr_arr = (treshold - field[1:nx,  :ny-1])/(field[1:nx,  1:ny] - field[1:nx,  :ny-1])
    fyl_arr = (treshold - field[:nx-1, :ny-1])/(field[:nx-1, 1:ny] - field[:nx-1, :ny-1])

    lines = []
    with_values_mask = np.logical_and(states > 0, states < 15)
    nonzeros = np.nonzero(with_values_mask)
    for x,y in zip(nonzeros[0], nonzeros[1]):
        fxt = fxt_arr[x,y]
        fxb = fxb_arr[x,y]
        fyr = fyr_arr[x,y]
        fyl = fyl_arr[x,y]

        state = states[x,y]

        if state == 1 or state == 14:
            lines.append((x, y+fyl, x+fxb, y))
        elif state == 2 or state == 13:
            lines.append((x+1, y+fyr, x+fxb, y))
        elif state == 3 or state == 12:
            lines.append((x, y+fyl, x+1, y+fyr))
        elif state == 4 or state == 11:
            lines.append((x+fxt, y+1, x+1, y+fyr))
        elif state == 5:
            lines.append((x, y+fyl, x+fxt, y+1))
            lines.append((x+1, y+fyr, x+fxb, y))
        elif state == 6 or state == 9:
            lines.append((x+fxb, y, x+fxt, y+1))
        elif state == 7 or state == 8:
            lines.append((x, y+fyl, x+fxt, y+1))
        elif state == 10:
            lines.append((x, y+fyl, x+fxb, y))
            lines.append((x+fxt, y+1, x+1, y+fyr))
        else:
            assert False 

    return lines

def marching_squares_naive(field, treshold):
    lines = []
    (nx, ny) = field.shape
    states = np.zeros((nx,ny))
    for y in range(0, ny-1):
        for x in range(0, nx-1):
            top_right = field[x+1,y+1] > treshold
            top_left  = field[x,y+1] > treshold
            bot_right = field[x+1,y] > treshold
            bot_left  = field[x,y] > treshold

            state = (0
                | int(top_left)  << 3
                | int(top_right) << 2
                | int(bot_right) << 1
                | int(bot_left)  << 0)

            fxt = (treshold - field[x,y+1])/(field[x+1,y+1] - field[x,y+1])
            fxb = (treshold - field[x,  y])/(field[x+1,y  ] - field[x  ,y])
            fyr = (treshold - field[x+1,y])/(field[x+1,y+1] - field[x+1,y])
            fyl = (treshold - field[x  ,y])/(field[x,  y+1] - field[x  ,y])

            # fxt = 0.5
            # fxb = 0.5
            # fyr = 0.5
            # fyl = 0.5

            states[x,y] = state
            if state == 1 or state == 14:
                lines.append((x, y+fyl, x+fxb, y))
            elif state == 2 or state == 13:
                lines.append((x+1, y+fyr, x+fxb, y))
            elif state == 3 or state == 12:
                lines.append((x, y+fyl, x+1, y+fyr))
            elif state == 4 or state == 11:
                lines.append((x+fxt, y+1, x+1, y+fyr))
            elif state == 5:
                lines.append((x, y+fyl, x+fxt, y+1))
                lines.append((x+1, y+fyr, x+fxb, y))
            elif state == 6 or state == 9:
                lines.append((x+fxb, y, x+fxt, y+1))
            elif state == 7 or state == 8:
                lines.append((x, y+fyl, x+fxt, y+1))
            elif state == 10:
                lines.append((x, y+fyl, x+fxb, y))
                lines.append((x+fxt, y+1, x+1, y+fyr))
            elif state == 0 or state == 15:
                pass
            else:
                assert False 

    return lines

def interpolate_outline(line, samples=10, k=3, smoothness=5, per=True):
    linex, liney = zip(*np.array(line))
    linex = np.array(linex) 
    liney = np.array(liney)
    try:
        f, u = scipy.interpolate.splprep([linex, liney], k=k, s=smoothness, per=per)
        xint, yint = scipy.interpolate.splev(np.linspace(0, 1, len(line)*samples), f)
        return xint, yint
    except:
        return linex, liney

def interpolate_outline2(line, samples=10, k=3, smoothness=5, per=True):
    points = np.array(line)

    try:
        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]

        # Build a list of the spline function, one for each dimension:
        splines = []
        for coords in points.T:
            splines += [scipy.interpolate.UnivariateSpline(distance, coords, k=k, s=smoothness)]

        alpha = np.linspace(0, 1, math.floor(samples*len(line)))
        xs = splines[0](alpha)
        ys = splines[1](alpha)
        return xs, ys
    except:
        transp = points.T
        return transp[0], transp[1]

def chaikins_corner_cutting(coords, refinements=10):
    coords = np.array(coords)

    for i in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

        print(f"refinement {i}")

    return coords

def dissable_plot_ticks(ax):
    ax.tick_params(axis='both', which='both', 
                    bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    
def plot_loaded_temperature_interface_map(maps1, F, U, smoothness=0.0055, min=0, max=1, save=None, line_color='white', text_color=None, colorbar='inset', background="U", do_outlines=True, old_outline=True, per=True):
    label = 'T'
    cmap = 'RdBu_r'
    if background == "F":
        cmap = 'viridis'
        label = 'ϕ'

    if text_color == None:
        text_color = line_color

    outlines = []
    if do_outlines:
        if old_outline:
            outlines = extract_outline(F)
        else:
            outlines = join_path_segments(marching_squares(F, 0.5))

    linx = np.linspace(0, maps1.dx*maps1.nx, maps1.nx+1)
    liny = np.linspace(0, maps1.dy*maps1.ny, maps1.ny+1)
    X, Y = np.meshgrid(linx, liny)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    img = ax.pcolormesh(X, Y, U, cmap=cmap, shading='flat', vmin=min, vmax=max)

    for outline in outlines:
        outline = (np.array(outline) + 0.5)*(maps1.dx, maps1.dy) 
        xint, yint = interpolate_outline(outline, k=3, smoothness=smoothness, per=per)
        ax.plot(yint, xint, c=line_color, linewidth=1)

    if colorbar == 'default':
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    elif colorbar == 'inset':
        cbaxes = mpl_toolkits.axes_grid1.inset_locator.inset_axes(ax, width="5%", height="40%", loc=4, borderpad=2) 
        cb = plt.colorbar(img, cax=cbaxes, ax=ax, ticks=[0.,1], orientation='vertical')
        cb.ax.tick_params(axis='both', direction='in', color=text_color)
        cb.set_label(label, loc='center', rotation="horizontal", color=text_color, labelpad = 0.5)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=text_color)
        dissable_plot_ticks(ax)
    else:
        dissable_plot_ticks(ax)

    ax.set_xlim(xmin=0, xmax=maps1.dx*maps1.nx)
    ax.set_ylim(ymin=0, ymax=maps1.dy*maps1.ny)

    if save == None:
        plt.show() 
    else:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0, dpi=DPI)

def plot_temperature_interface_map(base, path, i, smoothness=0.0035, min=0, max=1, save=None, text_color=None, line_color='white', colorbar='inset', background="U", do_outlines=True, old_outline=True, per=True):
    maps1 = load_dir_bin_map_file(path_rel(base, path), i)
    plot_loaded_temperature_interface_map(
        maps1, maps1.maps['F'], maps1.maps[background], smoothness=smoothness, line_color=line_color, text_color=text_color, min=min, max=max, 
        save=save, colorbar=colorbar, background=background, do_outlines=do_outlines, old_outline=old_outline, per=per)

def plot_phase_interface(base, path, i, xi = 0.0043, linewidth=4, save=None):
    maps1 = load_dir_bin_map_file(path_rel(base, path), i)
    F = maps1.maps["F"]
    half_nx = maps1.nx//2
    half_ny = maps1.ny//2
    interface = F[half_ny][0:half_nx]
    linx = np.linspace(0, maps1.dx*half_nx, half_nx)

    comp_epsilon = 0.01
    interface_indeces = np.flatnonzero(np.abs(interface - 0.5) < 0.5 - comp_epsilon)
    first_i = interface_indeces[0]
    last_i = interface_indeces[-1]
    diff = linx[last_i] - linx[first_i]
    print(f"interface: {linx[first_i]} - {linx[last_i]} = {diff} ({diff/xi} xi) ({diff/maps1.dx} dx)")

    extra_cells = 4
    display_epsilon = 0.001

    interface_indeces = np.flatnonzero(np.abs(interface - 0.5) < 0.5 - display_epsilon)
    first_i = interface_indeces[0]
    last_i = interface_indeces[-1]

    ex_first_i = max(first_i - extra_cells, 0)
    ex_last_i = min(last_i + extra_cells, half_nx)

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', pad=15)
    ax.set(xlim=(linx[ex_first_i], linx[ex_last_i]), ylim=(-0.05, 1.05))
    ax.plot(linx[ex_first_i:ex_last_i], interface[ex_first_i:ex_last_i], color='#17bf23', label='phase interface', linewidth=linewidth)
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(maps1.dx))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(maps1.dx*10))
    plt.xlabel('x') 
    plt.ylabel('ϕ', rotation="horizontal") 
    # plt.legend() 
    # plt.yscale('log')
    ax.grid()
    if save != None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0, dpi=DPI)
    else:
        plt.show() 

def format_latex_e(n):
    if n == 0:
        return '$0$' 
    formatted = '%.2E' % n
    mantissa = formatted.split('E')[0]
    exponent = formatted.split('E')[1]
    if exponent[0] == '+':
        exponent = exponent[1:].lstrip('0')
    else:
        exponent = exponent[0] + exponent[1:].lstrip('0')
        
    if exponent == '':
        return '$%.2f$' % n
    else:
        return '$' + mantissa + '\\times 10^{' + exponent + '}$'

def phase_field_dist(a, b, area, ord=1):
    x = (a - b).flatten()
    return np.linalg.norm(x, ord=ord)*area

def phase_field_discrete_dist(a, b, area, ord=1):
    x = (phi_map_discretize(a) - phi_map_discretize(b)).flatten()
    return np.linalg.norm(x, ord=ord)*area

def format_latex_table(middle_label:str, left_labels:list[str], top_labels:list[str], series:list[list[float | int]], spacing:str='[-1em]') -> str:
    output = ''
    separator_line = '\\hline\n' + '& '*len(top_labels) + '\\\\' + spacing + '\n'
    first_line = ''
    for label in top_labels:
        first_line += ' & \\multicolumn{1}{c|}{%s}' % label

    output += separator_line
    output += middle_label + first_line + '\\\\\n'

    max_label_size = 0
    for label in left_labels:
        max_label_size = max(max_label_size, len(label))

    max_column_sizes = []
    if len(series) > 0:
        max_column_sizes = [0]*len(series[0])

    for datapoints in series:
        for i,val in enumerate(datapoints):
            fmt = format_latex_e(val)
            max_column_sizes[i] = max(max_column_sizes[i], len(fmt))

    for label, datapoints in zip(left_labels, series):
        line = f'{label: <{max_label_size}}'
        for i,val in enumerate(datapoints):
            line += f" & {format_latex_e(val): <{max_column_sizes[i]}}"
        
        output += separator_line
        output += line + '\\\\\n'

    return output
    

def plot_phase_comparison_maps(base, paths, i, save=None, smoothness=0.0035, linewidth=1, xmin=1.6, xmax=2.4, ymin=2.6, ymax=None, legend_under=False, print_comp_table=False):
    if len(paths) == 0:
        return

    map_sets = []
    maps = []
    for entry in paths:
        map_set = load_dir_bin_map_file(path_rel(base, entry[0]), i) 
        map_sets += [map_set]
        maps += [map_set.maps['F']]

    if print_comp_table:
        if False:
            def print_comp_table(map_sets, dist):
                print(f"stats in {dist} =========================")

                separator_line = '\\hline\n' + '& '*len(map_sets) + '\\\\[-1em]'
                first_line = ''
                for i,map_set in enumerate(map_sets):
                    first_line += ' & \\multicolumn{1}{c|}{%s}' % paths[i][1]

                print(separator_line)
                print(f'Schemes {first_line} \\\\')

                for i,map_set in enumerate(map_sets):
                    dxdy = map_set.dx*map_set.dy
                    formated = paths[i][1].ljust(7)
                    for compare_with in map_sets:
                        val = 0
                        if dist == 'PFD':
                            val = phase_field_dist(map_set.maps['F'], compare_with.maps['F'], dxdy)
                        else:
                            val = phase_field_discrete_dist(map_set.maps['F'], compare_with.maps['F'], dxdy)
                        formated += f" & {format_latex_e(val)}"
                    
                    print(separator_line)
                    print(f'{formated} \\\\')

            print_comp_table(map_sets, 'PFD')
            print_comp_table(map_sets, 'PFDD')

        assert len(paths) == len(map_sets)
        labels = []
        seiries_pfd = []
        seiries_pfdd = []
        for i, map_set in enumerate(map_sets):
            labels += [paths[i][1]]
            dxdy = map_set.dx*map_set.dy
            datapoints_pfd = []
            datapoints_pfdd = []
            for compare_with in map_sets:
                pfd = phase_field_dist(map_set.maps['F'], compare_with.maps['F'], dxdy)
                pfdd = phase_field_discrete_dist(map_set.maps['F'], compare_with.maps['F'], dxdy)
                datapoints_pfd += [pfd]
                datapoints_pfdd += [pfdd]

            seiries_pfd += [datapoints_pfd]
            seiries_pfdd += [datapoints_pfdd]

        latex_table_PFD = format_latex_table("Schemes ", labels, labels, seiries_pfd)
        latex_table_PFDD = format_latex_table("Schemes ", labels, labels, seiries_pfdd)

        print("============== printing PFD table =============")
        print(latex_table_PFD)
        print("============== printing PFDD table =============")
        print(latex_table_PFDD)

    fig_width = 10
    if legend_under:
        fig_width = 16

    fig = plt.figure(figsize=(fig_width, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.tick_params(axis='both', which='major', pad=15)
    for i, map_set in enumerate(map_sets):
        outlines = extract_outline(map_set.maps['F'])
        xints = np.array([])
        yints = np.array([])
        for outline in outlines:
            scaled = (np.array(outline) + 0.5)*(map_set.dx, map_set.dy) 
            xint, yint = interpolate_outline2(scaled, k=5, smoothness=smoothness)
            xints = np.concatenate((xints, xint))
            yints = np.concatenate((yints, yint))

        color = None
        if len(paths[i]) >= 3:
            color = paths[i][2]

        ax.plot(xints, yints, color=color, label=paths[i][1], linewidth=linewidth)

    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)
    if legend_under:
        ax.legend(bbox_to_anchor=(1, 1.05), loc="upper left")
    else:    
        ax.legend()
    
    fig.tight_layout()

    if save == None:
        plt.show()
    else:
        fig.savefig(save, bbox_inches='tight', dpi=DPI)

    return 0

def plot_step_residual_comp(base, loop, corr_loop, name, save=None, linewidth=1.5):
    loop_stat = load_dir_stat_file(path_rel(base, loop)) 
    corr_loop_stat = load_dir_stat_file(path_rel(base, corr_loop)) 

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    colors = ['#E6B200', '#00E630', '#00E6CC', '#00B5E6']
    dxdy = 16/loop_stat.nx*loop_stat.ny
    ax.plot(loop_stat.time, loop_stat.Phi_delta_L1*dxdy, color='black', label=rf'{name} 3 reps. $\Delta\Phi$')
    for i in range(loop_stat.step_res_count):
        ax.plot(loop_stat.time, loop_stat.step_res_L1[i]*dxdy, linewidth=linewidth, color=colors[i], label=rf'{name} 3 reps. $r_\Phi$ k={i+1}')

    for i in range(corr_loop_stat.step_res_count):
        ax.plot(corr_loop_stat.time, corr_loop_stat.step_res_L1[i]*dxdy, linewidth=linewidth, color=colors[i], linestyle='dashed', label=rf'{name} corr. 3 reps. $r_\Phi$ k={i+1}')

    ax.set_xlim(xmin=0, xmax=0.01)
    plt.xlabel('t') 
    plt.yscale('log')
    ax.yaxis.grid()
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=2)
    fig.tight_layout()

    if save == None:
        plt.show() 
    else:
        fig.savefig(save, bbox_inches='tight', dpi=DPI)

    plt.cla()

def plot_bench_results(linewidth=2, save=None):

    Ys = [3, 10, 20, 30, 40, 50, 60, 70]
    Ns = [256**2, 512**2, 1024**2, 2048**2, 4096**2, 2*4096**2]
    Ns_labels = ['$256^2$', '$512^2$', '$1024^2$', '$2048^2$', '$4096^2$', '$2×4096^2$']
    inf = math.inf
    cpu = [[6.257900e-05, 3.90], [2.502550e-04, 3.90], [1.006565e-03, 3.88], [4.038604e-03, 3.87], [1.612710e-02, 3.88], [3.218363e-02, 3.88], ]
    thrust = [[2.048700e-05, 11.92], [3.345000e-05, 29.19], [7.647900e-05, 51.08], [2.468640e-04, 63.29], [1.086272e-03, 57.54], [2.353407e-03, 53.11], ]
    cust = [[1.701700e-05, 14.35], [2.933800e-05, 33.29], [8.709500e-05, 44.85], [2.953550e-04, 52.90], [9.418090e-04, 66.36], [1.838005e-03, 68.01], ]

    cpu = np.array(cpu)
    thrust = np.array(thrust)
    cust = np.array(cust)

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', pad=15)
    ax.plot(Ns, cpu.T[1], color='black', linestyle='-', marker='o', label='cpu', linewidth=linewidth)
    ax.plot(Ns, thrust.T[1], color='green', linestyle='-', marker='o', label='thrust', linewidth=linewidth)
    ax.plot(Ns, cust.T[1], color='blue', linestyle='-', marker='o', label='custom', linewidth=linewidth)

    str = "cpu "
    for x in cpu.T[0]:
        str += f"& {format_latex_e(x)} "
    print(str + '\\\\')

    str = "thrust "
    for x in thrust.T[0]:
        str += f"& {format_latex_e(x)} "
    print(str + '\\\\')

    str = "custom "
    for x in cust.T[0]:
        str += f"& {format_latex_e(x)} "
    print(str + '\\\\')

    ax.set_xscale('log')
    plt.xticks(Ns, Ns_labels, rotation=45)
    plt.yticks(Ys)
    plt.xlabel('N') 
    plt.ylabel('GB/s', rotation="horizontal") 
    plt.grid()
    plt.legend() 
    if save != None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0, dpi=DPI)
    else:
        plt.show() 

    None

# module load
# 
def plot_ellpased_time_comp_methods(linewidth=2, save=None):

    Ys = [1, 4, 16, 64, 256, 1024, 4096]
    Ns = [128**2, 256**2, 512**2, 1024**2, 2048**2]
    Ns_labels = ['$128^2$', '$256^2$', '$512^2$', '$1024^2$', '$2048^2$']
    
    inf = math.inf

    running_times = {
        'Euler':[0.86,  1.65,  4.93,  14.11,  50.58],
        'RK4'  :[3.53,  8.38, 24.16,  82.75, 313.40],
        'RKM'  :[4.77, 16.10, 47.77, 189.25, 18*60],
        'SI'   :[3.65, 11.18, 40.26, 187.20, 18*60],
    }


    colors=['#4E4DF0', '#944DF0', '#DA4DF0', '#F04D97', '#F05C4D']
    wider_color = ['#4DB4F0', '#5E4DF0', '#DA4DF0', '#F0544D', '#FA8E34', '#F5D63D', '#ABFF24']
    

    print('% ============== Running times ==============')
    print(format_latex_table('Running times ', list(running_times.keys()), Ns_labels, list(running_times.values())))

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', pad=15)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)

    i = 0
    for key, value in running_times.items():
        ax.plot(Ns, value, linestyle='-', marker='o', color=wider_color[i], label=key, linewidth=linewidth)
        i += 1

    plt.xticks(Ns, Ns_labels, rotation=45)
    plt.yticks(Ys)
    plt.xlabel('Number of cells') 
    plt.ylabel('Computation time [s]') 
    plt.subplots_adjust(bottom=0.2)
    plt.autoscale()
    plt.grid()
    plt.legend() 
    if save != None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0, dpi=DPI)
    else:
        plt.show() 

    scalings = {}
    for key, series in running_times.items():
        scaling = []
        for i in range(1, len(series)):
            scaling += [series[i]/series[i - 1]]
        scalings[key] = scaling

    print('% ============== Scaling times ==============')
    print(format_latex_table('Running times ', list(scalings.keys()), Ns_labels[1:], list(scalings.values())))

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', pad=15)
    ax.set_xscale('log', base=2)

    i = 0
    for key, value in scalings.items():
        ax.plot(Ns[1:], value, linestyle='-', marker='o', color=wider_color[i], label=key, linewidth=linewidth)
        i += 1

    plt.xticks(Ns, Ns_labels, rotation=45)
    # plt.yticks(Ys)
    plt.xlabel('Number of cells') 
    plt.ylabel('Computation time scaling') 
    plt.subplots_adjust(bottom=0.2)
    plt.autoscale()
    plt.grid()
    plt.legend() 
    if save != None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.0, dpi=DPI)
    else:
        plt.show() 


DPI = 180
font = {'size' : 22}
matplotlib.rc('font', **font)

if False:
    plot_ellpased_time_comp_methods()

if False:
    plot_phase_comparison_maps("showcase", [
        ("0_aniso_comp_256__explicit", "Euler"), 
        ("0_aniso_comp_256__explicit-rk4", "RK4"), 
        ("0_aniso_comp_256__explicit-rk4-adaptive", "RKM"), 
        ("0_aniso_comp_256__semi-implicit", "S-I")
    ], 10, print_comp_table=True)

plot_temperature_interface_map("showcase", "0_aniso_comp_256__explicit-rk4-adaptive", 10, background="U", smoothness=0.0005, old_outline=False)
if False:
    plot_temperature_interface_map("showcase", "0_aniso_comp_256__explicit", 10, background="U")
    plot_temperature_interface_map("showcase", "0_aniso_comp_256__explicit-rk4", 10, background="U", per=False)
    plot_temperature_interface_map("showcase", "0_aniso_comp_256__semi-implicit", 10, background="U")
exit()

# First showcase
if False:
    plot_temperature_interface_map("showcase", "show_low_xi", 5, background="U", save="showcase/exported/show_low_xi_U_5.png")
    plot_temperature_interface_map("showcase", "show_low_xi", 13, background="U", save="showcase/exported/show_low_xi_U_13.png")
    plot_temperature_interface_map("showcase", "show_low_xi", 20, background="U", save="showcase/exported/show_low_xi_U_20.png")

# Def values 8-fold
if False:
    plot_temperature_interface_map("showcase", "show_low_xi_anisofold_8", 6, background="U", save="showcase/exported/show_low_xi_anisofold_8_U_6.png")
    plot_temperature_interface_map("showcase", "show_low_xi_anisofold_8", 18, background="U", save="showcase/exported/show_low_xi_anisofold_8_U_18.png")
    plot_temperature_interface_map("showcase", "show_low_xi_anisofold_8", 30, background="U", save="showcase/exported/show_low_xi_anisofold_8_U_30.png")

# xi comparison
if False:
    plot_temperature_interface_map("showcase", "show_low_xi", 5, background="F", save="showcase/exported/show_low_xi_F_5.png")
    plot_temperature_interface_map("showcase", "show_low_xi", 13, background="F", save="showcase/exported/show_low_xi_F_13.png")
    plot_temperature_interface_map("showcase", "show_low_xi", 20, background="F", save="showcase/exported/show_low_xi_F_20.png")
    plot_temperature_interface_map("showcase", "show_medium_xi", 5, background="F", save="showcase/exported/show_medium_xi_F_5.png")
    plot_temperature_interface_map("showcase", "show_medium_xi", 13, background="F", save="showcase/exported/show_medium_xi_F_13.png")
    plot_temperature_interface_map("showcase", "show_medium_xi", 20, background="F", save="showcase/exported/show_medium_xi_F_20.png")
    plot_temperature_interface_map("showcase", "show_tiny_xi", 5, background="F", save="showcase/exported/show_tiny_xi_F_5.png")
    plot_temperature_interface_map("showcase", "show_tiny_xi", 13, background="F", save="showcase/exported/show_tiny_xi_F_13.png")
    plot_temperature_interface_map("showcase", "show_tiny_xi", 20, background="F", save="showcase/exported/show_tiny_xi_F_20.png")

# aniso comparison
if False:
    plot_temperature_interface_map("showcase", "show_aniso_0", 5, background="F", save="showcase/exported/show_aniso_0_F_5.png")
    plot_temperature_interface_map("showcase", "show_aniso_0", 13, background="F", save="showcase/exported/show_aniso_0_F_13.png")
    plot_temperature_interface_map("showcase", "show_aniso_0", 20, background="F", save="showcase/exported/show_aniso_0_F_20.png")
    plot_temperature_interface_map("showcase", "show_low_xi", 5, background="F", save="showcase/exported/show_aniso_0.3_F_5.png")
    plot_temperature_interface_map("showcase", "show_low_xi", 13, background="F", save="showcase/exported/show_aniso_0.3_F_13.png")
    plot_temperature_interface_map("showcase", "show_low_xi", 20, background="F", save="showcase/exported/show_aniso_0.3_F_20.png")
    plot_temperature_interface_map("showcase", "show_aniso_0.5", 5, background="F", save="showcase/exported/show_aniso_0.5_F_5.png")
    plot_temperature_interface_map("showcase", "show_aniso_0.5", 13, background="F", save="showcase/exported/show_aniso_0.5_F_13.png")
    plot_temperature_interface_map("showcase", "show_aniso_0.5", 20, background="F", save="showcase/exported/show_aniso_0.5_F_20.png")

# Dirichlet boundary comparison
if False:
    base = "showcase/exported/"
    plot_temperature_interface_map("showcase", "semi_long_neumann", 2, background="U", save=base+"semi_long_neumann_FU_2.png")
    plot_temperature_interface_map("showcase", "semi_long_neumann", 10, background="U", save=base+"semi_long_neumann_FU_10.png")
    plot_temperature_interface_map("showcase", "semi_long_neumann", 30, background="U", save=base+"semi_long_neumann_FU_30.png")
    plot_temperature_interface_map("showcase", "semi_long_neumann", 60, background="U", save=base+"semi_long_neumann_FU_60.png")
if False:
    base = "showcase/exported/"
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 2, background="F", text_color="black", save=base+"semi_long_dirichlet_F_2.png")
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 10, background="F", text_color="black", save=base+"semi_long_dirichlet_F_10.png")
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 30, background="F", text_color="black", save=base+"semi_long_dirichlet_F_30.png")
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 60, background="F", text_color="black", save=base+"semi_long_dirichlet_F_60.png")
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 2, background="U", text_color="black", do_outlines=False, save=base+"semi_long_dirichlet_U_2.png")
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 10, background="U", text_color="black", do_outlines=False, save=base+"semi_long_dirichlet_U_10.png")
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 30, background="U", text_color="black", do_outlines=False, save=base+"semi_long_dirichlet_U_30.png")
    plot_temperature_interface_map("showcase", "semi_long_dirichlet", 60, background="U", text_color="black", do_outlines=False, save=base+"semi_long_dirichlet_U_60.png")
    
    # t = 0.02
    # t = 0.1
    # t = 0.3
    # t = 0.6

# phase interface graph
if False:
    plot_phase_interface("showcase", "show_aniso_0", 13, save="showcase/exported/show_aniso_inteface_graph.pdf")

#Model comparison
if True:
    plot_phase_comparison_maps("showcase", [
        ("method_comp_euler", "Euler"), 
        ("method_comp_rk4", "RK4"), 
        ("method_comp_rkm", "RKM"), 
        ("method_comp_semi", "S-I")
    ], 20, save="showcase/exported/model_comp.pdf", print_comp_table=True)

#Correction comparison
if True:
    plot_phase_comparison_maps("showcase", [
        ("method_comp_semi",            "S-I", "black"),
        ("method_comp_semi_corr",       "S-I corr."), 
        ("method_comp_semi_loop3",      "S-I 3 reps."), 
        ("method_comp_semi_corr_loop3", "S-I corr., 3 reps."), 
        ("method_comp_rkm_corr",        "RKM corr."), 
    ], 20, legend_under=True, save="showcase/exported/correction_comp.pdf", print_comp_table=True)
    # ], 20, legend_under=True)

#Step residual graphs
if True:
    plot_step_residual_comp("showcase",
        "method_comp_semi_loop3",
        "method_comp_semi_corr_loop3",   
        "S-I",    
        save="showcase/exported/step_res_comp_semi.pdf")
    
    plot_step_residual_comp("showcase",
        "method_comp_euler_loop3",
        "method_comp_euler_corr_loop3",   
        "Euler",    
        save="showcase/exported/step_res_comp_euler.pdf")
if False:
    plot_setp_residual_comp("showcase",
        "method_comp_semi",
        "method_comp_semi_loop3",
        "method_comp_semi_corr",       
        "method_comp_semi_corr_loop3", 
        "method_comp_rkm_corr",
        "method_comp_rkm",
        save="showcase/exported/step_res_comp_l1.pdf", l1=True)
    
        # "method_comp_euler_corr",
        # "method_comp_euler",
        # save="showcase/exported/step_res_comp.png")

# Plot bencmark results
if False:
    plot_bench_results(save="showcase/exported/benchmark.pdf")

hue_colors = ['#7500E6', '#C000E6', '#E600A1', '#E60017', '#E65200', '#E6B200', '#00E630', '#00E6CC', '#00B5E6']
distinct_colors = ['#E6DF00', '#E63E00', '#00E6BA', '#6912E6', '#30917E', '#666533',
                    '#721DDB', '#00DBB8', '#DB531D', '#DBD813', '#86533E', '#493A5C']