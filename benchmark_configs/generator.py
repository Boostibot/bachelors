config_template = """
; This is a config file to be used with the allen_cahn solvers. 
; Place this file in the directory from which the executable will be run.
; A copy of this file gets saved alongside the captured state 

[program]
run_tests = false
run_benchmarks = false
run_simulation = true

collect_stats = false
collect_step_residual = false
interactive = false
debug = false ; If set to true launches in debug mode. Debug mode has only effect in interactive mode and can be toggled with D
display_min = 0 ; Can be changed with R and inputting range into the console
display_max = 1 
linear_filtering = false ; Linearly interpolates the shown value making the gradient *prettier*. Can be toggled with L @TODO
collect_stats_every = 0
print_in_noninteractive = false

[simulation]
; Which solver to use? Below are comment options:

solver = {0}
; solver = exact
; solver = explicit
; solver = explicit-rk4
; solver = explicit-rk4-adaptive
; solver = semi-implicit

T_boundary = neumann
Phi_boundary = neumann

; Stops the experiment after 'stop_after' seconds. 
; If is interactive pauses the simulation. If not closes the program.
; stop_after = 99999
stop_after = 0.04

scale = {1}
; scale = 0.25
; scale = 0.5
; scale = 1
; scale = 2
; scale = 4

dt = 0.000005 
gamma = 1
mesh_size_x = {2}
mesh_size_y = {2}

; target conjughate gradient solver average error. 
; If error tolerance cannot be reached until Phi/T_max_iters gives up and returns whatever
; was the last approximation.
; T_tolerance = 5e-9
; Phi_tolerance = 5e-9
T_tolerance = 5e-9
Phi_tolerance = 5e-9
; max ammount of iterations in the conjugate gradient method in solution of T/Phi 
; before giving up
T_max_iters = 20 
Phi_max_iters = 20 

; Tolerance of step residual corrector loop. 
; corrector_max_iters is the maximum number of iters before giving up on the target accuracy.
; This loop is very expensive and requires reruining of the entire step as such it effectively 
; multiplies the total running time by (1 + corrector_max_iters)
corrector_tolerance = 0
corrector_max_iters = 3
do_corrector_loop = false
do_corrector_guess = false
do_exact = false

L = 2
xi = 0.0043
a = 2
b = 1
alpha = 3
beta = 1400
Tm = 1
Tini = 1
L0 = 4

; anisothrophy
S = 0
m = 6
theta0 = 0 

[initial]
inside_phi = 1
outside_phi = 0

inside_T= 0
outside_T= 0 

circle_center = 2 2
circle_radius = 0.05
circle_fade = 0 ;the circle transitions from solid to liquid in `circle_fade*xi` real units

square_from = 0 0
square_to = 0 0

[snapshot]
every = 9999 ;Capture snapshot every 'every' seconds (9999999999 effectively means never)
times = 10 ;Capture times snapshots uniformly distributed over the total running time
folder = snapshots
prefix = 
postfix = {3}
snapshot_initial_conditions = 1
"""

config_cpu_template = """
# INTERTRACK phase interface evolution simulator
# sample parameters file
# ---------------------------------------------

# Initial conditions definition
# ----------------------------

resolution_multiplier	{0}

set icond_formula_u = "0"
set icond_formula_p = "((y-L2/2)^2+(z-L3/2)^2) < 0.05^2"

# File names definition
# ---------------------

set logfile = OUTPUT/intertrack.log
set out_file = OUTPUT/image out_file_suffix = .ncd

# Debug settings
# ----------------

#set debug_logfile=OUTPUT/RK.log
#set snapshot_trigger=OUTPUT/t

# Batch mode postprocessing options
# ---------------------------------

#set pproc_script = PostProc_Scripts/Compress
#set pproc_nofail pproc_nowait #pproc_waitfirst

# =============================================

# Model parameters
# ----------------

# domain dimensions
L1		1
L2		4*resolution_multiplier
L3		4*resolution_multiplier

xi		0.0043
a		2
b		1
alpha		3
beta		1400
L		2
u_star		1

# gamma_0 is not used in model 3
gamma_0		1

u_noise_amp	0

# anisotropy strength (A2 used for 6-fold anisotropy only)
A1		0
A2		0

# Calculation parameters
# ----------------------

final_time	0.04
saved_files	11
delta		5e-9
h_min		1e-10
h		5e-6

# calc_mode help (bit flags)
# Bit		ZERO			ONE
# --------------------------------------------------------
# 0	(1)	2nd_order flux		MPFA
# 1	(2)	Dirichlet b.c. for u	Neumann b.c. for u
# 2	(4)	Dirichlet b.c. for p	Neumann b.c. for p
# 3	(8)	4-fold anisotropy	6-fold anisotropy
# 4	(16)	anis. set by bit 3	8-fold anisotropy

calc_mode	2+4

# Grid dimensions
# ---------------

# The grid will have 'grid_nodes' along the longest side of the domain

grid_nodes	{1}*resolution_multiplier

n1		1
n2		grid_nodes
n3		grid_nodes

set comment="Testing run, isotropic"
"""

queue_submit_template = """
#!/bin/bash 
### Job Name 
#PBS -N {0} 
### required runtime 
#PBS -l walltime=00:30:00 
### queue for submission 
#PBS -q gpuA
 
### Merge output and error files 
#PBS -j oe 
 
#PBS -l select=1:mem=10G:ncpus=1:ngpus=1 
 
### start job in the directory it was submitted from 
cd $PBS_O_WORKDIR 
 
### load the necessary software modules 
module load cuda/12.4.1 
module load gcc/13.3
 
### run the application 
./build/main.out {1}
"""

solvers = ['explicit', 'explicit-rk4', 'explicit-rk4-adaptive', 'semi-implicit', 'cpu']
scales = [0.25, 0.5, 1, 2, 4]
base_size = 512
base_path = "benchmark_configs"

for solver in solvers:
    for scale in scales:
        size = int(base_size*scale)
        specialized = ""
        extension = ".ini"
        if solver == 'cpu':
            extension = ""
            specialized = config_cpu_template.format(scale, base_size)
        else:
            specialized = config_template.format(solver, scale, base_size, size)

        filename = f"{base_path}/config_{solver}_{size}{extension}"
        submit_filename = f"{base_path}/submit_{solver}_{size}.sh"
        submit_command = queue_submit_template.format(f"dendrtic_cuda_{solver}_{size}", filename)

        with open(filename, "w") as text_file:
            text_file.write(specialized)

        if solver != 'cpu':
            with open(submit_filename, "w") as text_file:
                text_file.write(submit_command)

import os
if False:
    for solver in solvers:
        for scale in scales:
            size = int(base_size*scale)
            submit_filename = f"{base_path}/submit_{solver}_{size}.sh"
            os.system(f"qsub {submit_filename}")

if True:
    for solver in solvers:
        for scale in scales:
            size = int(base_size*scale)
            config_filename = f"{base_path}/config_{solver}_{size}.ini"
            os.system(f"build/main.out {config_filename}")