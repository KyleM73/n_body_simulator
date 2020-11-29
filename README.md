# N-Body Dynamics Simulator
 
SIMULATES THE DYNAMICS OF N MASSIVE BODIES UNDER THE GRAVITATIONAL FORCE
MASSES ARE SIMULATED AS POINT MASSES AND PROPEGATED FORWARD IN TIME VIA
KICK-DRIFT-KICK DISCRETE TIME PROPEGATION (SEE n_body.simulate() BELOW)

AUTHOR: KYLE MORGENSTEIN (kylem@utexas.edu)

See www.kylemorgenstein.com for more

DATE: 11/28/2020

## USAGE DOCUMENTATION


this example generates the movie file listed in the repository under "n_body_trials/*"
using:
G = 1
view_lim = 5
save = 100
run_time = 100

set initial conditions

>init_conds = np.array([
	[100,0,0,0,0,0,0],
	[1,0,-3,0,3,4,0],
	[1,0,3,0,-3,-4,0],
	[1,-3,0,0,0,-4,0],
	[1,3,0,0,0,4,0]
	])

set experiment runtime
>run_time = 10 [seconds]

initialize simulation class

>sim = n_body(init_conds,run_time)

run simulation

>sim.simulate()

plot the results

sets framerate for saved simulation - set to 0 to playback only

>save = 100

automatically adjusts view to keep all masses in view

>autoscroll = False 

better to just generate the video and watch it at full speed

>replay = False

scales view to [-view_lim, view_lim]

>view_lim = 20

>sim.plot(save,autoscroll,replay,view_lim)

saved simulations take up ~2-20 MB depedning on run_time
generating the simulated video will take ~1-15 minutes 
depending on the length of simulation and your hardware

### WAIT UNTIL THE CODE FINISHES RUNNING COMPLETLY BEFORE TRYING TO OPEN THE VIDEO FILE

## OTHER HELPFUL USAGE TIPS

if you don'y care about selecting parameters, 
the entire simulation can be run inline as

>n = 3

>sim = n_body(3).run()

or 
	
randomly generates 2-5 masses

>sim = n_body().run()  

The parameters you should focus on changing are:

- initial_conds - be creative with your initial conditions!
	
- self.G - strength of gravity
	
- self.S - damping on collisions

- run_time - simulation length

- scale - scales maximum radius of uniform random distribution for random poisition generation

- save - frames per second of simulation output video

- view_lim - axes ranges for visualization

DEV NOTES:

- visualizations are mapped from X-Y-Z space to the X-Y plane for visualization

- visualizations are shown in the Center-of-Mass (COM) frame

- everything *should* work but I wrote this in three days so pls don't roast me on Twitter for my hacky code thx <3

- lmk if something is broken though, thx!!!

- have fun :)
