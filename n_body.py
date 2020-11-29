#!/usr/bin/env python3
#
# N-BODY DYNAMICS SIMULATOR
# 
# SIMULATES THE DYNAMICS OF N MASSIVE BODIES UNDER THE GRAVITATIONAL FORCE
# MASSES ARE SIMULATED AS POINT MASSES AND PROPEGATED FORWARD IN TIME VIA
# KICK-DRIFT-KICK DISCRETE TIME PROPEGATION (SEE n_body.simulate() BELOW)
#
# AUTHOR: KYLE MORGENSTEIN (KYLEM@UTEXAS.EDU)
# DATE: 11/28/2020
#
# MIT License
#
# Copyright (c) 2020 KYLE MORGENSTEIN
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import datetime

np.random.seed(69) #nice

class n_body:
	'''
	USAGE DOCUMENTATION

	=====================================================================

	#only necessary if the code is being run in a different file
	from n_body import *
	
	#otherwise copy and paste as follows:
	#(this code is supplied at the bottom of the file as well)

	#this example generates the movie file listed in the repository under "n_body_trials/*"
	#using:
	# G = 1
	# view_lim = 5
	# save = 100
	# run_time = 100

	#set initial conditions
	init_conds = np.array([
		[100,0,0,0,0,0,0],
		[1,0,-3,0,3,4,0],
		[1,0,3,0,-3,-4,0],
		[1,-3,0,0,0,-4,0],
		[1,3,0,0,0,4,0]
		])

	#set experiment runtime
	run_time = 10 #[seconds]

	#initialize simulation class
	sim = n_body(init_conds,run_time)

	#run simulation
	sim.simulate()

	#plot the results
	save = 100 #sets framerate for saved simulation - set to 0 to playback only
	autoscroll = False #automatically adjusts view to keep all masses in view
	replay = False #better to just generate the video and watch it at full speed
	view_lim = 20 #scales view to [-view_lim, view_lim]

	sim.plot(save,autoscroll,replay,view_lim)

	#saved simulations take up ~2-20 MB depedning on run_time
	#generating the simulated video will take ~1-15 minutes 
	#depending on the length of simulation and your hardware

	#WAIT UNTIL THE CODE FINISHES RUNNING COMPLETLY BEFORE TRYING TO OPEN THE VIDEO FILE

	=====================================================================

	OTHER HELPFUL USAGE TIPS

	#if you don'y care about selecting parameters, 
	#the entire simulation can be run inline as
	n = 3 #number of masses to simulate
	sim = n_body(3).run()

	or 
	
	#randomly generates 2-5 masses
	sim = n_body().run()  

	The parameters you should focus on changing are:
	# initial_conds - be creative with your initial conditions!
	* self.G - strength of gravity
	* self.S - damping on collisions
	* run_time - simulation length
	* scale - scales maximum radius of uniform random distribution for random poisition generation
	* save - frames per second of simulation output video
	* view_lim - axes ranges for visualization

	DEV NOTES:
	* visualizations are mapped from X-Y-Z space to the X-Y plane for visualization
	* visualizations are shown in the Center-of-Mass (COM) frame
	* everything ~should~ work but I wrote this in three days so pls don't roast me on Twitter for my hacky code thx <3
	* lmk if something is broken though, thx!!!
	* have fun :)
	
	'''

	def __init__(self,init_conds=0,run_time=10,mass_equal=False,init_vel=False,max_pts=6):
		'''
		INITIALIZE N_BODY CLASS

		ARGS
		init_conds: initial data - may contain:
			int: 
				0,1 => generates a random number of masses with randomized locations up to (max_pts-1)
				2+  => generates (init_conds) number of masses with randomized locations
			float:
				cast to int, see above
			ndarray:
				[n x 7] => uses as initial masses, positions, and velocties
				[n x 6] => uses as initial positions and velocties, with random mass vector (see mass_equal below)
				[n x 4] => uses as initial massses and positions, with random velocity vector (see init_vel below)
				[n x 3] => uses as initial poisitons, with randoml mass and velocity vector (see below)
				[n x 1] => uses as initial masses, with random position and velocity vector (see below)
				[empty] => generates a random number of masses with randomized locations up to (max_pts-1)
			list:
				cast to ndarray, see above
		mass_equal: bool 
			1 => masses all contain equal mass fractions
			0 => masses contain random mass fractions
		init_vel: bool 
			1 => masses have non-zero initial velocity
			0 => masses have zero initial velocity
		max_states: int 
			maximum number of masses randomly generated

		STATE DESCRIPTION
		the entire state space is represented by an [n x 7] ndarray with each point containing
		[mass, x, y, z, vx, vy, vz]

		randomly generated initial states contain:
			x,y,z bounded within the unit sphere
			mass is represented as the normalized mass fraction
				all randomizedmasses sum to 1 and are constant in time

		TUNING PARAMETERS
		self.G: this is essentially the strength of gravity
			higher => raises the attraction between masses
			lower => lowers the attraction between masses
		self.S: this provides damping as masses approach each other
			higher => damps accelerations at close distances
			lower => allows acceleration to scale asymptotically
			0 => causes numerical errors, don't do this
		self.dt: time propegation step size
			higher => decreases expressiveness of model because the model is integrated over larger time steps
			lower => increases expressiveness of model at cost of computation time
			TBH I WOULDN"T TOUCH THIS IF I WERE YOU

		'''
		#CONSTANTS
		self.G = 1 #Gravitational constant, normalized
		self.S = 0.1 #softening
		self.t0 = 0 #start time [seconds]
		self.tf = run_time #[seconds]
		self.dt = 0.01 #timestep size [seconds]
		self.T = int(np.ceil(self.tf/self.dt)) #number of total time steps

		#get initial state information
		self.states = self.set_states(init_conds,mass_equal,init_vel,max_pts) #state of each point INCLUDING MASS
		self.n = self.states.shape[0] #number of masses
		self.mass = self.states[:,0].reshape((self.n,1)) #mass vector
		
		#initialize state tracking arrays
		self.acc_ = np.zeros((self.n,3,self.T+1)) #array containing accelerations at each timestep
		self.KE_ = np.zeros(self.T+1) #array containing kinetic energy at each timestep
		self.PE_ = np.zeros(self.T+1) #array containing potential energy at each timestep
		self.states_ = np.zeros((self.n,6,self.T+1)) #array containing state information at each timestep NOT INCLUDING MASS

		#transform velocities into the Center-of-Mass (COM) frame
		self.states[:,4:] -= np.sum(self.mass*self.states[:,4:],0)/np.sum(self.mass)

		#get initial accelerations
		self.acc = self.get_accelerations()

		#get initial energy of the system
		KE,PE = self.get_energy()

		#set initial states
		self.acc_[:,:,0] = self.acc
		self.KE_[0] = KE
		self.PE_[0] = PE
		self.states_[:,:,0] = self.states[:,1:]

		#testing
		#print(self.states)
		#print(self.states.shape)
		

	def set_states(self,init_conds,mass_equal,init_vel,max_pts=6):
		'''
		See __init__() for description of behavior

		'''
		#casts float to int
		if isinstance(init_conds,float):
			init_conds = int(init_conds)
		#casts list to ndarray
		if isinstance(init_conds,list):
			init_conds = np.array(init_conds)
		
		#generate random initial state space
		if isinstance(init_conds, int):
			init_conds = abs(init_conds)
			n = [init_conds if init_conds>1 else np.random.randint(2,max_pts)][0]#determines how many point masses
			mass_vec = self.get_mass_vec(n,mass_equal)
			states = self.get_rand_states(n,mass_vec,init_vel)
		
		#use user provided initial conditions, randomy generate missing information
		elif isinstance(init_conds,np.ndarray):
			try:
				#try loading all 7 state variables (mass, position3D, velocity3D)
				init_conds = init_conds.astype('float64')
				n = init_conds.shape[0]
				states = init_conds.reshape((n,7))
			except:
				try:
					#try loading 6 state variables (position3D, velocity3D)
					init_conds = init_conds.astype('float64')
					n = init_conds.shape[0]
					mass_vec = self.get_mass_vec(n,mass_equal) #generate random mass vector
					states = np.hstack((mass_vec,init_conds.reshape((n,6))))
				except:
					try:
						#try loading 4 state variables (mass, position3D)
						init_conds = init_conds.astype('float64')
						n = init_conds.shape[0]
						p = init_vel*(2*np.random.random_sample((n,3))-1)
						states = np.hstack((init_conds.reshape((n,4)),p))
					except:
						try:
							#try loading 3 state variables (position3D)
							init_conds = init_conds.astype('float64')
							n = init_conds.shape[0]
							mass_vec = self.get_mass_vec(n,mass_equal)
							p = init_vel*(2*np.random.random_sample((n,3))-1)
							states = np.hstack((mass_vec,init_conds.reshape((n,3)),p))
						except:
							try:
								#try loading 1 state variable (mass)
								init_conds = init_conds.astype('float64')
								n = init_conds.shape[0]
								mass_vec = init_conds.reshape((-1,1))
								states = self.get_rand_states(n,mass_vec,init_vel)
							except:
								#if all fail, load random initial conditions and notify user
								print("\nWARNING: INVALID INITIAL CONDITIONS\ncheck input array dimensions\ninitial conditions randomized... done.\n")
								n = np.random.randint(2,max_pts)
								mass_vec = self.get_mass_vec(n,mass_equal)
								states = self.get_rand_states(n,mass_vec,init_vel)
		else:
			#if init_conds contains unreadable initial conditions, load random initial conditions and notify user
			print("\nWARNING: INVALID INITIAL CONDITIONS\ncheck input array dimensions\ninitial conditions randomized... done.\n")
			n = np.random.randint(2,max_pts)
			mass_vec = self.get_mass_vec(n,mass_equal)
			states = self.get_rand_states(n,mass_vec,init_vel)

		return states


	def get_rand_states(self,n,mass_vec,init_vel,scale=10):
		'''
		Generates random state vector

		ARGS
		n: int
			number of masses to generate
		mass_vec: ndarray
			vector of masses in simulation
		init_vel: bool
			determines initial velocity of masses (nonzero if True)
		scale: int
			scaled uniformly distributed coordinates from unit sphere to scale*unit sphere

		RETURNS
		state: ndarray
			full [n x 7] initial state of the system

		'''
		p = init_vel*scale*(2*np.random.random_sample((n,3))-1)
		#return np.hstack((mass_vec,2*np.random.random_sample((n,3))-1,p)) #samples unit cube [-1,1]
		return np.hstack((mass_vec,scale*self.get_rand_spherical_coords_2_electric_boogaloo(n),p)) #samples unit sphere [r=1]

	def get_rand_spherical_coords(self,n):
		'''
		[DEPRECATED]
		Method for generating cartesian coordiantes uniformly distibuted in the unit n-ball via Method 22:
		http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

		ISSUE
		because we sample in d+2 dimension space we run into "the curse of dimensionality"
		i.e. the ratio of the volume between the unit sphere and unit cube falls off rapidly
		this means all our masses are distributed very close to zero - this works but is not what we want
		see get_rand_spherical_coords_2_electric_boogaloo() below for an improved solution
		'''
		d = 3 #unit sphere is 3 dims
		u = np.random.random_sample((d+2,n)) #5D space woo
		x = (u/np.sum(u**2)**.5)[0:d]#3D space aww
		return x.reshape((n,d))

	def get_rand_spherical_coords_2_electric_boogaloo(self,n):
		'''
		Method for generating cartesian coordiantes uniformly distibuted in the unit n-ball via Method 20:
		http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
		'''
		d = 3 #unit sphere is 3 dims
		u = np.random.normal(0,1,(d,n))
		norm = np.sum(u**2,0)**(0.5)
		r = np.random.random_sample(n)**(1/d)
		return (r*u/norm).reshape((n,d))



	def get_mass_vec(self,n,me):
		'''
		Method for generating random mass vectors
		don't just multipy by some coefficient to increase the strength of gravity
		I see you cheating
		go play god and turn up self.G in n_bpdy.__init__() like a self-respectiting physicist 

		ARGS
		n: number of masses
		me: bool
			"mass equal" determines whether to equalize masses

		RETURNS
		mass_vec: ndarray
			[n x 1] vector of masses

		'''
		#returns equal mass fractions over n states
		if me:
			return 100*np.ones((n,1))/n
		#generates random mass fractions summing to 1
		else:
			m = np.random.random_sample((n,1))
			return 100*m/np.sum(m)

	def get_accelerations(self):
		'''
		Gets the accelerations at each point via Newton's Law
		
		RETURNS
		accelerations: ndarray
			[n x 3] array of accelerations 
		
		adapted from code via Philip Mocz
		https://github.com/pmocz/nbody-python/blob/master/nbody.py
		'''
		x,y,z = [self.states[:,j].reshape((-1,1)) for j in range(1,4)]

		dx,dy,dz = [x.T-x,y.T-y,z.T-z]
		
		r3_inv = (dx**2 + dy**2 + dz**2 + self.S**2)
		r3_inv[r3_inv>0] = r3_inv[r3_inv>0]**(-3/2)
		#print(r3_inv.shape)

		ax = self.G*(dx*r3_inv)@self.mass
		ay = self.G*(dy*r3_inv)@self.mass
		az = self.G*(dz*r3_inv)@self.mass

		return np.hstack((ax,ay,az))

	def get_energy(self):
		'''
		Gets the Kinetric and Potential energy of the system 

		RETURNS
		[KE, PE]: list containing:
			KE: ndarray
			PE: ndarray

		adapted from code via Philip Mocz
		https://github.com/pmocz/nbody-python/blob/master/nbody.py
		'''
		#get Kinetic Energy .5*m*v**2
		KE = np.sum(self.mass*self.states[:,4:]**2)/2

		#get Potential Energy G*m1*m2/r**2
		x = self.states[:,1].reshape((-1,1))
		y = self.states[:,2].reshape((-1,1))
		z = self.states[:,3].reshape((-1,1))

		dx,dy,dz = [x.T-x,y.T-y,z.T-z]

		r_inv = np.sqrt(dx**2 + dy**2 + dz**2)
		r_inv[r_inv>0] = 1/r_inv[r_inv>0]

		PE = self.G*np.sum(np.sum(np.triu(-(self.mass*self.mass.T)*r_inv,1)))

		return [KE,PE]

	def simulate(self):
		'''
		Main simulation loop using kick-drift-kick propegation

		'''
		#start = time.time()
		print("\n")
		print("RUNNING SIMULATION")
		for i in range(self.T):
			#apply half kick to velocity
			self.states[:,4:] += self.acc*self.dt/2.0

			#drift to new position
			self.states[:,1:4] += self.states[:,4:]*self.dt

			#update accelerations
			self.acc = self.get_accelerations()

			#apply half kick to velocity with updated acceleration
			self.states[:,4:] += self.acc*self.dt/2.0

			#get system energy
			[KE,PE] = self.get_energy()

			#save system states
			self.acc_[:,:,i+1] = self.acc
			self.KE_[i+1] = KE
			self.PE_[i+1] = PE
			self.states_[:,:,i+1] = self.states[:,1:]

		#end = time.time()
		#print("ELAPSED TIME: ",end-start)
		#print("AVG TIME PER LOOP: ",(end-start)/self.T)
		
		#testing
		#print(self.states_[:,:,0]) #initial state
		#print(self.states_[:,:,-1]) #final state

	def plot(self,save=100,autoscroll=False,replay=False,view_lim=20):
		'''
		Simulation visualization method

		ARGS
		view_lim: int 
			sets range [-view_lim, view_lim] for visualization 
		save: int
			0 => do not save
				Warning: if save and replay are both False, the code won't do anything
			else => save file at [save] frames per second
				anything above 30 fps looks fine, 60-100 fps is ideal IMO
				this will largely determine how strong you set gravity
				the stronger you set gravity/the stronger the interactions -> the faster the masses will move -> the slower you'll want playback
		autoscroll: bool
			True => visualization automatically scrolls to keep all data in frame
				it looks pretty bad NGL
			False => centers plot on [-view_lim,view_lim] from the COM Reference Frame
		replay: bool
			True => replays simulation frame by frame
				good for testing initial confirgurations before setting up longer runs
			False => does not replay simulation
				Warning: if save and replay are both False, the code won't do anything
		'''
		#setup visualization
		plt.style.use('dark_background')
		self.fig = plt.figure(figsize=(10,10), dpi=80)
		self.fig.suptitle('N-Body Dynamics', fontsize=26)
		grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
		
		#create mass visualization axis
		self.ax1 = plt.subplot(grid[0:2,0])
		self.ax1.set(xlim=(-view_lim, view_lim), ylim=(-view_lim, view_lim))
		self.ax1.set_axis_off() #turn off axes to better see motion

		#automatically set axis bounds
		yl = self.get_bound()

		#create energy visualization axes
		self.ax2 = plt.subplot(grid[2,0])
		self.ax2.set(xlim=(0, self.T), ylim=(-yl, yl))
		self.ax2.set_xticks([0,self.T/4,self.T/2,3*self.T/4,self.T])
		self.ax2.set_yticks([-yl,-yl/2,0,yl/2,yl])
		self.ax2.set_xlabel("Time [dt]")
		self.ax2.set_ylabel("Energy")

		ti = 100 #tail length

		#define color scheme for masses and tails
		cs = ['tab:blue','tab:red','tab:green','tab:purple','tab:orange','tab:pink','tab:cyan']
		cs_pts = [cs[j%len(cs)] for j in range(self.n)]
		cs_tail = [cs_pts[j%len(cs_pts)] for j in range(self.n) for k in range(ti)]

		#define marker sizes for masses and tails
		s = np.linspace(8,16,ti).reshape((ti,))
		s_tail = np.vstack(tuple([s for p in range(self.n)]))
		s_pts = self.normalize_mass()
		

		#create collections objects for animation
		trails = self.ax1.scatter([],[],c=cs_tail,s=s_tail,zorder=1)
		pts = self.ax1.scatter([],[],c=cs_pts,s=s_pts,zorder=2)

		KE_line, = self.ax2.plot([], [],'b',lw=2)
		PE_line, = self.ax2.plot([], [],'darkorange',lw=2)
		E_line, = self.ax2.plot([], [],'lawngreen',lw=2)


		def init():
			'''
			Initialize animation
			'''
			KE_line.set_data([],[])
			PE_line.set_data([],[])
			E_line.set_data([],[])

			self.KE_step = []
			self.PE_step = []
			self.E_step = []

			return pts, trails, KE_line, PE_line, E_line,

		def animate(i):
			'''
			Animation loop

			NOTE: no actual computation is occuring during animation
			all values are precomputed during the n_body.simulate() routine

			'''
			#draw tails
			if i-ti<0:
				trail_i = np.vstack(tuple([np.vstack((self.states_[j,:2,0]*np.ones((ti-i,2)),self.states_[j,:2,max(i-ti,0):i].T)) for j in range(self.n)]))
			else:
				trail_i = np.vstack(tuple([self.states_[j,:2,max(i-ti,0):i].T for j in range(self.n)]))
			trails.set_offsets(trail_i)

			#draw masses
			xy = [self.states_[j,:2,i].T for j in range(self.n)]
			pts_i = np.vstack(tuple(xy))
			pts.set_offsets(pts_i)

			#get energy
			KE = self.KE_[i]
			PE = self.PE_[i]
			E = KE+PE

			#save energy states
			self.KE_step.append(KE)
			self.PE_step.append(PE)
			self.E_step.append(E)
			t_steps = np.linspace(0,i,len(self.E_step))

			#draw energy curves
			KE_line.set_data(t_steps,self.KE_step)
			PE_line.set_data(t_steps,self.PE_step)
			E_line.set_data(t_steps,self.E_step)

			#sets autoscroll
			if autoscroll:
				#cast xy to an array and take its transpose
				xyT = np.array(xy).T #new dims [2 x n]

				#set the maximum scrolling behavior
				xymax = max(np.amax(xyT[1]),np.amax(xyT[0]))
				xymax_lim = max(view_lim,max(1.1*xymax,xymax+5)) #applies additive scrolling near the origin and mutiplicative scrolling farther away

				#set the minimum scrolling behavior
				xymin = min(np.amin(xyT[1]),np.amin(xyT[0]))
				xymin_lim = min(-view_lim,min(-1.1*xymin,xymin-5)) #see above

				#apply autoscroll
				self.ax1.set(xlim=(xymin_lim,xymax_lim),ylim=(xymin_lim,xymax_lim))

			return pts, trails, KE_line, PE_line, E_line,

		#animate plots
		ani = animation.FuncAnimation(self.fig,animate,frames=self.T,interval=1,blit=True,init_func=init,repeat=False)
		
		#apply legends
		#self.ax1.legend(self.make_mass_legend()) #not currently working
		self.ax2.legend(("KE","PE","E"))
		
		#show plot as it is calculated
		if replay:
			plt.show()

		#saves simulation run
		if save:
			#set up formatting
			Writer = animation.writers['ffmpeg']
			writer = Writer(fps=save, metadata=dict(artist='Kyle Morgenstein'), bitrate=1800)
			now = datetime.datetime.now()
			date_time = now.strftime("%m_%d_%Y")
			date_time_s = now.strftime("%H_%M_%S")
			sv = "n_body_trials/"+"simulation_"+str(date_time)+"__"+str(date_time_s)+".mp4"
			
			print("SAVING SIMULATION AS....      ","./"+sv)
			ani.save(sv, writer=writer)
			print("done.")
		print("\n")
		print("SIMULATION COMPLETE")
		print("\n")


	def run(self):
		'''
		Wrapper function for the simulation

		Provided for ease of use, but I recommend just running it step by step
		'''
		self.simulate()
		self.plot()
		

	# =====================================================================
	# HELPER FUNCTIONS
	# =====================================================================

	def make_mass_legend(self):
		'''
		Method for generating legend markers for each mass

		Not currently used
		Functionality will be provided in a future push

		'''
		mass_string = []
		mass_str = "Mass: "
		for m in range(self.n):
			mass_string.append(mass_str+str(self.mass[m][0])[:5])
		return mass_string

	def get_bound(self):
		'''
		Method to automatically scale energy plot bounds
		'''
		#get the largest energy reading from the simulated run and add 1
		b = int(max(np.amax(np.abs(self.KE_)),np.amax(np.abs(self.PE_))))+1

		#if less than 10, bound to the max energy + 1
		if b<10:
			return b
		#if less than 100, bound to the nearest 20
		elif b<100:
			return (b+19)//20*20
		#if less than 1000 bound to the nearest 50
		elif b<1000:
			return (b+49)//50*50
		#else bound to the nearest 100
		else:
			return (b+99)//100*100

	def normalize_mass(self):
		'''
		Method to scale visulaization marker areas to mass

		'''
		#get mass vector
		m = self.mass

		#get the average mass
		m_avg = np.ones((m.shape))*np.sum((m**2)**.5)/m.shape[0]

		#get the standard deviation
		m_std = (np.std(m)**2)**.5
		
		#account for std ~= 0 (happens when all masses are equal)
		if m_std<0.0001:
			return 225 #marker size is determined by area i.e. 225 => size 15
		
		#get z-score for each mass
		m_score = (m-m_avg)/m_std

		#bound minimum size marker
		m_score[np.sign(m_score)*(10*m_score)**2<-144] = -144 #225-144 = 81 => size 9 minimumm

		#z score scaled to enforce distribution of marker sizes
		return 225+np.sign(m_score)*(10*m_score)**2

	
if __name__=="__main__":
	
	#set initial conditions
	init_conds = np.array([
		[100,0,0,0,0,0,0],
		[1,0,-3,0,3,4,0],
		[1,0,3,0,-3,-4,0],
		[1,-3,0,0,0,-4,0],
		[1,3,0,0,0,4,0]
		])

	#set experiment runtime
	run_time = 30 #[seconds]

	#initialize simulation class
	sim = n_body(init_conds,run_time)

	#run simulation
	sim.simulate()

	#plot the results
	save = 100 #sets framerate for saved simulation - set to 0 to playback only
	autoscroll = False #automatically adjusts view to keep all masses in view
	replay = False #better to just generate the video and watch it at full speed
	view_lim = 5 #scales view to [-view_lim, view_lim]

	sim.plot(save,autoscroll,replay,view_lim)
