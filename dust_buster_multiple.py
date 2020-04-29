import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

# 4 possible moves of the dust particle: left, up, right, down
moves = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}


class DustBuster:
	"""
	The buster class is the colored blob that keeps absorbing all the brownian dust.
	This is the multiple version of the code, so it takes a list of 2D numpy ndarrays as zero_surface
	and a list of matplotlib colors with same length of  zero_surface.
	It has a reset method, a load method and an absorb method that applies the busted function to the dust.
	"""
	def __init__(self, zero_surface: list, colors: list):
		if len(zero_surface) != len(colors):
			raise Exception("The lenght of zero_surface and color is different :(")
		self.zero = zero_surface
		self.colors = colors
		self.set = self.zero
		self.dead = Pool(cpu_count())
		self.busted = [partial(busted, s.T) for s in self.set]

	def reset(self):
		self.set = self.zero
		self.busted = [partial(busted, s.T) for s in self.set]

	def load(self, loaded_set):
		self.set = loaded_set
		self.busted = [partial(busted, s.T) for s in self.set]

	def absorb(self, points: np.ndarray, old_points: np.ndarray):
		"""
		This function creates a boolean mask of the points, checking if they have been captured by the Buster,
		then adds the previous position of the captured dust to the Buster set (the actual one is inside!), making it bigger.
		:param points: 2D ndarray containing the actual position of the dust particles.
		:param old_points: 2D ndarray containing the previous position of the dust particles.
		:return: 2D ndarray containing the actual position of points which have not been captured by the Buster.
		"""
		for i in range(len(self.set)):
			mask = np.array(self.dead.map(self.busted[i], points))
			self.set[i] = np.append(self.set[i], old_points[mask], axis=0)
			self.busted[i] = partial(busted, self.set[i].T)
			old_points = old_points[np.invert(mask)]
			points = points[np.invert(mask)]
		return points


def beADict(idx):  # transforms a random int in range 0-3 into a move
	return moves[idx]


def busted(buster, x):  # function that checks if the dust particle is inside the Buster
	if np.intersect1d(np.argwhere(buster[0] == x[0]), np.argwhere(buster[1] == x[1])).shape[0] == 0:
		return False
	else:
		return True


class Space:
	"""
	Main class of the program. It contains information about the dust and the buster. It has a reset method, a save method,
	a load method, a step method (brown), a plot method and an animate method.
	"""
	def __init__(self, radius: int, particles: int, dust_speed: int, buster_zero: list, buster_color: list,
				 save_dir='./static_multiple/', dust_color='xkcd:steel grey', space_color='xkcd:black', video_name='space'):
		"""
		:param radius: apotheme of the square space.
		:param particles: # of dust particles.
		:param dust_speed: speed of the dust particles (squares per move).
		:param buster_zero: zero_surfaces of the Buster.
		:param buster_color: color-list of the buster.
		:param save_dir: save directory.
		:param dust_color: color of the dust particles.
		:param space_color: background color.
		:param video_name: name of the various files.
		"""
		self.limits = [[-radius, radius], [-radius, radius]]
		self.radius = radius
		self.color = dust_color
		self.background = space_color
		self.particles = particles
		self.speed = dust_speed
		self.dust = np.array([[np.random.randint(-self.radius, self.radius),
							   np.random.randint(-self.radius, self.radius)] for _ in range(self.particles)])
		self.buster = DustBuster(buster_zero, buster_color)
		self.old_dust = self.dust
		self.dust = self.buster.absorb(self.dust, self.old_dust)
		self.save_dir = save_dir
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		self.dead = Pool(cpu_count())
		self.info = ''
		self.video_name = video_name
		self.n_steps = 0

	def reset(self):
		self.dust = np.array([[np.random.randint(-self.radius, self.radius),
							   np.random.randint(-self.radius, self.radius)] for _ in range(self.particles)])
		self.old_dust = self.dust
		self.buster.set = self.buster.zero
		self.dust = self.buster.absorb(self.dust, self.old_dust)
		self.n_steps = 0
		self.info = f'Step: {self.n_steps} | Buster: {np.sum([b.shape[0] for b in self.buster.set])} | Dust: {self.dust.shape[0]}'

	def save(self):
		bundle = [self.dust, self.old_dust, self.buster.set, self.n_steps]
		pickle.dump(bundle, open(f'{self.save_dir}{self.video_name}.pickle', "wb"))

	def load(self):
		bundle = pickle.load(open(f'{self.save_dir}{self.video_name}.pickle', "rb"))
		self.dust = bundle[0]
		self.old_dust = bundle[1]
		self.buster.load(bundle[2])
		self.n_steps = bundle[3]

	def brown(self):  # makes a random step for each dust particle
		self.old_dust = self.dust
		indices = np.random.randint(4, size=self.dust.shape[0])
		step = np.array(self.dead.map(beADict, indices))
		self.dust = self.dust + step * self.speed
		self.dust = self.buster.absorb(self.dust, self.old_dust)
		self.n_steps += 1
		self.info = f'Step: {self.n_steps} | Buster: {np.sum([b.shape[0] for b in self.buster.set])} | Dust: {self.dust.shape[0]}'

	def plot(self, save_name=None, show=False):  # just matplotlib here
		figa, ax = plt.subplots(1, 1, figsize=(14, 14))
		ax.set_facecolor(self.background)
		figa.set_facecolor(self.color)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.scatter(*self.dust.T, color=self.color, s=0.5)
		for s in range(len(self.buster.set)):
			ax.scatter(*self.buster.set[s].T, color=self.buster.colors[s], s=0.5)
		ax.set_xlim(-self.radius, self.radius)
		ax.set_ylim(-self.radius, self.radius)
		if save_name:
			plt.savefig(f'{self.save_dir}{save_name}', bbox_inches='tight', dpi=150)
		if show:
			plt.show()
		plt.close(figa)

	def animate(self, iters=100, frequency=1, load=False):
		"""
		The function plots a frame each frequency and saves it in the img_folder. At the end of the process, it joins all of
		them in a video using the cv2 library. It stops at iters and if load is True, it loads both the Space and the images.
		:param iters: number of total steps
		:param frequency: frequency of captured steps. Example: if frequency=10 it renders a frame every 10 steps.
		:param load: if True it loads the model, otherwise it resets it.
		:return: None, it saves a video and a bunch of frames.
		"""
		img_array = []
		if not os.path.exists(f"{self.save_dir}{self.video_name}/"):
			os.makedirs(f"{self.save_dir}{self.video_name}/")
		if load:
			S.load()
			for i in range(0, S.n_steps, frequency):
				img = cv2.imread(f'{self.save_dir}{self.video_name}/{i}.png')
				if img is not None:
					img_array.append(img)
				else:
					raise Exception(f'[{i}] Not Existing Iteration :(')
		else:
			S.reset()
			img_path = f"{self.video_name}/0.png"
			self.save()
			self.plot(save_name=img_path)
			img = cv2.imread(f'{self.save_dir}{img_path}')
			if img is not None:
				img_array.append(img)
			else:
				raise Exception(f'[0] Not Existing Iteration :(')
		while self.n_steps <= iters:
			S.brown()
			if self.n_steps % frequency == 0:
				img_path = f"{self.video_name}/{self.n_steps}.png"
				self.save()
				self.plot(save_name=img_path)
				img = cv2.imread(f'{self.save_dir}{img_path}')
				if img is not None:
					img_array.append(img)
				else:
					raise Exception(f'[{self.n_steps}] Not Existing Iteration :(')
				print(S.info)
		height, width, _ = img_array[0].shape
		size = (width, height)
		out = cv2.VideoWriter(f'{self.save_dir}{self.video_name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)
		for frame in img_array:
			out.write(frame)
		out.release()
		cv2.destroyAllWindows()


#########################################################
# HERE THERE ARE A FEW ZERO_SURFACES.                   #
# THIS IS THE MULTIPLE VERSION OF THE CODE,             #
# SO YOU CAN'T USE THE NDARRAY ONES DIRECTLY,           #
# BUT YOU CAN USE THEM TO COMPOSE MORE COMPLEX SHAPES,  #
# OR JUST WRAP THEM IN A 1-ELEMENT LIST.                #
#########################################################


def circleSet(radius, center_x, center_y):  # SINGLE! CAN'T USE IT DIRECTLY
	points = np.array([[x, y] for y in range(center_y - radius, center_y + radius)
					   for x in range(- int(np.sqrt(radius ** 2 - (y - center_y) ** 2)) + center_x,
									  int(np.sqrt(radius ** 2 - (y - center_y) ** 2)) + center_x)])
	return points


def protonSet(a, radius):
	points = [circleSet(radius, 0, a), circleSet(radius, int(a * 3 ** 0.5) // 2, - a // 2),
			  circleSet(radius, - int(a * 3 ** 0.5) // 2, - a // 2)]
	return points


def circumferenceSet(radius, center_x, center_y):  # SINGLE! CAN'T USE IT DIRECTLY
	points = np.array([[x, int(np.sqrt(radius ** 2 - (x - center_x) ** 2)) + center_y]
					   for x in range(center_x - radius, center_x + radius)])
	points = np.concatenate((points, np.array([[x, - int(np.sqrt(radius ** 2 - (x - center_x) ** 2)) + center_y]
											   for x in range(center_x - radius, center_x + radius)])))
	points = np.concatenate((points, np.array([[+ int(np.sqrt(radius ** 2 - (y - center_y) ** 2)) + center_x, y]
											   for y in range(center_y - radius, center_y + radius)])))
	points = np.concatenate((points, np.array([[- int(np.sqrt(radius ** 2 - (y - center_y) ** 2)) + center_x, y]
											   for y in range(center_y - radius, center_y + radius)])))
	return points


def axisSet(slope, radius, q=0):  # SINGLE! CAN'T USE IT DIRECTLY
	points = np.array([[x, slope * x + q] for x in range(-radius, radius)]).astype(int)
	return points


def xSet(angle, radius):  # SINGLE! CAN'T USE IT DIRECTLY
	points = np.concatenate((axisSet(np.tan(angle), radius), axisSet(- np.tan(angle), radius)))
	return points


def logoSet(a, radius, big_radius):
	points = [circleSet(radius, 0, a), circleSet(radius, int(a * 3 ** 0.5) // 2, - a // 2),
			  circleSet(radius, - int(a * 3 ** 0.5) // 2, - a // 2), circumferenceSet(big_radius, 0, 0)]
	return points


if __name__ == '__main__':
	# Here how to make our logo animation :D
	S = Space(300, 67500, 1, logoSet(42, 14, 200), dust_color="xkcd:dark", video_name='seemulate_physics',
			  buster_color=["xkcd:fire engine red", "xkcd:electric green", "xkcd:electric blue", "xkcd:marigold"])
	S.animate(iters=15000, frequency=50, load=False)
