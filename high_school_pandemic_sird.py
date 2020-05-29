import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import cv2


COLOR_DICT = {'white': "\x1b[30m", 'red': "\x1b[31m", 'green': "\x1b[32m",
			  'yellow': "\x1b[33m", 'blue': "\x1b[34m", 'violet': "\x1b[35m",
			  'azul': "\x1b[36m", 'black': "\x1b[37m", 'neuter': "\x1b[38m"}


# A nice separator with a color option
def segments(color='yellow'):
	print(COLOR_DICT[color] +
		  '---------------------------------------------' +
		  COLOR_DICT['neuter'])


def curtigghiuBuilder(spirtizza=0.25, min_rating=1):
	"""
	Network builder. "Curtigghiu" is a sicilian word which means "rumors" or "gossip", used here because the network
	is made of rated social relationships. "spirtizza" is another sicilian word which means "cleverness",
	used here in a sarcastic way (people who avoid the lockdown are indeed "very clever")
	:param spirtizza: float [0, 1] defining the probability of a forbidden social link to exist
	:param min_rating: threshold defining which links are forbidden (rating < min_rating)
	:return: None, saves the network as pickle
	"""
	el = pd.read_csv("./social_data/out.moreno_health_health", header=None, skiprows=2, sep=' ').values.astype(int)
	path = f"./curtigghiu-nets/"
	segments('blue')
	print(f'Generating curtigghiu-nets (spirtizza = {spirtizza})')
	segments('azul')
	if not os.path.exists(path):
		os.makedirs(path)
	G = nx.DiGraph()
	for line in el:
		if line[2] >= min_rating or np.random.choice([True, False], p=[spirtizza, 1 - spirtizza]):
			G.add_edge(line[0], line[1], affection=line[2])
	print(f"Minimum stars: {min_rating} | # of components: {len(list(nx.connected_components(G.to_undirected())))}")
	nx.write_gpickle(G, f"./{path}curtigghiu-net_{spirtizza}-spirtizza_{min_rating}-stars.pickle")
	segments('blue')


def sirdPlotter(infected, recovered, dead, path, n_frame, spirtizza, rating, steps):
	"""
	Frame-by-frame plotter
	:param infected: # of infected people
	:param recovered: # of recovered people
	:param dead: # of dead people
	:param path: save folder
	:param n_frame: n_frame
	:param spirtizza: explained above
	:param rating: int 1-6
	:param steps: x upper limit
	:return: None, saves png frame
	"""
	time = np.arange(0, len(infected), 1)
	peak = np.max(infected)
	figa, ax = plt.subplots(1, 1, figsize=(14, 10))
	ax.plot(time, infected, color='xkcd:tangerine', zorder=50)
	ax.plot(time, dead, color='xkcd:scarlet', zorder=50)
	ax.plot(time, recovered, color='xkcd:leaf green', zorder=50)
	ax.scatter(time[-1], infected[-1], s=64, color='xkcd:tangerine', label='infected', zorder=90)
	ax.scatter(time[-1], dead[-1], s=64, color='xkcd:scarlet', label='dead', zorder=90)
	ax.scatter(time[-1], recovered[-1], s=64, color='xkcd:leaf green', label='recovered', zorder=90)
	ax.axhline(y=peak, color='xkcd:charcoal', linestyle='--', zorder=10)
	ax.annotate(f"peak = {peak}", (steps, peak + 75), ha='right', va='top', fontsize=14)
	ax.set_xlim(0, steps)
	ax.set_ylim(0, 2200)
	ax.set_xlabel('Time-steps', fontsize=21)
	ax.set_ylabel('# of people', fontsize=21)
	ax.set_title(f"Friendship level = {rating}+ | Cheating probability = {spirtizza}", fontsize=24)
	ax.legend(loc='upper left', fancybox=True, facecolor='xkcd:eggshell', edgecolor='xkcd:scarlet', fontsize=17)
	plt.savefig(f"{path}{n_frame}.png")
	plt.close(figa)


def pandemicSIRD(spirtizza, min_affection, i_p, h_p, d_p, starters=1, max_steps=None):
	"""
	Heart of the code. This function simulates the epidemic and makes a video from the png frames.
	:param spirtizza: explained above
	:param min_affection: int 1-6
	:param i_p: infection probability
	:param h_p: healing probability
	:param d_p: death probability
	:param starters: # of infected at time zero
	:param max_steps: max steps
	:return: None, saves a video
	"""
	inpath = f"./curtigghiu-nets/"
	G = nx.read_gpickle(f"./{inpath}curtigghiu-net_{spirtizza}-spirtizza_{min_affection}-stars.pickle")
	G = G.to_undirected()
	path = f"./curtigghiu_frames/{spirtizza}-spirtizza_{min_affection}-stars/"
	v_path = f"./curtigghiu_videos/{spirtizza}-spirtizza_{min_affection}-stars.avi"
	if not os.path.exists(path):
		os.makedirs(path)
	if not os.path.exists(f"./curtigghiu_videos/"):
		os.makedirs(f"./curtigghiu_videos/")
	if starters >= len(G.nodes) or starters == 0:
		print('No spreading is possible :(')
		exit()
	infected = np.random.choice(list(G.nodes), size=starters)
	pandemic = True
	n_steps = 0
	N = len(G.nodes)
	tabbutu = []
	immuni = []
	I = []
	R = []
	D = []
	img_array = []
	if not max_steps:
		max_steps = np.inf
		steps = 15000
	else:
		steps = max_steps
	while pandemic:
		n_steps += 1
		maskull = np.random.choice([True, False], size=infected.shape[0], replace=True, p=[d_p, 1 - d_p])
		maskerina = np.random.choice([True, False], size=infected.shape[0], replace=True, p=[h_p, 1 - h_p])
		guariti = infected[maskerina]
		dead = infected[maskull]
		infected = np.setdiff1d(infected, dead)
		infected = np.setdiff1d(infected, guariti)
		G.remove_nodes_from(dead)
		G.remove_nodes_from(guariti)
		tabbutu.extend(dead)
		immuni.extend(guariti)
		congiunti = [node for i in infected for node in nx.neighbors(G, i)]
		congiunti = np.setdiff1d(congiunti, infected)
		mask = np.random.choice([True, False], size=congiunti.shape[0], replace=True, p=[i_p, 1 - i_p])
		infected = np.concatenate((infected, congiunti[mask]))
		I.append(infected.shape[0])
		R.append(len(immuni))
		D.append(len(tabbutu))
		if infected.shape[0] == 0 or n_steps >= max_steps:
			pandemic = False
		if n_steps % 50 == 0:
			print(f"Bulletin: {infected.shape[0]} infected ({100 * infected.shape[0] // N}%),"
				  f"{len(tabbutu)} dead ({100 * len(tabbutu) // N}%) e {len(immuni)} recovered "
				  f"({100 * len(immuni) // N}%) [{n_steps} steps]")
			sirdPlotter(I, R, D, path=path, n_frame=n_steps, spirtizza=spirtizza, rating=min_affection, steps=steps)
			img = cv2.imread(f"{path}{n_steps}.png")
			if img is not None:
				img_array.append(img)
			else:
				raise Exception(f'Not Existing Iteration :(')
	height, width, _ = img_array[0].shape
	size = (width, height)
	out = cv2.VideoWriter(v_path, cv2.VideoWriter_fourcc(*'XVID'), 30, size)
	for frame in img_array:
		out.write(frame)
	out.release()
	cv2.destroyAllWindows()
	segments('azul')
	print(f"Final Bulletin: {infected.shape[0]} infected ({100 * infected.shape[0] // N}%), "
		  f"{len(tabbutu)} dead ({100 * len(tabbutu) // N}%) e {len(immuni)} recovered "
		  f"({100 * len(immuni) // N}%) [{n_steps} steps, {len(G.nodes)} non infected]")


if __name__ == '__main__':
	curtigghiuBuilder(0., min_rating=1)
	pandemicSIRD(spirtizza=0., min_affection=1, i_p=1 * 1e-3, h_p=0.5 * 1e-3, d_p=0.075 * 1e-3, starters=1)
	for stars in [2, 3, 4, 5, 6]:
		for spz in [0., 0.2, 0.3, 0.5, 0.75]:
			curtigghiuBuilder(spz, min_rating=stars)
			pandemicSIRD(spirtizza=spz, min_affection=stars, i_p=1*1e-3, h_p=0.5*1e-3, d_p=0.075*1e-3, starters=1)
