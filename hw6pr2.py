"""
Starter file for hw6pr2 of Big Data Summer 2017

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

Note:
1. When filling out the functions below, note that
	1) Let k be the rank for approximation

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import urllib

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	# NOTE: Be sure to install Pillow with "pip3 install Pillow"
	print('==> Loading image data...')
	img = ndimage.imread(urllib.request.urlopen('http://i.imgur.com/X017qGH.jpg'), flatten=True)

	"*** YOUR CODE HERE ***"
	shuffle_img = img.copy().flatten()
	np.random.shuffle(shuffle_img)
	"*** END YOUR CODE HERE ***"
	# reshape the shuffled image
	shuffle_img = shuffle_img.reshape(img.shape)

	# =============STEP 1: RUNNING SVD ON IMAGES=================
	print('==> Running SVD on images...')

	"*** YOUR CODE HERE ***"
	U, S, V = np.linalg.svd(img)
	U_s, S_s, V_s = np.linalg.svd(shuffle_img)
	"*** END YOUR CODE HERE ***"

	# =============STEP 2: SINGULAR VALUE DROPOFF=================
	print('==> Singular value dropoff plot...')
	k = 100
	plt.style.use('ggplot')

	"*** YOUR CODE HERE ***"
	orig_S_plot, = plt.plot(S[:k], 'g')
	shuf_S_plot, = plt.plot(S_s[:k], 'r')
	"*** END YOUR CODE HERE ***"

	plt.legend((orig_S_plot, shuf_S_plot), \
		('original', 'shuffled'), loc = 'best')
	plt.title('Singular Value Dropoff for Clown Image')
	plt.ylabel('singular values')
	plt.savefig('dropoff.png', format='png')
	plt.close()

	# =============STEP 3: RECONSTRUCTION=================
	print('==> Reconstruction with different ranks...')
	rank_list = [2, 10, 20]
	plt.subplot(2, 2, 1)
	plt.imshow(img, cmap='Greys_r')
	plt.axis('off')
	plt.title('Original Image')

	for index in range(len(rank_list)):
		k = rank_list[index]
		plt.subplot(2, 2, 2 + index)

		"*** YOUR CODE HERE ***"
		U_k = U[:, :k] 
		S_k = np.diag(S)[:k, :k]
		V_k = V[:k, :]
		reconstructed_img = U_k @ S_k @ V_k
		plt.imshow(reconstructed_img, cmap='Greys_r')
		"*** END YOUR CODE HERE ***"

		plt.title('Rank {} Approximation'.format(k))
		plt.axis('off')

	plt.tight_layout()
	plt.savefig('reconstruction.png', format='png')
	plt.close()
