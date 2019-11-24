import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# coordinates of rectangle drawn on image
ref_point = []

def shape_selection(event, x, y, flags, param):
	'''
	Function for drawing rectangle
	'''
	global ref_point

	if event == cv2.EVENT_LBUTTONDOWN: 
		ref_point = [(x, y)] 

	elif event == cv2.EVENT_LBUTTONUP: 
		ref_point.append((x, y)) 
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

def do_patches(nnf, inp1, inp2, siz):
	'''
	Copy best matching patches to input image
	nnf: Nearest neighbour field
	inp1: Input image
	inp2: Reference image
	siz: Patch size
	'''
	inp_shape = inp1.shape
	w = int((siz - 1) / 2)
	out = np.zeros(inp_shape, np.float)

	for i in range(w, inp_shape[0] - w, siz):
		for j in range(w, inp_shape[1] - w, siz):
			
			x = nnf[0][i][j]
			y = nnf[1][i][j]

			temp = inp2[x - w: x + w + 1, y - w: y + w + 1]
			out[i - w: i + w + 1, j - w: j + w + 1] = temp

	out = np.uint8(out)
	return out
	

def nearestnf(inp1, inp2, siz, iterations):
	'''
	This function computes nearest neighbour field followed by propagation and random search process.
	inp1: Input image
	inp2: Reference image
	siz: Patch size
	iterations: Number of iterations for which algorithm runs
	'''

	inp1, inp2 = np.array(inp1, np.float), np.array(inp2, np.float)
	w = int((siz - 1) / 2)
	
	inp_shape = np.shape(inp1)
	old_sz = inp_shape
	
	# create rectangle divisible by patch size
	new_inp_shape = np.zeros(len(inp_shape))
	new_inp_shape[0] = inp_shape[0] + siz - inp_shape[0] % siz
	new_inp_shape[1] = inp_shape[1] + siz - inp_shape[1] % siz
	
	# preserve the 3rd dimension if colored image
	for i in range(2, len(inp_shape)):
		new_inp_shape[i] = inp_shape[i]
	
	new_inp_shape = np.uint(new_inp_shape)
	
	new_inp = np.zeros(new_inp_shape)
	new_inp[:inp_shape[0], :inp_shape[1]] = inp1[:inp_shape[0], :inp_shape[1]]
	new_inp = np.uint8(new_inp)
	
	inp1 = np.copy(new_inp)
	inp_shape = new_inp_shape
	ref_shape = np.shape(inp2)
	
	# outx if NNF containing x coordinates of reference image
	outx = np.random.randint(w, ref_shape[0] - w, (inp_shape[0], inp_shape[1]))
	# outy if NNF containing y coordinates of reference image
	outy = np.random.randint(w, ref_shape[1] - w, (inp_shape[0], inp_shape[1]))
	
	pad_image = np.pad(inp1, ((w,w),(w,w),(0,0)), 'constant', constant_values=(np.nan, np.nan)) # padded image

	off = np.full((inp_shape[0], inp_shape[1]), np.inf) # offset array which error metric between two patches
	
	#initial copmutation of offsets
	for i in range(inp_shape[0]):
		for j in range(inp_shape[1]):
			x = outx[i, j]
			y = outy[i, j]
			a = pad_image[i: i + siz, j: j + siz, :]
			b = inp2[x - w: x + w + 1, y - w: y + w + 1, :]
			temp = a - b 
			temp = temp[~np.isnan(temp)]
			temp2 = np.sum(temp ** 2) / len(temp)
			off[i, j] = temp2

	# Initial NNF
	final = do_patches([outx, outy], inp1, inp2, siz)
	final = final[:old_sz[0], :old_sz[1]]

	plt.subplot(331)
	plt.axis('off')
	plt.imshow(final)
	plt.title("Initial")


	for itr in range(iterations):
		if itr % 2 == 0:
			tot = inp_shape[0] * inp_shape[1]
			tot = int(tot / 4)
			ctr = 0			
			# Scan Order: Left to Right, Top to Bottom
			for i in range(inp_shape[0]):
				for j in range(inp_shape[1]):

					# Propagation:
					cur = off[i][j] #current patch
					left = off[max(i - 1, 0)][j] # left patch
					top = off[i][max(j - 1, 0)] # top patch
					mn = min(cur, left, top) # best match patch from above three
					
					if mn == left:
						x = outx[i - 1][j] + 1
						y = outy[i - 1][j]
						if x < ref_shape[0] - w and y < ref_shape[1] - w:
							outx[i, j] = x
							outy[i, j] = y
							a = pad_image[i: i + siz, j: j + siz, :]
							b = inp2[x - w: x + w + 1, y - w: y + w + 1, :]
							temp = a - b 
							temp = temp[~np.isnan(temp)]
							temp2 = np.sum(temp ** 2) / len(temp)
							off[i, j] = temp2
					elif mn == top:
						x = outx[i][j - 1]
						y = outy[i][j - 1] + 1
						if x < ref_shape[0] - w and y < ref_shape[1] - w:
							outx[i, j] = x
							outy[i, j] = y
							a = pad_image[i: i + siz, j: j + siz, :]
							b = inp2[x - w: x + w + 1, y - w: y + w + 1, :]
							temp = a - b 
							temp = temp[~np.isnan(temp)]
							temp2 = np.sum(temp ** 2) / len(temp)
							off[i, j] = temp2

					# Random Search
					alpha = 0.5
					radius = np.min(ref_shape[:2]) * (alpha**2)

					x = outx[i][j]
					y = outy[i][j]

					while radius > 1:
						x_min, x_max = max(x - radius, w), min(x + radius, ref_shape[0] - w - 1)
						y_min, y_max = max(y - radius, w), min(y + radius, ref_shape[1] - w - 1)

						random_x = np.random.randint(x_min, x_max)
						random_y = np.random.randint(y_min, y_max)

						#offset random search
						a = pad_image[i: i + siz, j: j + siz, :]
						b = inp2[random_x - w: random_x + w + 1, random_y - w: random_y + w + 1, :]
						temp = a - b 
						temp = temp[~np.isnan(temp)]
						temp2 = np.sum(temp ** 2) / len(temp)
						off_rs = temp2					
						
						# update if better patch found
						if off_rs < off[i, j]:
							off[i][j] = off_rs
							outx[i][j] = random_x
							outy[i][j] = random_y

						radius *= alpha

					# Various plots at 1/4th 3/4th iteration
					ctr += 1
					if ctr == tot and itr == 0:
						plt.subplot(332)
						plt.axis('off')
						final = do_patches([outx, outy], inp1, inp2, siz)
						final = final[:old_sz[0], :old_sz[1]]
						plt.imshow(final)
						plt.title("1 / 4 Iteration")
					elif ctr == 3 * tot and itr == 0:
						plt.subplot(333)
						plt.axis('off')
						final = do_patches([outx, outy], inp1, inp2, siz)
						final = final[:old_sz[0], :old_sz[1]]
						plt.imshow(final)
						plt.title("3 / 4 Iteration")
		else:
			# Reverse Scan Order: Right to Left, Bottom to Top
			inp_s = [np.uint64(inp_shape[0] - 1), np.uint64(inp_shape[1] - 1)]
			for i in range(inp_s[0], -1, -1):
				for j in range(inp_s[1], -1, -1):
					# Propagation:
					cur = off[i][j] # current patch
					right = off[min(i + 1, inp_s[0])][j] # right patch
					bottom = off[i][min(j + 1, inp_s[1])] # bottom patch
					mn = min(cur, right, bottom) # best of above three
					if mn == right and cur != right:
						x = outx[i + 1][j] - 1
						y = outy[i + 1][j]
						if x >= w and y >= w:
							outx[i, j] = x
							outy[i, j] = y
							a = pad_image[i: i + siz, j: j + siz, :]
							b = inp2[x - w: x + w + 1, y - w: y + w + 1, :]
							temp = a - b 
							temp = temp[~np.isnan(temp)]
							temp2 = np.sum(temp ** 2) / len(temp)
							off[i, j] = temp2
					elif mn == bottom and cur != bottom:
						x = outx[i][j - 1]
						y = outy[i][j - 1] - 1
						if x >= w and y >=  w:
							outx[i, j] = x
							outy[i, j] = y
							a = pad_image[i: i + siz, j: j + siz, :]
							b = inp2[x - w: x + w + 1, y - w: y + w + 1, :]
							temp = a - b 
							temp = temp[~np.isnan(temp)]
							temp2 = np.sum(temp ** 2) / len(temp)
							off[i, j] = temp2

					# Random Search
					alpha = 0.5
					radius = np.min(ref_shape[:2]) * (alpha**2)

					x = outx[i][j]
					y = outy[i][j]

					while radius > 1:
						x_min, x_max = max(x - radius, w), min(x + radius, ref_shape[0] - w - 1)
						y_min, y_max = max(y - radius, w), min(y + radius, ref_shape[1] - w - 1)

						random_x = np.random.randint(x_min, x_max)
						random_y = np.random.randint(y_min, y_max)

						#offset random search
						a = pad_image[i: i + siz, j: j + siz, :]
						b = inp2[random_x - w: random_x + w + 1, random_y - w: random_y + w + 1, :]
						temp = a - b 
						temp = temp[~np.isnan(temp)]
						temp2 = np.sum(temp ** 2) / len(temp)
						off_rs = temp2					
						
						# update if better patch found
						if off_rs < off[i, j]:
							off[i][j] = off_rs
							outx[i][j] = random_x
							outy[i][j] = random_y

						radius *= alpha

		plt.subplot(3, 3, itr + 4)
		plt.axis('off')
		final = do_patches([outx, outy], inp1, inp2, siz)
		final = final[:old_sz[0], :old_sz[1]]
		plt.imshow(final)
		plt.title("{} Iteration".format(itr + 1))					

	# final image made by matching patches
	final =  do_patches([outx, outy], inp1, inp2, siz)
	final = final[:old_sz[0], :old_sz[1]]
	return final

if len(sys.argv) != 5:
	print("Please provide proper command line arguments")
	exit(0)

# Input image
input_img = cv2.imread(sys.argv[1])
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img_copy = np.copy(input_img)

# Refernce image
ref_img = cv2.imread(sys.argv[2])
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

im = nearestnf(input_img, ref_img, int(sys.argv[3]), int(sys.argv[4]))

plt.subplot(3,3,7)
plt.axis('off')
plt.imshow(input_img_copy)
plt.title("Original Image")
plt.subplot(3,3,8)
plt.axis('off')
plt.imshow(ref_img)
plt.title("Reference Image")
plt.subplot(3,3,9)
plt.axis('off')
plt.imshow(im)
plt.title("Reconstructed Image")
plt.show()