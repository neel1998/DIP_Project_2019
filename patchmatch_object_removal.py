import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import time

a = np.arange(10000, 10000 + 40000 * 3).reshape(200, 200, 3)
b = np.arange(10000, 10000 + 40000 * 3).reshape(200, 200, 3)

ref_point = []
crop = False

def shape_selection(event, x, y, flags, param): 
	global ref_point, crop 

	if event == cv2.EVENT_LBUTTONDOWN: 
		ref_point = [(x, y)] 

	elif event == cv2.EVENT_LBUTTONUP: 
		ref_point.append((x, y)) 
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

def do_patches(nnf, inp1, inp2, siz):
	inp_shape = inp1.shape
	w = int((siz - 1) / 2)
	out = np.zeros(inp_shape, np.float)
	print(nnf[0].max(), nnf[1].min())
	for i in range(w, inp_shape[0] - w, siz):
		for j in range(w, inp_shape[1] - w, siz):
			x = nnf[0][i][j]
			y = nnf[1][i][j]
			# print(x, y, i, j, inp2.shape)
			temp = inp2[x - w: x + w + 1, y - w: y + w + 1]
			out[i - w: i + w + 1, j - w: j + w + 1] = temp
	out = np.uint8(out)
	return out
	

def nearestnf(inp1, inp2, siz, iterations):
	w = int((siz - 1) / 2)
	inp_shape = np.shape(inp1)
	old_sz = inp_shape
	new_inp_shape = np.zeros(len(inp_shape))
	new_inp_shape[0] = inp_shape[0] + siz - inp_shape[0] % siz
	new_inp_shape[1] = inp_shape[1] + siz - inp_shape[1] % siz
	for i in range(2, len(inp_shape)):
		new_inp_shape[i] = inp_shape[i]
	new_inp_shape = np.uint(new_inp_shape)
	new_inp = np.zeros(new_inp_shape)
	new_inp[:inp_shape[0], :inp_shape[1]] = inp1[:inp_shape[0], :inp_shape[1]]
	new_inp = np.uint8(new_inp)
	inp1 = np.copy(new_inp)
	inp_shape = new_inp_shape
	ref_shape = np.shape(inp2)
	outx = np.random.randint(w, ref_shape[0] - w, (inp_shape[0], inp_shape[1]))
	outy = np.random.randint(w, ref_shape[1] - w, (inp_shape[0], inp_shape[1]))
	
	pad_image = np.pad(inp1, ((w,w),(w,w),(0,0)), 'constant', constant_values=(np.nan, np.nan))

	off = np.full((inp_shape[0], inp_shape[1]), np.inf)
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

	print(outx, outy)
	print(np.linalg.norm(off))
	iter1 = 1
	iter2 = 1
	for itr in range(iterations):
		if itr % 2 == 0:
			print(inp_shape)
			for i in range(inp_shape[0]):
				for j in range(inp_shape[1]):
					# print("WOHOOOO")
					# Propagation:
					cur = off[i][j]
					left = off[max(i - 1, 0)][j]
					top = off[i][max(j - 1, 0)]
					mn = min(cur, left, top)
					if mn == left:
						x = outx[i - 1][j] + 1
						y = outy[i - 1][j]
						if x < ref_shape[0] - w and y < ref_shape[1] - w:
							iter1 += 1
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
							iter1 += 1
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
					# print(radius)
					x = outx[i][j]
					y = outy[i][j]
					while radius > 1:
						x_min, x_max = max(x - radius, w), min(x + radius, ref_shape[0] - w - 1)
						y_min, y_max = max(y - radius, w), min(y + radius, ref_shape[1] - w - 1)
						# print(x_min, x_max)
						# print(y_min, y_max)

						random_x = np.random.randint(x_min, x_max)
						random_y = np.random.randint(y_min, y_max)

						#offset random search
						a = pad_image[i: i + siz, j: j + siz, :]
						b = inp2[random_x - w: random_x + w + 1, random_y - w: random_y + w + 1, :]
						temp = a - b 
						temp = temp[~np.isnan(temp)]
						temp2 = np.sum(temp ** 2) / len(temp)
						off_rs = temp2					
						# print(off_rs)
						if off_rs < off[i, j]:
							# print("RS")
							iter2 += 1
							off[i][j] = off_rs
							outx[i][j] = random_x
							outy[i][j] = random_y

						radius *= alpha
		else:
			inp_s = [np.uint64(inp_shape[0] - 1), np.uint64(inp_shape[1] - 1)]
			for i in range(inp_s[0], -1, -1):
				for j in range(inp_s[1], -1, -1):
					# Propagation:
					cur = off[i][j]
					right = off[min(i + 1, inp_s[0])][j]
					bottom = off[i][min(j + 1, inp_s[1])]
					mn = min(cur, right, bottom)
					if mn == right and cur != right:
						x = outx[i + 1][j] - 1
						y = outy[i + 1][j]
						if x >= w and y >= w:
							iter1 += 1
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
							iter1 += 1
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
					# print(radius)
					x = outx[i][j]
					y = outy[i][j]
					while radius > 1:
						x_min, x_max = max(x - radius, w), min(x + radius, ref_shape[0] - w - 1)
						y_min, y_max = max(y - radius, w), min(y + radius, ref_shape[1] - w - 1)
						# print(x_min, x_max)
						# print(y_min, y_max)

						random_x = np.random.randint(x_min, x_max)
						random_y = np.random.randint(y_min, y_max)

						#offset random search
						a = pad_image[i: i + siz, j: j + siz, :]
						b = inp2[random_x - w: random_x + w + 1, random_y - w: random_y + w + 1, :]
						temp = a - b 
						temp = temp[~np.isnan(temp)]
						temp2 = np.sum(temp ** 2) / len(temp)
						off_rs = temp2					
						# print(off_rs)
						if off_rs < off[i, j]:
							# print("RS")
							iter2 += 1
							off[i][j] = off_rs
							outx[i][j] = random_x
							outy[i][j] = random_y

						radius *= alpha


	print(np.linalg.norm(off))
	final =  do_patches([outx, outy], inp1, inp2, siz)
	print(old_sz)
	final = final[:old_sz[0], :old_sz[1]]
	return final
	# for i in range(a.shape[0]):
	# 	for j in range(a.shape[1]):
	# 		print(a[i][j], end=' ')
	# 	print()

# for i in range(a.shape[0]):
# 	for j in range(a.shape[1]):
# 		print(a[i][j], end=' ')
# 	print()
# print("------")
# for i in range(b.shape[0]):
# 	for j in range(b.shape[1]):
# 		print(b[i][j], end=' ')
# 	print()
# print("------")
if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("Please provide proper command line arguments")
		exit(0)

	image = cv2.imread(sys.argv[1])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	immm = np.copy(image)

	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", shape_selection)

	while True: 
		cv2.imshow("image", image) 
		key = cv2.waitKey(1) & 0xFF

		if key == ord("r"): 
			image = clone.copy() 

		elif key == ord("c"): 
			break

	if len(ref_point) == 2: 
		crop_img1 = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]: 
															ref_point[1][0]] 
		cv2.imshow("crop_img", crop_img1)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

	rf = np.copy(np.array(ref_point))
	image = cv2.imread(sys.argv[2])
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	immm1 = np.copy(image)
	# clone = image.copy()
	# cv2.namedWindow("image")
	# cv2.setMouseCallback("image", shape_selection)

	# while True: 
	# 	cv2.imshow("image", image) 
	# 	key = cv2.waitKey(1) & 0xFF

	# 	if key == ord("r"): 
	# 		image = clone.copy() 

	# 	elif key == ord("c"): 
	# 		break

	# if len(ref_point) == 2: 
	# 	crop_img2 = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]: 
	# 														ref_point[1][0]] 
	# 	cv2.imshow("crop_img", crop_img2)
	# 	cv2.waitKey(0)

	# cv2.destroyAllWindows()

	# crop_img1 = np.double(crop_img1)
	# crop_img2 = np.double(crop_img2)
	im = nearestnf(crop_img1, immm1, int(sys.argv[3]), int(sys.argv[4]))

	immm[rf[0][1]:rf[1][1], rf[0][0]:rf[1][0]] = im
	name = './static/results/' + sys.argv[1].split("/")[-1].split(".")[0] + "_" + sys.argv[2].split("/")[-1].split(".")[0]	 + "_res.jpg"
	print(name)
	cv2.imwrite(name, immm)
	plt.subplot(1,2,1)
	plt.imshow(immm1)
	plt.subplot(1,2,2)
	plt.imshow(immm)
	plt.show()