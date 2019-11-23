import numpy as np
import matplotlib.pyplot as plt

a = np.arange(100, 100 + 81 * 3).reshape(9, 9, 3)
b = np.arange(100, 100 + 81 * 3).reshape(9, 9, 3)

def do_patches(nnf, inp1, inp2, siz):
	sh1 = inp1.shape
	w = int((siz - 1) / 2)
	out = np.zeros(sh1)
	for i in range(w, sh1[0] - w, siz):
		for j in range(w, sh1[1] - w, siz):
			x = nnf[0][i][j]
			y = nnf[1][i][j]
			print(x, y, i, j, inp2.shape)
			out[i - w: i + w + 1, j - w: j + w + 1] = inp2[x - w: x + w + 1, y - w: y + w + 1]
	return out
	
def lnorm(i, j, pad_image, inp2, siz, nnf):
	w = int((siz - 1) / 2)
	outx, outy = nnf
	temp = pad_image[i: i + siz, j: j + siz, :] - inp2[outx[i, j] - w: outx[i, j] + w + 1, outy[i, j] - w: outy[i, j] + w + 1, :]
	temp = temp[~np.isnan(temp)]
	return np.sum(temp ** 2) / len(temp)

def nearestnf(inp1, inp2, siz, iterations):
	w = int((siz - 1) / 2)
	sh1 = np.shape(inp1)
	sh2 = np.shape(inp2)
	outx = np.random.randint(w, sh2[0] - w, (sh1[0], sh1[1]))
	outy = np.random.randint(w, sh2[1] - w, (sh1[0], sh1[1]))
	# a = do_patches([outx, outy], inp1, inp2, siz)
	pad_image = np.pad(inp1, ((w,w),(w,w),(0,0)), 'constant', constant_values=(np.nan, np.nan))

	off = np.full((sh1[0], sh1[1]), np.inf)
	for i in range(sh1[0]):
		for j in range(sh1[1]):
			off[i, j] = lnorm(i, j, pad_image, inp2, siz, [outx, outy])

	for _ in range(iterations):

		for i in range(sh1[0]):
			for j in range(sh1[0]):

				# Propagation:
				cur = off[i][j]
				left = off[max(i - 1, 0)][j]
				top = off[i][max(j - 1, 0)]
				mn = min(cur, left, top)
				if mn == left:
					x = outx[i][j]
					y = outy[i][j]
					if x + 1 < siz - w and y < siz - w:
						outx[i, j] = outx[i - 1, j] + 1
						outy[i, j] = outy[i - 1, j]
						off[i, j] = lnorm(i, j, pad_image, inp2, siz, [outx, outy])
				elif mn == top:
					x = outx[i][j]
					y = outy[i][j]
					if x < siz - w and y + 1 < siz - w:
						outx[i, j] = outx[i, j - 1]
						outy[i, j] = outy[i, j - 1] + 1
						off[i, j] = lnorm(i, j, pad_image, inp2, siz, [outx, outy])

				# Random Search
				

	# for i in range(a.shape[0]):
	# 	for j in range(a.shape[1]):
	# 		print(a[i][j], end=' ')
	# 	print()

for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		print(a[i][j], end=' ')
	print()
print("------")
for i in range(b.shape[0]):
	for j in range(b.shape[1]):
		print(b[i][j], end=' ')
	print()
print("------")

nearestnf(a, b, 3, 5)