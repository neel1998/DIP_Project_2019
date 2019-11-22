import numpy as np 
import cv2
import matplotlib.pyplot as plt
def RandomSearch(NNF,offsets,Radius,lenRad,ii,jj,w,inimgpad,srcimg,s):
	
	pass
def patchMatch(inImg,srcImg,psz):
	w = int((psz-1)/2)
	max_iter = 4
	s_size = inImg.shape[:2]
	i_size  = srcImg.shape[:2]
	inimgpad = np.pad(inImg,(w),'constant',constant_values=(np.nan))
	NNF = np.zeros((i_size[0],i_size[1],2))
	NNF[:,:,0] = np.random.randint(low=w,high=s_size[0]-w-1,size=(i_size[0],i_size[1]))
	NNF[:,:,1] = np.random.randint(low=w,high=s_size[1]-w-1,size=(i_size[0],i_size[1]))
	NNF = NNF.astype(np.int64)
	im1 = np.zeros(inImg.shape)
	for i in range(w,inImg.shape[0]-w,psz):
		for j in range(w,inImg.shape[1]-w,psz):
			im1[i-w:i+w+1,j-w:j+w+1,0]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,0]
			im1[i-w:i+w+1,j-w:j+w+1,1]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,1]
			im1[i-w:i+w+1,j-w:j+w+1,2]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,2]
	plt.figure()
	plt.imshow(cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_BGR2RGB))
	plt.show()

	offsets = np.full((i_size[0],i_size[1]),np.inf)
	for ii in range(i_size[0]):
		for jj in range(i_size[1]):
		    temp = inimgpad[w+ii-w:w+ii+w+1,w+jj-w:w+jj+w+1,:]- srcImg[NNF[ii,jj,0]-w:NNF[ii,jj,0]+w+1,NNF[ii,jj,1]-w:NNF[ii,jj,1]+w+1,:]
		    temp = temp[~np.isnan(temp)]
		    offsets[ii,jj] = np.sum(temp^2)/len(temp)

	for iteration in range(max_iter):
		for ii in range(i_size[0]):
			for jj in range(i_size[1]):
				ofs_prp[0] = offsets[ii,jj]
				ofs_prp[1] = offsets[np.max(0,ii-1),jj]
				ofs_prp[2] = offsets[ii,np.max(0,jj-1)]
				idx = np.argmin(ofs_prp)
				if idx == 1:
					pass
				elif idx == 2:	
					pass
im1 = cv2.imread('a.png')
im2 = cv2.imread('b.png')
patchMatch(im1,im2,1)