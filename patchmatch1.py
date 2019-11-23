import numpy as np 
import cv2
import matplotlib.pyplot as plt

def patchMatch(inImg,srcImg,psz):
	w = int((psz-1)/2)
	max_iter = 4
	inImg = inImg.astype(np.float64)
	srcImg = srcImg.astype(np.float64)
	i_size = inImg.shape[:2]
	s_size  = srcImg.shape[:2]
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
			temp = inimgpad[w+ii-w:w+ii+w+1,w+jj-w:w+jj+w+1,:] - srcImg[NNF[ii,jj,0]-w:NNF[ii,jj,0]+w+1,NNF[ii,jj,1]-w:NNF[ii,jj,1]+w+1,:]
			temp = temp[~np.isnan(temp)]
			offsets[ii,jj] = np.sum(temp**2)/len(temp)
	ofs_prp = np.zeros(3)
	for iteration in range(max_iter):
		for ii in range(i_size[0]):
			for jj in range(i_size[1]):
				ofs_prp[0] = offsets[ii,jj]
				ofs_prp[1] = offsets[np.max([0,ii-1]),jj]
				ofs_prp[2] = offsets[ii,np.max([0,jj-1])]
				idx = np.argmin(ofs_prp)
				if idx == 1:
					if NNF[ii-1,jj,0] + 1 + w < s_size[0] and NNF[ii-1,jj,1] + w < s_size[1]:
						NNF[ii,jj,:] = NNF[ii-1,jj,:]
						NNF[ii,jj,0] = NNF[ii,jj,0] + 1
						temp = inimgpad[w+ii-w:w+ii+w+1,w+jj-w:w+jj+w+1,:] - srcImg[NNF[ii,jj,0]-w:NNF[ii,jj,0]+w+1,NNF[ii,jj,1]-w:NNF[ii,jj,1]+w+1,:]
						temp = temp[~np.isnan(temp)]
						offsets[ii,jj] = np.sum(temp**2)/len(temp)
				elif idx == 2:	
					if NNF[ii,jj-1,0] < s_size[0] and NNF[ii,jj-1,1] + 1 + w< s_size[1]:
						NNF[ii,jj,:] = NNF[ii,jj-1,:]
						NNF[ii,jj,1] = NNF[ii,jj,1] + 1
						temp = inimgpad[w+ii-w:w+ii+w+1,w+jj-w:w+jj+w+1,:] - srcImg[NNF[ii,jj,0]-w:NNF[ii,jj,0]+w+1,NNF[ii,jj,1]-w:NNF[ii,jj,1]+w+1,:]
						temp = temp[~np.isnan(temp)]
						offsets[ii,jj] = np.sum(temp**2)/len(temp)

				radius = s_size[0]/4
				alpha = 0.5
				Radius = np.round(radius*alpha**np.arange(0,(-np.floor(np.log(radius)/np.log(alpha)))))
				lenRad = len(Radius)
				iis_min = np.max([w,np.max(NNF[ii,jj,0]-Radius)])
				iis_max = np.min([np.min(NNF[ii,jj,0]+Radius),s_size[0]-w-1])
				jjs_min = np.max([w,np.max(NNF[ii,jj,1]-Radius)])
				jjs_max = np.min([np.min(NNF[ii,jj,1]+Radius),s_size[1]-w-1])
				iis = np.floor(np.random.rand(lenRad)*(iis_max-iis_min)) + iis_min
				jjs = np.floor(np.random.rand(lenRad)*(jjs_max-jjs_min)) + jjs_min
				iis = iis.astype(np.uint64)
				jjs = jjs.astype(np.uint64)
				temp1=offsets[ii,jj]
				temp2=NNF[ii,jj,0]
				temp3=NNF[ii,jj,1]
				for itr_rs in range(lenRad):
					tmp1 = inimgpad[w+ii-w:w+ii+w+1,w+jj-w:w+jj+w+1,:] - srcImg[int(iis[itr_rs]-w):int(iis[itr_rs]+w+1),int(jjs[itr_rs]-w):int(jjs[itr_rs]+w+1),:]
					tmp2 = tmp1[~np.isnan(tmp1)]
					pic = np.sum(tmp2**2)/len(tmp2)
					if pic < temp1:
						temp1 = pic
						temp2 = iis[itr_rs]
						temp3 = jjs[itr_rs]
				
				offsets[ii,jj] = temp1
				NNF[ii,jj,0] = temp2
				NNF[ii,jj,1] = temp3
			if (ii == np.round(i_size[0]/4) or ii == np.round(i_size[0]*0.75)) and iteration == 1:
				im1 = np.zeros(inImg.shape)
				for i in range(w,inImg.shape[0]-w,psz):
					for j in range(w,inImg.shape[1]-w,psz):
						im1[i-w:i+w+1,j-w:j+w+1,0]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,0]
						im1[i-w:i+w+1,j-w:j+w+1,1]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,1]
						im1[i-w:i+w+1,j-w:j+w+1,2]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,2]
				plt.figure()
				plt.imshow(cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_BGR2RGB))
				plt.show()			
		im1 = np.zeros(inImg.shape)
		for i in range(w,inImg.shape[0]-w,psz):
			for j in range(w,inImg.shape[1]-w,psz):
				im1[i-w:i+w+1,j-w:j+w+1,0]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,0]
				im1[i-w:i+w+1,j-w:j+w+1,1]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,1]
				im1[i-w:i+w+1,j-w:j+w+1,2]=srcImg[NNF[i,j,0]-w:NNF[i,j,0]+w+1,NNF[i,j,1]-w:NNF[i,j,1]+w+1,2]
		plt.figure()
		plt.imshow(cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_BGR2RGB))
		plt.show()		
					
im1 = cv2.imread('a.png')
im2 = cv2.imread('b.png')
im1 = cv2.resize(im1, (int(im1.shape[0]/2),int(im1.shape[1]/2)))
im2 = cv2.resize(im2, (int(im2.shape[0]/2),int(im2.shape[1]/2)))
plt.figure()
plt.imshow(cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.show()	

patchMatch(im1,im2,1)