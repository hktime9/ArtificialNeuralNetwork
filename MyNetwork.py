import numpy as np
import sys
import time
import matplotlib.pyplot as plt

def genweights(m,n):
	weightvec= np.random.rand(m,n)
	return weightvec

def readTrainingData(filename):
	
	f= open(filename, "r")
	string=  f.read()
	string = string.split("]\n")
	strlen= len(string)
	for i in range (0,strlen):
		string[i]= string[i]+']'
	string.pop()

	return string

def readW(string):

	string = string.split("]\n")
	strlen= len(string)
	for i in range (0,strlen):
		string[i]= string[i]+']'
	string.pop()

	return string

def sig(x):
	return 1/(1+np.exp(-x))

def der_sig(x):
	return x*(1-x)

def replacespace(array=[]):

	retstr= array.replace("  ",",")
	retstr= retstr.replace(" ", ",")
	retstr= retstr.replace(",,",",")
	retstr= retstr.replace("\n","")
	newstr = retstr[:1] + retstr[1+1:]
	newstr= newstr[1:]
	newstr= newstr.rstrip("]")
	vec= newstr.split(',')
	vec= map(int, vec)
	return vec

def readTargetFile(filename):

	f= open(filename, "r")
	string= f.read()
	string= string.split('\n')
	retvec=[]
	strlen= len(string)
	for i in range(0,strlen):
		retvec.append(string[i])
	retvec.pop()
	retvec= map(int, retvec)
	return retvec

def targetVector(x):

	retvec=[]
	for i in range(0,10):
		if(i==x):
			retvec.append(1)
		else:
			retvec.append(0)
	return retvec

def updateW_ho(output=[],target=[],hidden=[]):

	sub= output-target
	return np.transpose(hidden).dot(sub)

def updateW_ih(output=[],target=[],w_ho=[],hidden=[],inputs=[]):

	sub= output-target
	sub= sub.transpose()
	err = w_ho.dot(sub)
	hidden = hidden*(1-hidden)
	err = np.multiply(err,hidden.transpose())
	final= inputs.transpose()
	final= final.dot(err.transpose())
	return final

def cross_error(output=[],target=[]):
	sum1= np.multiply(target,np.log(output))
	one_tarr= 1-target
	one_output= 1-output
	sum2= np.multiply(one_tarr,np.log(one_output))
	total= sum1+sum2

	return (-1)*np.sum(total)


def readWeights(string):

	vec=[]
	#string= string.split("***") # sting is now a list
	w2= readW(string)
	w2size= len(w2)
	for j in range(0,w2size):
		k=w2[j][1:]
		k=k[:-1]
		k=k.replace("\n","")
		vec.append(k)
	vecsize= len(vec)
	retvec=[]
	for k in range(0,vecsize):
		put= vec[k].replace("   ",'')
		put= vec[k].replace("  ",'')
		put= vec[k].split(" ")
		ele= len(put)
		putlist=[]
		for l in range(0,ele):
			if(put[l]!=''):
				putlist.append(put[l])
		retvec.append(map(float,putlist))

	return retvec
		

def train(data, labels, rate):	

	
	inputs= readTrainingData(data)
	#reading target file
	targets= readTargetFile(labels)
	#generating random weights
	w1= 2*genweights(784,30)-1
	w2= 2*genweights(30,10)-1
	
	cycles= len(inputs)
	for k in range(1,3):
		print("Doing Epoch: {}".format(k))
		errors=[]
		time_vec=[]
		start= time.time()
		for i in range(0,cycles):
			fwithcomma= replacespace(inputs[i])
			
			finmatrix= np.array(fwithcomma)
			finmatrix= finmatrix/255.0
			#input to hidden
			wx= finmatrix.dot(w1) #x*w
			a= sig(wx) #sigmoid(xw)
			#hidden to output
			w2x2= a.dot(w2)
			a2= sig(w2x2)

			tarvec= targetVector(targets[i])
			tarmat= np.array(tarvec)
			
			a2= np.array([a2]) #output layer
			a= np.array([a]) #hidden layer
			tarmat= np.array([tarmat]) #target
			finmatrix= np.array([finmatrix]) #input layer

			error_c= cross_error(a2,tarmat)
			time_p= time.time()-start
			time_vec.append(time_p)
			errors.append(error_c)

			w1= w1- rate*(updateW_ih(a2,tarmat,w2,a,finmatrix))
			w2= w2- rate*(updateW_ho(a2,tarmat,a))
		print("Epoch {} done!".format(k))

	print("done training")
	#Sampling data after every 800 element
	plot_time=[]
	plot_error=[]
	index=0
	for u in range(0,60000/800):
		plot_time.append(time_vec[index])
		plot_error.append(errors[index])
		index= index+800

	plt.plot(plot_time, plot_error)
	plt.ylabel("Accuracy")
	plt.xlabel("Time")
	
	#writing weights to files
	files= open("netWeights.txt","w")
	for item in w1:
  		files.write("%s\n" % item)
	files.close()
	files= open("netWeights.txt","a")
	files.write("***")
	for items in w2:
  		files.write("%s\n" % items)
  	files.close()
  	plt.show()#drawing graph

def test(filename, labels, weight_text):

	weight_text= open(weight_text)
	allstring= weight_text.read()
	allstring= allstring.split("***")
	weight_text.close()
	w1=readWeights(allstring[0])
	w2=readWeights(allstring[1])

	tests_in= readTrainingData(filename)
	tests_lab= readTargetFile(labels)
	cyc= len(tests_lab)
	correct= 0
	for j in range(0,cyc):
		wcomm= replacespace(tests_in[j])
		fmat= np.array(wcomm)
		fmat= fmat/255.0
		twx= fmat.dot(w1)
		ta= sig(twx)
		tw2x2= ta.dot(w2)
		ta2= sig(tw2x2)

		res= np.argmax(ta2)
	
		if(res==tests_lab[j]):
			correct= correct+1

	print("correct: {}".format(correct))
	print("total preditions: {}".format(cyc))
	print("accuracy: {}%".format((float(correct))*100/(float(cyc))))

cmd_args= sys.argv
if(cmd_args[1]=="train"):
	print("training...")
	train(cmd_args[2],cmd_args[3],float(cmd_args[4]))
elif(cmd_args[1]=="test"):
	print("testing...")
	test(cmd_args[2],cmd_args[3],cmd_args[4])
else:
	print("Write appropriate parameters")









