#-*-coding:utf8-*-
#########################################################################
#   Copyright (C) 2017 All rights reserved.
# 
#   FileName:BPThaiSyllable.py
#   Creator: yuliu1@microsoft.com
#   Time:12/07/2017
#   Description:
#
#   Updates:
#
#########################################################################
#!/usr/bin/python
# please add your code here!
#training set size
import numpy as np;
import sys;
num_examples = 15762;
#input layer dimensionality
nn_input_dim = 100;
#output layer dimensinaltiy
nn_output_dim = 4;
# gradient descent parameters
# learning rate for gradient descent
epsilon = 0.01;
#regularization strength
reg_lambda = 0.01

def FormatVector(line):
    col = line.split("\t");
    intcol = [int(float(elem)) for elem in col];
    x = intcol[:-1];
    y = intcol[-1];
    return x, y;

def LoadTrainFile(filename):
    m = 4;
    n = 100;
    XList=[];
    YList=[[]];
    linecount = 0;
    with open(filename,"r") as f:
        for line in f:
            line = line.strip();
            if line=="":
                continue;
            col = line.split("\t");
            if (len(col) == 1):
                continue;
            linesegment = "\t".join(col[:n+1]); 
            x,y = FormatVector(linesegment);
            XList.append(x);
            YList[0].append(y);
    X=np.array(XList);
    y=np.array(YList); 
    return X,y;
#Helper function to evaluate the total loss on the dataset
def calculate_loss(model,X,y):
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2'];
    #Forward propagation to calculate our perdictions
    z1 = X.dot(W1)+b1;
    a1 = np.tanh(z1);
    z2 = a1.dot(W2)+b2;
    z2max=np.max(z2,axis=1,keepdims=True);
    z2normalize = z2 - z2max;
    exp_scores = np.exp(z2normalize);
    probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True);
    temp1 = probs[range(num_examples),y];
    temp2 = temp1[temp1!=0.0];
    temp3 = temp1[temp1==0.0];
    correct_logprobs2 = np.ones(temp3.shape); 
    correct_logprobs1 = -np.log(temp2);
    maxcost = np.max(correct_logprobs1);
    maxcost*=10;
    correct_logprobs2*=maxcost;
    correct_logprobs = np.hstack((correct_logprobs1, correct_logprobs2));
    data_loss = np.sum(correct_logprobs);
    data_loss += reg_lambda/2*(np.sum(np.square(W1))+np.sum(np.square(W2)));
    return 1.0/num_examples * data_loss;

def Predict(model,x):
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2'];
    #Forward propagation
    z1 = x.dot(W1)+b1;
    a1 =np.tanh(z1);
    z2 = a1.dot(W2)+b2;
    z2max=np.max(z2,axis=1,keepdims=True);
    z2normalize = z2 - z2max;
    exp_scores = np.exp(z2normalize);
    probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True);
    labelId=np.argmax(probs,axis=1);
    return GetLabelById(labelId[0]);


 
def Run(model,infilename,outfilename):
    linecount = 0;
    m = 4;
    n = 100;
    fout = open(outfilename,"w");
    with open(infilename,"r") as f:
        for line in f:
            line = line.strip();
            col = line.split("\t");
            if (len(col) == 1):
                fout.write("\n");
                continue;
            linesegment = "\t".join(col[:n+1]); 
            x,y = FormatVector(linesegment);
            x2=[];
            x2.append(x);
            x_test = np.array(x2);
            label = Predict(model,x_test);
            newcol = col[-1].split("#");
            newline = "\t".join(newcol);
            fout.write("%s\t%s\n"%(newline,label));
            linecount+=1;
            if(linecount%5 == 0):
                sys.stderr.write("linecount=%d\n"%linecount);

#This function learns parameters for the neural network and returns the model;
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True,print the loss every 1000 iterations

def build_model(nn_hdim,X,y,num_passes=500, print_loss=True):
    #Initialize the parameters to random values. We need to learn these.
    np.random.seed(0);
    W1 = np.random.randn(nn_input_dim,nn_hdim)/np.sqrt(nn_input_dim);
    b1 = np.zeros((1,nn_hdim));
    W2 = np.random.randn(nn_hdim,nn_output_dim)/np.sqrt(nn_hdim);
    b2 = np.zeros((1,nn_output_dim));
    # This is what we return at the end
    model = {};
    #Gradient descent. For each batch..
    for i in xrange(0,num_passes):
        print "num_passes=%d\n"%i;
        #Forward propagation
        z1 = X.dot(W1)+b1;
        a1 =np.tanh(z1);
        z2 = a1.dot(W2)+b2;
        z2max=np.max(z2,axis=1,keepdims=True);
        z2normalize = z2 - z2max;
        exp_scores = np.exp(z2normalize);
        probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True);
        #Backpropagation
        delta3 = probs;
        delta3[range(num_examples),y] -=1;
        dW2 = (a1.T).dot(delta3);
        db2 = np.sum(delta3,axis=0,keepdims=True);
        delta2 = delta3.dot(W2.T)*(1-np.power(a1,2));
        dW1 = np.dot(X.T,delta2);
        db1 = np.sum(delta2,axis=0);

        #Add regularization terms (b1 and b2 don't have regularization terms);
        dW2 += reg_lambda * W2;
        dW1 += reg_lambda * W1;

        # Gradient descent parameter update
        W1 += -epsilon*dW1;
        b1 += -epsilon*db1;
        W2 += -epsilon*dW2;
        b2 += -epsilon*db2;

        #Assign new parameters to the model
        model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2};

        #Optionally print the loss.
        # This is expensive because it uses the whole dataset
        if print_loss and i %1 == 0:
            print "Loss after iteration %i:%f" %(i,calculate_loss(model,X,y));
    return model;

def GetLabelById(itsId):
    if (itsId==0):
        return "S";
    elif (itsId==1):
        return "B";
    elif (itsId==2):
        return "M";
    else:
        return "E";


if __name__ == "__main__":
    if (len(sys.argv)!=4):
        sys.stderr.write("no enough params\n");
        sys.exit(1);
    X,y=LoadTrainFile(sys.argv[1]);
    print X.shape,y.shape;
    '''
    for i in range(0,len(X)):
        xlist = X[i].tolist();
        t1=[];
        t1.append(xlist);
        t=np.array(t1);
        model = build_model(100,t,y);
        sys.stderr.write("i=%d"%i);
    '''
    model = build_model(300,X,y);
    Run(model,sys.argv[2],sys.argv[3]); 
