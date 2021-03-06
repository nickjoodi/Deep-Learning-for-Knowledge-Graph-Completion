library(RSNNS)
library(dplyr)
library(parallel)

# Manual Here
# http://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node143.html
# Information on parameters available at
# http://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node18.html 

##setwd('Documents/School/ECS289N/Project/Deep-Learning-for-Knowledge-Graph-Completion/MLP')
df <- read.csv('../data/EncodedData.csv',stringsAsFactors=F)
#load('MLP.rda')
df.save <- df

pred.key <- data.frame(key=c('p0','p1','p2','p3','p4'),
                       val=c('isFather','isMother','isSpouse','isSibling','isChild'))

## Split the data up into training and testing sets
df <- df.save
m = dim(df)[1]
n = dim(df)[2]

if (F){ # Change to true to pre-normalize the data
  good.idx <- apply(df,1,function(x)sum(is.na(x))==0)
  df[good.idx,-c(1:2,(n-4):n)] <- normalizeData(df[good.idx,-c(1:2,(n-4):n)])
}

trn.idx <- sample(m,floor(0.9*m))
train <- df[trn.idx,]

tst.idx <- setdiff(seq(m),trn.idx)
test <- df[tst.idx,]
test[,(n-4):n] <- apply(test[,(n-4):n],2,function(x)sapply(x,function(y)max(c(y,0))))


get_acc <- function(preds,test,gd){
  temp <- as.data.frame(t(sapply(seq(5),function(i)sum(preds[,i]==test[gd,n-5+i])/nrow(test))))
  names(temp) <- pred.key$val
  return(temp)
}
## Run a neural net on each predicate (could also do multiclass)
layers <- list(c(16),c(64),c(128),c(128,16),c(128,64),c(128,128),
               c(128,128,16),c(128,128,32),c(256),c(256,16),c(256,128),
               c(256,256),c(512),c(1024),c(32,32,32,32),c(600,256),c(256,64,16))

# Change to true if running from the beginning
if (T){
  cms <- list()
  all_preds <- list()
  targets <- list()
  times <- NULL
  nets <- list()
  results <- list(cms=cms,all_preds=all_preds,targets=targets,nets=nets,times=times)
}

Train.Plot <- function(train,test,name,layers,results,maxit=250,normalize=F,
                       learnFunc="Std_Backpropagation",learnFuncParams=c(0.2,0),
                       initFunc="Randomize_Weights",initFuncParams=c(-0.3,0.3),
                       updateFunc="Toplogical_Order",updateFuncParams=c(0),
                       pruneFunc=NULL,pruneFuncparams=NULL, outdir='plots/',
                       hiddenActFunc="Act_Logistic",shufflePatterns=T,linOut=F,
                       outputActFunc=if(linOut)"Act_Identity"else"Act_Logistic"){
  
  cms <- results$cms
  all_preds <- results$all_preds
  times <- results$times
  targets <- results$targets
  nets <- results$nets
  
  gd     <- which(apply(test,1,function(x)sum(is.na(x)))==0)
  gd.trn <- which(apply(train,1,function(x)sum(is.na(x)))==0)
  
  if (normalize){
    train[gd.trn,-c(1,2,(n-4):n)] <- normalizeData(train[gd.trn,-c(1,2,(n-4):n)])
    test[gd,-c(1,2,(n-4):n)]  <- normalizeData(test[gd,-c(1,2,(n-4):n)])
  }
  
  for (lay in seq(length(layers))){
    ptm <- proc.time()
    nn <- mlp(x=train[gd.trn,-c(1,2,(n-4):n)],y=train[gd.trn,(n-4):n],size = layers[[lay]],
              inputsTest = test[gd,-c(1,2,(n-4):n)],targetsTest = test[gd,(n-4):n],maxit=maxit, 
              learnFunc=learnFunc,learnFuncParams=learnFuncParams,
              initFunc=initFunc, initFuncParams=initFuncParams,
              hiddenActFunc=hiddenActFunc,outputActFunc=outputActFunc,
              shufflePatterns=shufflePatterns,linOut=linOut)
    
    print(paste0('Time: ',paste0(round((proc.time()-ptm)[3],2),collapse=' '),' s' ))
    times <- as.data.frame(cbind(times,cbind(proc.time()-ptm)))
    names(times)[ncol(times)] <- paste0(name,paste0(layers[[lay]],collapse='/'))
    
    # Plot the MSE per epoch
    png(filename = paste0(outdir,'mse/',paste0(name,'_',paste0(layers[[lay]],collapse='-')),'.png'))
    plotIterativeError(nn,main=paste0('MSE per Epoch of ',paste0(name,paste0(layers[[lay]],collapse='/'))))
    preds <- round(predict(nn,test[gd,-c(1,2,(n-4):n)]),0)
    print(paste0('Accuracy for ',paste0(name,paste0(layers[[lay]],collapse='/')),': ',round(mean(t(get_acc(preds,test,gd)),na.rm=T),3)*100,' %' ))
    graphics.off()
    
    # Store the confusion matrix
    for (i in 1:5){
      each.col <- paste0('p',0:4)[i]
      cm <- confusionMatrix(predictions=preds[,i],targets=test[gd,each.col])
      cms <- append(cms,list(cm))
      names(cms)[length(cms)] <- paste0(name,' size: ',paste0(layers[[lay]],collapse='/'),' col: ',pred.key$val[which(pred.key$key==each.col)])
    }
    cms <- append(cms,list(confusionMatrix(predictions=preds,apply(test[gd,(n-4):n],2,as.numeric))))
    names(cms)[length(cms)] <- paste0(name,' size: ',paste0(layers[[lay]],collapse='/'),' col: All')
    cms <- append(cms,list(get_acc(preds,test,gd)))
    names(cms)[length(cms)] <- paste0(name,' size: ',paste0(layers[[lay]],collapse='/'),' col: Accuracy')
    
    # Store the predictions
    all_preds <- append(all_preds,list(preds))
    names(all_preds)[length(all_preds)] <- paste0(name,'_',paste0(layers[[lay]],collapse='/'))
    targets <- append(targets,list(test[gd,(n-4):n]))
    names(targets)[length(targets)] <- paste0(name,'_',paste0(layers[[lay]],collapse='/'))
    nets <- append(nets,list(nn))
    names(nets)[length(nets)] <- paste0(name,'_',paste0(layers[[lay]],collapse='/'))
    
    # Plot ROC Curves
    png(filename = paste0(outdir,'roc/',paste0(name,'_',paste0(layers[[lay]],collapse='-')),'.png'))
    plotROC(T=preds,D=apply(test[gd,(n-4):n],2,as.numeric),main=paste0('ROC Curve for Model ',paste0(paste0(layers[[lay]],collapse='/'))),sub=name)
    graphics.off()
  }
  return(list(cms=cms,all_preds=all_preds,targets=targets,nets=nets,times=times))
}


layers <- list(c(64))  # Best set of layers

# CesarsTestRun
results<-Train.Plot(train=train,test=test,name='Testing',layers=layers,updateFuncParams=c(0.0025,0.001),results=results)

# Standard
results<-Train.Plot(train=train,test=test,name='Std',layers=layers,updateFuncParams=c(0.0025,0.001),results=results)

# Standard w/ Normalize & Momentum
results<-Train.Plot(train=train,test=test,name='StdNormMomentum',layers=layers,normalize=T, maxit=70,
           results=results,learnFunc = "BackpropMomentum",updateFuncParams=c(0.0025,0.001),outdir='plots/random/',
           learnFuncParams = c(0.005,0.01))

# Standard w/ Normalize & Momentum & RBF
results<-Train.Plot(train=train,test=test,name='StdNormMomentumRBF',layers=layers,normalize=T, maxit=100,
                    results=results,learnFunc = "BackpropMomentum",updateFuncParams=c(0.0025,0.001),
                    learnFuncParams = c(0.005,0.01), hiddenActFunc = 'Act_RBF_Gaussian')

# TDNN
results<-Train.Plot(train=train,test=test,name='TDNN',layers=layers,updateFuncParams=c(0.0025,0.001),
                    results=results,hiddenActFunc='Act_TD_Logistic',updateFunc='TimeDelay_Order',learnFunc='TimeDelayBackprop')

# Momentum, Synch, RBF
results<-Train.Plot(train=train,test=test,name='BackPropSyncRBF',layers=layers,updateFuncParams=c(0.0025,0.001),
                    results=results,hiddenActFunc='Act_Logistic',updateFunc='Synchronous_Order',learnFunc='BackpropMomentum',
                    initFunc='RBF_Weights',initFuncParams=c(0.5))

save.image('MLP.rda')

# Test many params
