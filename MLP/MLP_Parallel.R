library(parallel)
library(RSNNS)
library(dplyr)

df <- read.csv('../data/EncodedDataLrg.csv',stringsAsFactors=F)
df.save <- df
m = dim(df)[1]
n = dim(df)[2]

pred.key <- data.frame(key=c('p0','p1','p2','p3','p4'),
                       val=c('isFather','isMother','isSpouse','isSibling','isChild'),
                       name=c('P22','P25','P26','P3373','P40'))

get_acc <- function(preds,test,gd){
  temp <- as.data.frame(t(sapply(seq(5),function(i)sum(preds[,i]==sapply(test[gd,n-5+i],function(x)max(c(0,x))))/nrow(test))))
  names(temp) <- pred.key$val
  return(temp)
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

#no_cores <- detectCores() - 1
#c1 <- makeCluster(no_cores,type='FORK')
bin.sz <- 5
idx <- sample(m,m)
bins <- lapply(seq(bin.sz),function(i)idx[((i-1)*length(idx)/bin.sz+1):min(length(idx),i*length(idx)/bin.sz)])
results <- list(cms=list(),all_preds=list(),targets=list(),nets=list(),times=NULL)
alphas <- c(1e-4)
betas <- c(0.001)
layers <- list(c(256))
for (bin.num in seq(length(bins))){
  for (beta in betas){
    for (alpha in alphas){
      
      trn.idx <- unlist(bins[-c(bin.num)])
      train <- df[trn.idx,]
      tst.idx <- setdiff(seq(m),trn.idx)
      test <- df[tst.idx,]
      test[,(n-4):n] <- apply(test[,(n-4):n],2,function(x)sapply(x,function(y)max(c(y,0))))
      
      maxiter <- 120
      name <- paste0('StdNormMomentum a=',alpha,' b=',beta, ' bin=',bin.num)
      print(paste0('Running............................... ',name))
      results<-Train.Plot(train=train,test=test,name=name,layers=layers,normalize=T, maxit=maxiter,outdir='plots/cv/',
                          results=results,learnFunc = "BackpropMomentum",learnFuncParams = c(alpha,beta))
    }
  }
}

get_vals <- function(cm){
  pos = cm[which(row.names(cm)=='1'),]
  sens = pos[which(names(pos)=='1')]/sum(pos,na.rm=T)
  
  neg = cm[which(row.names(cm)=='0'),]
  spec = neg[which(names(neg)=='0')]/sum(neg,na.rm=T)
  
  pos = t(cm)[which(row.names(t(cm))=='1'),]
  prec = pos[which(names(pos)=='1')]/sum(pos,na.rm=T)
  
  neg = t(cm)[which(row.names(t(cm))=='0'),]
  recall = neg[which(names(neg)=='0')]/sum(neg,na.rm=T)
  
  out = data.frame(fallout=1-spec,sensitivity=sens,precision=prec,recall=sens,stringsAsFactors=F)
  return(out)
}

library(plyr)
all_targets <- NULL
all_preds <- NULL
for (i in seq(length(results$targets))){
  targs <- results$targets[[i]]
  targs <- targs[match(seq(nrow(df)),row.names(targs)),]
  targs[is.na(targs)] <- -1
  
  preds <- results$all_preds[[i]]
  preds <- preds[match(seq(nrow(df)),row.names(preds)),]
  preds[is.na(preds)] <- -1
  
  if (is.null(all_targets)){
    all_targets <- targs
  }else{
    all_targets <- cbind(all_targets,targs)
  }
   
  if (is.null(all_preds)){
    all_preds <- preds
  }else{
    all_preds <- cbind(all_preds,preds)
  } 
  
  #plotROC(T=all_preds,D=all_targets)
    
}

#stopCluster(c1)
save.image('plots/cv/MLP_cv.rda')
save.image('plots/grid/MLP_grid.rda')
save.image('plots/random/MLP_random.rda')


