package parser;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import parser.Options.LearningMode;
import parser.decoding.DependencyDecoder;
import parser.io.DependencyReader;
import parser.pruning.BasicArcPruner;
import parser.sampling.RandomWalkSampler;
import utils.FeatureVector;

public class DependencyParser implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	protected Options options;
	protected DependencyPipe pipe;
	protected Parameters parameters;
	
	DependencyParser pruner;
	
	double pruningGoldHits = 0;
	double pruningTotGold = 1e-30;
	double pruningTotUparcs = 0;
	double pruningTotArcs = 1e-30;
	
	public static void main(String[] args) 
		throws IOException, ClassNotFoundException, CloneNotSupportedException
	{
		
		Options options = new Options();
		options.processArguments(args);
		options.printOptions();
		
		DependencyParser pruner = null;
		if (options.train && options.pruning && options.learningMode != LearningMode.Basic) {
			Options prunerOptions = new Options();
			prunerOptions.processArguments(args);
			prunerOptions.maxNumIters = 10;
			
			prunerOptions.learningMode = LearningMode.Basic;
			prunerOptions.pruning = false;
			prunerOptions.test = false;
			prunerOptions.learnLabel = false;
			prunerOptions.gamma = 1.0;
			prunerOptions.gammaLabel = 1.0;
			
			//pruner = new DependencyParser();
			pruner = new BasicArcPruner();
			pruner.options = prunerOptions;
			
			DependencyPipe pipe = new DependencyPipe(prunerOptions);
			pruner.pipe = pipe;
			
			pipe.createAlphabets(prunerOptions.trainFile);
			DependencyInstance[] lstTrain = pipe.createInstances(prunerOptions.trainFile);
			
			Parameters parameters = new Parameters(pipe, prunerOptions);
			pruner.parameters = parameters;
			
			pruner.train(lstTrain);
		}
		
		if (options.train) {
			DependencyParser parser = new DependencyParser();
			//DependencyParser parser = new BasicArcPruner();
			parser.options = options;
			
			DependencyPipe pipe = new DependencyPipe(options);
			parser.pipe = pipe;
			
			if (options.pruning) parser.pruner = pruner;
			
			pipe.createAlphabets(options.trainFile);
			DependencyInstance[] lstTrain = pipe.createInstances(options.trainFile);
			
			Parameters parameters = new Parameters(pipe, options);
			parser.parameters = parameters;
			
			parser.train(lstTrain);
			parser.saveModel();
		}
		
		if (options.test) {
			DependencyParser parser = new DependencyParser();
			parser.options = options;
			
			parser.loadModel();
			
			if (!options.train && options.wordVectorFile != null)
            	parser.pipe.loadWordVectors(options.wordVectorFile);
			
			System.out.printf(" Evaluating: %s%n", options.testFile);
			parser.evaluateSet(true, false, null);
		}
		
	}
	
    public void saveModel() throws IOException 
    {
    	ObjectOutputStream out = new ObjectOutputStream(
    			new GZIPOutputStream(new FileOutputStream(options.modelFile)));
    	out.writeObject(pipe);
    	out.writeObject(parameters);
    	if (options.pruning && options.learningMode != LearningMode.Basic) 
    		out.writeObject(pruner);
    	out.close();
    }
	
    public void loadModel() throws IOException, ClassNotFoundException 
    {
        ObjectInputStream in = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(options.modelFile)));    
        pipe = (DependencyPipe) in.readObject();
        parameters = (Parameters) in.readObject();
        if (options.pruning/*&& options.learningMode != LearningMode.Basic*/)
        	//pruner = (DependencyParser) in.readObject();
        	pruner = (BasicArcPruner) in.readObject();
        pipe.options = options;
        parameters.options = options;        
        in.close();
        pipe.closeAlphabets();
    }
    
	public void printPruningStats()
	{
		System.out.printf("  Pruning Recall: %.4f\tEffcy: %.4f%n",
				pruningGoldHits / pruningTotGold,
				pruningTotUparcs / pruningTotArcs);
	}
	
	public void resetPruningStats()
	{
		pruningGoldHits = 0;
		pruningTotGold = 1e-30;
		pruningTotUparcs = 0;
		pruningTotArcs = 1e-30;
	}
	
    public void train(DependencyInstance[] lstTrain) 
    	throws IOException, CloneNotSupportedException 
    {
    	long start = 0, end = 0;
    	
        if (options.R > 0 && options.gamma < 1 && options.initTensorWithPretrain) {

        	Options optionsBak = (Options) options.clone();
        	options.learningMode = LearningMode.Basic;
        	options.R = 0;
        	options.gamma = 1.0;
        	options.gammaLabel = 1.0;
        	options.maxNumIters = options.numPretrainIters;
            options.useHO = false;
        	parameters.gamma = 1.0;
        	parameters.gammaLabel = 1.0;
        	parameters.rank = 0;
    		System.out.println("=============================================");
    		System.out.printf(" Pre-training:%n");
    		System.out.println("=============================================");
    		
    		start = System.currentTimeMillis();

    		System.out.println("Running MIRA ... ");
    		trainIter(lstTrain, false);
    		System.out.println();
    		
    		System.out.println("Init tensor ... ");
    		LowRankParam tensor = new LowRankParam(parameters);
    		pipe.fillParameters(tensor, parameters);
    		tensor.decompose(1, parameters);
            System.out.println();
    		end = System.currentTimeMillis();
    		
    		options.learningMode = optionsBak.learningMode;
    		options.R = optionsBak.R;
    		options.gamma = optionsBak.gamma;
    		options.gammaLabel = optionsBak.gammaLabel;
    		options.maxNumIters = optionsBak.maxNumIters;
            options.useHO = optionsBak.useHO;
    		parameters.rank = optionsBak.R;
    		parameters.gamma = optionsBak.gamma;
    		parameters.gammaLabel = optionsBak.gammaLabel;
    		parameters.clearTheta();
            parameters.printUStat();
            parameters.printVStat();
            parameters.printWStat();
            System.out.println();
            System.out.printf("Pre-training took %d ms.%n", end-start);    		
    		System.out.println("=============================================");
    		System.out.println();	    

        } else {
        	parameters.randomlyInitUVW();
        }
        
		System.out.println("=============================================");
		System.out.printf(" Training:%n");
		System.out.println("=============================================");
		
		start = System.currentTimeMillis();

		System.out.println("Running MIRA ... ");
		trainIter(lstTrain, true);
		System.out.println();
		
		end = System.currentTimeMillis();
		
		System.out.printf("Training took %d ms.%n", end-start);    		
		System.out.println("=============================================");
		System.out.println();		    	
    }
    
    public void trainIter(DependencyInstance[] lstTrain, boolean evalAndSave) throws IOException
    {
    	
    	DependencyDecoder decoder = DependencyDecoder
    			.createDependencyDecoder(options);
    	BufferedWriter bw = new BufferedWriter(new FileWriter("debug." + Options.langString[options.lang.ordinal()]));
    	
    	int N = lstTrain.length;
    	int printPeriod = 10000 < N ? N/10 : 1000;
    	
    	int updCnt = 0;
    	
    	BufferedWriter bw_f = new BufferedWriter(new FileWriter("feat." + Options.langString[options.lang.ordinal()]));
    	BufferedWriter bw_m = new BufferedWriter(new FileWriter("model." + Options.langString[options.lang.ordinal()]));

    	for (int iIter = 0; iIter < options.maxNumIters; ++iIter) {
    	    
    		decoder.resetCount();
    		//parameters.randomParams();
    		
    		if (pruner != null) pruner.resetPruningStats();
    		
            // use this offset to change the udpate ordering of U, V and W
            // when N is a multiple of 3, such that U, V and W get updated
            // on each sentence.
            int offset = (N % 3 == 0) ? iIter : 0;

    		long start = 0;
    		double loss = 0;
    		int uas = 0, tot = 0;
    		start = System.currentTimeMillis();
                		    		
    		for (int i = 0; i < N; ++i) {
    			
    			if ((i + 1) % printPeriod == 0) {
				System.out.printf("  %d (time=%ds)", (i+1),
					(System.currentTimeMillis()-start)/1000);
    			}

    			//DependencyInstance inst = new DependencyInstance(lstTrain[i]);
    			DependencyInstance inst = lstTrain[i];
    		    int n = inst.length;
    		    
    			{
	    			LocalFeatureData lfd = new LocalFeatureData(inst, this, true);
	    		    GlobalFeatureData gfd = new GlobalFeatureData(lfd);
	    		    
	    		    DependencyInstance predInst = decoder.decode(inst, lfd, gfd, true);
	    		    
	        		int ua = evaluateUnlabelCorrect(inst, predInst), la = 0;
	        		if (options.learnLabel)
	        			la = evaluateLabelCorrect(inst, predInst);        		
	        		uas += ua;
	        		tot += n-1;
	        		
	        		if ((options.learnLabel && la != n-1) ||
	        				(!options.learnLabel && ua != n-1)) {
	        			
	        			if (options.updateLocal) {

		        			for (int m = 1; m < predInst.length; ++m) {
	        					if (predInst.heads[m] != inst.heads[m]) {
	    		        			updCnt++;

	        						int oldHead = predInst.heads[m];
	        						FeatureVector predFV = lfd.getPartialFeatureVector(predInst.heads, m);
	        	    		    	//FeatureVector predFV = lfd.getArcFeatureVector(predInst.heads[m], m);
	        						double predScore = parameters.dotProduct(predFV);
	        						
	        						predInst.heads[m] = inst.heads[m];
	        						FeatureVector goldFV = lfd.getPartialFeatureVector(predInst.heads, m);
	        						//FeatureVector goldFV = lfd.getArcFeatureVector(inst.heads[m], m);
	        						double goldScore = parameters.dotProduct(goldFV);
	        						
	        						predInst.heads[m] = oldHead;
	        						
	        						loss += predScore + 1.0 - goldScore;
	        						parameters.updateTheta(goldFV, predFV, predScore + 1.0 - goldScore, updCnt);
	        					}
	        				}
	        			}
	        			
	        			updCnt++;
	        			
	        			loss += parameters.update(inst, predInst, lfd, gfd, updCnt, offset);
	                }

	        		// output feat
	        		if (iIter == 10000) {
	        			//System.out.println("aaa");
	        			for (int z = 1; z < inst.length; ++z) {
	        				bw_f.write("" + z + "\t" + inst.forms[z] + "\t" + inst.postags[z] + "\t" + inst.heads[z] + "\n");
	        			}
	        			FeatureVector goldFV = lfd.getFeatureVector(inst);
	        			goldFV.addEntries(gfd.getFeatureVector(inst));
	        			
	        			int[] feat = goldFV.x();
	        			String str = "";
	        			for (int z = 0; z < feat.length; ++z) {
	        				str += " " + feat[z];
	        			}
	        			bw_f.write(str.trim() + "\n\n");
	        			bw_f.flush();
	        			//System.out.println("bbb");
	        		}
    			}

    		    // update for gold optimum?
        		if (options.updateGold)
        		{
        			LocalFeatureData goldLfd = new LocalFeatureData(inst, this, true);
        		    GlobalFeatureData goldGfd = new GlobalFeatureData(goldLfd);

        		    DependencyInstance predGoldInst = decoder.decodeGold(inst, goldLfd, goldGfd, true);
        		    
            		int ua = evaluateUnlabelCorrect(inst, predGoldInst);
            		int la = 0;
            		if (options.learnLabel)
            			la = evaluateLabelCorrect(inst, predGoldInst);        		
            		uas += ua;
            		tot += n-1;
            		
            		if ((options.learnLabel && la != n-1) ||
            				(!options.learnLabel && ua != n-1)) {
            			updCnt++;
            			//loss += parameters.update(inst, predGoldInst, goldLfd, goldGfd,
            			//		iIter * N + i + 1, offset);
            			//loss += parameters.update(inst, predGoldInst, goldLfd, goldGfd,
            			//		2 * (iIter * N + i + 1), offset);
                    }
       			
        		}
    		}
    		System.out.printf("%n  Iter %d\tloss=%.4f\tuas=%.4f\t[%ds]%n", iIter+1,
    				loss, uas/(tot+0.0),
    				(System.currentTimeMillis() - start)/1000);
    		
    		if (options.learningMode != LearningMode.Basic && options.pruning && pruner != null)
    			pruner.printPruningStats();
    		
    		decoder.outputArcCount(bw);
    		
    		// evaluate on a development set
    		if (evalAndSave && options.test && ((iIter+1) % 1 == 0 || iIter+1 == options.maxNumIters)) {		
    			System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
	  			System.out.printf(" Evaluation: %s%n", options.testFile);
	  			System.out.println(); 
                if (options.average) {
                	parameters.averageParameters(updCnt);
                }
	  			double res = evaluateSet(false, false, bw);
                System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
                if (options.average) 
                	parameters.unaverageParameters();
    		} 
    	}
    	
    	if (evalAndSave && options.average) {
            parameters.averageParameters(updCnt);
    	}

    	bw.close();
        decoder.shutdown();
        
        //for (int z = 0; z < parameters.params.length; ++z) {
        //	bw_m.write("" + z + "\t" + parameters.params[z] + "\n");
        //}
        
        bw_f.close();
        bw_m.close();
    }
    
    public int evaluateUnlabelCorrect(DependencyInstance act, DependencyInstance pred) 
    {
    	int nCorrect = 0;
    	for (int i = 1, N = act.length; i < N; ++i) {
    		if (act.heads[i] == pred.heads[i])
    			++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public int evaluateLabelCorrect(DependencyInstance act, DependencyInstance pred) 
    {
    	int nCorrect = 0;
    	for (int i = 1, N = act.length; i < N; ++i) {
    		if (act.heads[i] == pred.heads[i] && act.deplbids[i] == pred.deplbids[i])
    			++nCorrect;
    	}    		  		
    	return nCorrect;
    }
    
    public static int evaluateUnlabelCorrect(DependencyInstance inst, 
    		DependencyInstance pred, boolean evalWithPunc) 
    {
    	int nCorrect = 0;    	
    	for (int i = 1, N = inst.length; i < N; ++i) {

            if (!evalWithPunc)
            	if (inst.forms[i].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;

    		if (inst.heads[i] == pred.heads[i]) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public static int evaluateLabelCorrect(DependencyInstance inst, 
    		DependencyInstance pred, boolean evalWithPunc) 
    {
    	int nCorrect = 0;    	
    	for (int i = 1, N = inst.length; i < N; ++i) {

            if (!evalWithPunc)
            	if (inst.forms[i].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;

    		if (inst.heads[i] == pred.heads[i] && inst.deplbids[i] == pred.deplbids[i]) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public double evaluateSet(boolean output, boolean evalWithPunc, BufferedWriter bw)
    		throws IOException {
    	
    	if (pruner != null) pruner.resetPruningStats();
    	
    	DependencyReader reader = DependencyReader.createDependencyReader(options);
    	reader.startReading(options.testFile);
    	
    	BufferedWriter out = null;
    	if (output && options.outFile != null) {
    		out = new BufferedWriter(
    			new OutputStreamWriter(new FileOutputStream(options.outFile), "UTF8"));
    	}
    	
    	DependencyDecoder decoder = DependencyDecoder.createDependencyDecoder(options);   	
    	int nUCorrect = 0, nLCorrect = 0;
    	int nDeps = 0, nWhole = 0, nSents = 0;
    	
    	int maxErr = 100;
    	int[] numError = new int[maxErr];
    	int[] numErrorWoPunc = new int[maxErr];
    	int token = 0;
    	
		long start = System.currentTimeMillis();

		DependencyInstance inst = pipe.createInstance(reader);    	
    	while (inst != null) {
    		LocalFeatureData lfd = new LocalFeatureData(inst, this, true);
    		GlobalFeatureData gfd = new GlobalFeatureData(lfd); 
    		
    		++nSents;
            
            int nToks = 0;
            if (evalWithPunc)
    		    nToks = (inst.length - 1);
            else {
                for (int i = 1; i < inst.length; ++i) {
                	if (inst.forms[i].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;
                    ++nToks;
                }
            }
            nDeps += nToks;
            token += inst.length - 1;
    		    		
            DependencyInstance predInst = decoder.decode(inst, lfd, gfd, false);

    		int ua = evaluateUnlabelCorrect(inst, predInst, evalWithPunc), la = 0;
    		if (options.learnLabel)
    			la = evaluateLabelCorrect(inst, predInst, evalWithPunc);
    		nUCorrect += ua;
    		nLCorrect += la;
    		if ((options.learnLabel && la == nToks) ||
    				(!options.learnLabel && ua == nToks)) 
    			++nWhole;
    		
    		int corr = evaluateUnlabelCorrect(inst, predInst, true);
    		int err = Math.min(maxErr - 1, inst.length - 1 - corr);
    		numError[err]++;
    		err = Math.min(maxErr - 1, nToks - ua);
    		numErrorWoPunc[err]++;
    		
    		if (out != null) {
    			int[] deps = predInst.heads, labs = predInst.deplbids;
    			String line1 = "", line2 = "", line3 = "", line4 = "";
    			for (int i = 1; i < inst.length; ++i) {
    				line1 += inst.forms[i] + "\t";
    				line2 += inst.postags[i] + "\t";
    				line3 += (options.learnLabel ? pipe.types[labs[i]] : labs[i]) + "\t";
    				line4 += deps[i] + "\t";
    			}
    			out.write(line1.trim() + "\n" + line2.trim() + "\n" + line3.trim() + "\n" + line4.trim() + "\n\n");
    		}
    		
    		inst = pipe.createInstance(reader);
    	}
    	
    	reader.close();
    	if (out != null) out.close();
    	
    	System.out.printf("  Tokens: %d%n", nDeps);
    	System.out.printf("  Sentences: %d%n", nSents);
    	System.out.printf("  UAS=%.6f\tLAS=%.6f\tCAS=%.6f\t[%ds]%n",
    			(nUCorrect+0.0)/nDeps,
    			(nLCorrect+0.0)/nDeps,
    			(nWhole + 0.0)/nSents,
    			(System.currentTimeMillis() - start)/1000);
    	if (options.pruning && options.learningMode != LearningMode.Basic && pruner != null)
    		pruner.printPruningStats();
        
    	//if (bw != null)
    	decoder.outputArcCount(bw);

        decoder.shutdown();
        
        /*
        double sum = 0.0;
        double oracleErr = 0.0;
        int sent = 0;
        for (int i = 0; i < maxErr; ++i) {
        	System.out.print(numError[i] + " ");
        	sum += numError[i] * i;
        	sent += numError[i];
        	if (i > 2)
        		oracleErr += (i - 2) * numError[i];
        }
        System.out.println();
        double avg = sum / sent;
        double var = 0.0;
        for (int i = 0; i < maxErr; ++i) {
        	var += numError[i] * (i - avg) * (i - avg);
        }
        double stddev = Math.sqrt(var / (sent - 1));
        
        System.out.println("avg: " + avg + " std: " + stddev + " oracle: " + (token - oracleErr) / token);

        double sumP = 0.0;
        double oracleErrP = 0.0;
        for (int i = 0; i < maxErr; ++i) {
        	System.out.print(numErrorWoPunc[i] + " ");
        	sumP += numErrorWoPunc[i] * i;
        	if (i > 2)
        		oracleErrP += (i - 2) * numErrorWoPunc[i];
        }
        System.out.println();
        double avgP = sumP / sent;
        double varP = 0.0;
        for (int i = 0; i < maxErr; ++i) {
        	varP += numErrorWoPunc[i] * (i - avgP) * (i - avgP);
        }
        double stddevP = Math.sqrt(varP / (sent - 1));
        
        System.out.println("avgP: " + avgP + " stdP: " + stddevP + " oracleP: " + (nDeps - oracleErrP) / nDeps);
         */
        
    	return (nUCorrect+0.0)/nDeps;
    }
}
