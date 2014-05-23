package parser.pruning;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import parser.DependencyInstance;
import parser.DependencyParser;
import parser.DependencyPipe;
import parser.GlobalFeatureData;
import parser.LocalFeatureData;
import parser.LowRankParam;
import parser.Options;
import parser.Parameters;
import parser.Options.LearningMode;
import parser.decoding.DependencyDecoder;
import utils.FeatureVector;

public class BasicArcPruner extends DependencyParser {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	@Override
	public void train(DependencyInstance[] lstTrain) 
	    	throws IOException, CloneNotSupportedException 
	{
    	long start = 0, end = 0;
    	        
		System.out.println("=============================================");
		System.out.printf(" Training Pruner:%n");
		System.out.println("=============================================");
		
		start = System.currentTimeMillis();

		System.out.println("Running MIRA ... ");
		trainIter(lstTrain);
		System.out.println();
		
		end = System.currentTimeMillis();
		
		System.out.printf("Training took %d ms.%n", end-start);    		
		System.out.println("=============================================");
		System.out.println();		    	
	}
	
	public void trainIter(DependencyInstance[] lstTrain) throws IOException
	{
		int N = lstTrain.length;
		int updCnt = 0;
    	
    	for (int iIter = 0; iIter < options.maxNumIters; ++iIter) {
			
        	long start = 0;
			double loss = 0;
			int uas = 0, tot = 0;
			start = System.currentTimeMillis();
			
			for (int i = 0; i < N; ++i) {
    			
				DependencyInstance inst = lstTrain[i];
    			LocalFeatureData lfd = new LocalFeatureData(inst, this, true);
    		    int n = inst.length;
    		    
    		    for (int m = 1; m < n; ++m) {
    		    	
    		    	int goldhead = inst.heads[m];
    		    	FeatureVector goldfv = lfd.getArcFeatureVector(goldhead, m);
    		    	//FeatureVector goldfv = lfd.getFeatureVector(inst);
    		    	double goldscore = parameters.dotProduct(goldfv);
    		    	
    		    	int predhead = -1;
    		    	FeatureVector predfv = null;
    		    	double best = Double.NEGATIVE_INFINITY;
    		    	
    		    	
    		    	for (int h = 0; h < n; ++h)
    		    		if (h != m) {
    		    			FeatureVector fv = lfd.getArcFeatureVector(h, m);
    		    			double va = parameters.dotProduct(fv)
    		    					+ (h != goldhead ? 1.0 : 0.0);
    		    			if (va > best) {
    		    				best = va;
    		    				predhead = h;
    		    				predfv = fv;
    		    			}
    		    		}
    		    	
    		    	
    		    	/*
					for (int h = 0; h < n; ++h)
						if (h != m && !lfd.isPruned(h, m)
						&& !isAncestorOf(inst.heads, m, h)) {
							inst.heads[m] = h;
							FeatureVector fv = lfd.getFeatureVector(inst);
    		    			double va = parameters.dotProduct(fv)
    		    					+ (h != goldhead ? 1.0 : 0.0);
							if (va > best) {
    		    				best = va;
    		    				predhead = h;
    		    				predfv = fv;
 							}
						}
   		    	*/
    		    	if (goldhead != predhead) {
    		    		++updCnt;
    		    		loss += best - goldscore;
    		    		parameters.updateTheta(goldfv, predfv, best - goldscore, updCnt);
    		    	} else ++uas;
    		    	
    		    	inst.heads[m] = goldhead;
    		    	++tot;
    		    }
			}
			
    		System.out.printf("  Iter %d\tloss=%.4f\tuas=%.4f\t[%ds]%n", iIter+1,
    				loss, uas/(tot+0.0),
    				(System.currentTimeMillis() - start)/1000);
    		
    		{
	    		System.out.println();
	    		System.out.println("_____________________________________________");
	    		System.out.println();
	    		System.out.printf(" Evaluation: %s%n", options.testFile);
	    		System.out.println(); 
	    		if (options.average) {
	    			parameters.averageParameters(updCnt);
	    		}
	    		double res = evaluateSet(false, false, null);
	    		System.out.println();
	    		System.out.println("_____________________________________________");
	    		System.out.println();
	    		if (options.average) 
	    			parameters.unaverageParameters();
    		}
    	}
    	
		if (options.average) {
			parameters.averageParameters(updCnt);
		}

	}
	 	
	private boolean isAncestorOf(int[] heads, int par, int ch) 
	{
		//int cnt = 0;
		while (ch != 0) {
			if (ch == par) return true;
			ch = heads[ch];
		}
		return false;
	}
}
