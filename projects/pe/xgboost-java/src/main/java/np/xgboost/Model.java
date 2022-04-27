package np.xgboost;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import np.XyData;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;

public class Model {
    private HashMap<String, Object> mDefaultParams;
    private int mRounds = 100, mOutputs;
    private Booster[] mTrainedModelFeat;
    // # return MultiOutputRegressor(xgb.XGBRegressor(objective ='reg:squarederror', n_jobs = 10, verbose = 1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=6, min_child_weight=7, n_estimators=1300, num_parallel_tree=4))

    public Model(int numOutputs, HashMap<String, Object> params) {
        mDefaultParams = params; 
        mOutputs = numOutputs;       
    }

    public Model(int numOutputs) {
        mDefaultParams = new HashMap<String, Object>();
        mRounds = 1300;
        mDefaultParams.put("eta", 0.1);
        mDefaultParams.put("max_depth", 6);
        mDefaultParams.put("objective", "reg:squarederror");
        mDefaultParams.put("max_delta_step", 0);
        mDefaultParams.put("gamma", 0);
        mDefaultParams.put("min_child_weight", 7);
        mDefaultParams.put("num_parallel_tree", 4);
        mOutputs = numOutputs;
        mTrainedModelFeat = new Booster[mOutputs];
    }

    public void train(XyData trainMat, XyData testMat) throws XGBoostError {
        HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
        float [] featuresTrain = trainMat.getFeatures();
        float [] featuresTest = testMat.getFeatures();
        float [] trainLabels = trainMat.getLabels();
        float [] testLabels = testMat.getLabels();

        float [] labels1 = new float[trainLabels.length/trainMat.labelsPerRow()];
        float [] labels2 = new float[testLabels.length/testMat.labelsPerRow()];
        
        DMatrix dFeaturesTrain = new DMatrix(featuresTrain, featuresTrain.length/trainMat.featuresPerRow(), trainMat.featuresPerRow(), 0.0f);
        DMatrix dFeaturesTest = new DMatrix(featuresTest, featuresTest.length/testMat.featuresPerRow(), testMat.featuresPerRow(), 0.0f);
        watches.put("train", dFeaturesTrain);
        watches.put("test", dFeaturesTest);

        System.out.println("Num outputs: " + mOutputs);
       
        for(int m = 0; m < mOutputs; m++) {

            for(int i = 0 ; i < labels1.length; i++) {
                labels1[i] = trainLabels[m * labels1.length + i];
            }
            dFeaturesTrain.setLabel(labels1);
            
            for(int i = 0 ; i < labels2.length; i++) {
                labels2[i] = testLabels[m * labels2.length + i];
            }
            dFeaturesTest.setLabel(labels2);

            mTrainedModelFeat[m] = XGBoost.train(dFeaturesTrain, mDefaultParams, mRounds, watches, null, null);
        }
    }

    public float[][] predict(XyData testMat) throws XGBoostError{
        float [] featuresTest = testMat.getFeatures();
        float [][][] res = new float[mOutputs][][];
        DMatrix dFeaturesTest = new DMatrix(featuresTest, featuresTest.length/testMat.featuresPerRow(), testMat.featuresPerRow(), 0.0f);

        for(int i = 0; i < mOutputs; i++) {
            res[i] = mTrainedModelFeat[i].predict(dFeaturesTest);
        }

        float[][] ret = new float[featuresTest.length/testMat.featuresPerRow()][testMat.labelsPerRow()];

        for(int j = 0; j < mOutputs; j++) {
            for(int i = 0; i < ret.length; i++) {
                ret[i][j] = res[j][i][0]; 
            }
        }

        return ret;
    }

    public void save(String path) throws XGBoostError, IOException{
        boolean flag = new File(path+".ml").mkdirs();
        if(flag) {
            for(int i = 0; i < mOutputs; i++) {
                Booster m = mTrainedModelFeat[i];
                m.saveModel(path+".ml"+"/m"+i);
            }
        }
        else {
            System.out.println("Failed to save model");
        }
    }

    public void load(String path) throws XGBoostError{
        int numModels = new File(path).list().length;
        mTrainedModelFeat = new Booster[numModels];
        mOutputs = numModels;

        for(int i = 0; i < numModels; i++) {
            mTrainedModelFeat[i] = XGBoost.loadModel(path+"/m"+i);
        }
    }


}

