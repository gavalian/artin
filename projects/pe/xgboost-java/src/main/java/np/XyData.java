package np;

public class XyData {
    
    private float[] mFeatures, mLabels;
    private int mFeaturesPerRow, mLabelsPerRow;

    public XyData(float[] f, int featuresPerRow, float[] l, int labelsPerRow) {
        mFeatures = f;
        mFeaturesPerRow  = featuresPerRow;
        mLabels = l;
        mLabelsPerRow  = labelsPerRow;
    }

    public float[] getFeatures() {
        return mFeatures;
    }

    public int featuresPerRow() {
        return mFeaturesPerRow;
    }

    public int labelsPerRow() {
        return mLabelsPerRow;
    }

    public  float[] getLabels() {
        return mLabels;
    }
}
