package np;
import java.io.*;
import java.util.ArrayList;

public class Dataset {
    private String mPath;
    private float [] mFeatures;
    private float [] mLabels;
    private float [] mSectors, mCharges;
    private ArrayList<String> mInput;
    private static final int NUM_FEATURES = 6;
    private static final int NUM_OUTPUTS = 3;

    public Dataset(String path) throws IOException {
        mPath = path;
        BufferedReader br = new BufferedReader(new FileReader(new File(mPath)));
        
        String line;
        ArrayList<Float[]> feats = new ArrayList<>();
        mInput = new ArrayList<>();
        while((line = br.readLine()) != null)
        {
            mInput.add(line);
            feats.add(parseLine(line));
        }
        mFeatures = new float[feats.size() * NUM_FEATURES];
        mLabels = new float[feats.size() * NUM_OUTPUTS];
        mCharges = new float[feats.size()];
        mSectors = new float[feats.size()];

        for(int i = 0; i < mFeatures.length / NUM_FEATURES; i++){
            
            for(int j = 0; j < NUM_FEATURES; j++) {
                mFeatures[i * NUM_FEATURES + j] =  feats.get(i)[j];
            }
            
            for(int j = 0; j < NUM_OUTPUTS; j++) {
                mLabels[i * NUM_OUTPUTS + j] =  feats.get(i)[NUM_FEATURES + j];
            }

            mSectors[i] = feats.get(i)[NUM_FEATURES + NUM_OUTPUTS];
            mCharges[i] = feats.get(i)[NUM_FEATURES + NUM_OUTPUTS + 1];
        } 
    }

    public void show(int lines) {
        System.out.println("Chrg   Sec | \t\t\t    Features    \t\t\t|\t\tLabels \t ");
        for(int i = 0; i < lines && i < mFeatures.length; i++) {
            System.out.printf("%4.1f  %4.1f |", mCharges[i], mSectors[i]);
            System.out.printf("%8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  | ", mFeatures[i * NUM_FEATURES + 0], mFeatures[i * NUM_FEATURES + 1], mFeatures[i * NUM_FEATURES + 2], mFeatures[i * NUM_FEATURES + 3], mFeatures[i * NUM_FEATURES + 4], mFeatures[i * NUM_FEATURES + 5]);
            System.out.printf("%7.3f  %8.3f  %8.3f\n", mLabels[i * NUM_OUTPUTS + 0], mLabels[i * NUM_OUTPUTS + 1], mLabels[i * NUM_OUTPUTS + 2]);
        }
    }

    public void show() {
        show(5);
    }

    private Float[] parseLine(String line) {

        String [] res = line.split(":");
        String [] first_part = res[0].strip().split("[\\s]+");
        Float sector = Float.parseFloat(first_part[0]);
        Float charge = Float.parseFloat(first_part[1]);
        String [] outputs = res[1].strip().split("[\\s]+");
        String [] res2 = line.split("==>");
        String [] outs = res2[1].strip().split("[\\s]+");
        Float[] features = new Float[NUM_FEATURES + NUM_OUTPUTS + 2];
        for(int i = 0; i < NUM_FEATURES; i++)
        {
            features[i] = Float.parseFloat(outs[i]);
        }

        for(int i = 0; i < NUM_OUTPUTS; i++)
        {
            features[NUM_FEATURES + i] = Float.parseFloat(outputs[i]);
        }

        features[NUM_FEATURES + NUM_OUTPUTS] = sector;
        features[NUM_FEATURES + NUM_OUTPUTS + 1] = charge;
        
        return features;
    }

    public XyData getFiltered(double sector, double charge) {
        ArrayList<Integer> indices = new ArrayList<>();

        for(int i = 0; i < mSectors.length; i++) {
            if((sector == -1 || mSectors[i] == sector) && (charge == 0 || mCharges[i] == charge)) {
                indices.add(i);
            }
        }
        ArrayList<String> filteredInput = new ArrayList<>();


        float retFeatures[] = new float[indices.size() * NUM_FEATURES];
        float retLabeles[] = new float[indices.size() * NUM_OUTPUTS];
        int curr_idx = 0;
        for(int idx: indices) {
            filteredInput.add(mInput.get(idx));
            for(int i = 0; i < NUM_FEATURES; i++) {
                retFeatures[curr_idx * NUM_FEATURES + i] = mFeatures[idx * NUM_FEATURES + i];
            }
            for(int i = 0; i < NUM_OUTPUTS; i++) {
                retLabeles[curr_idx + i * indices.size()] = mLabels[idx * NUM_OUTPUTS + i];
            }
            curr_idx++;
        }

        mInput = filteredInput;
        return new XyData(retFeatures, NUM_FEATURES, retLabeles, NUM_OUTPUTS);
    }

    public ArrayList<String> getInput() {
        return mInput;
    }

}
