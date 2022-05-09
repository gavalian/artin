package np.xgboost;

// import ml.dmlc.xgboost4j.java.DMatrix;
import np.Dataset;
import np.XyData;
import np.ArgParsing.*;
import np.ArgParsing;
import com.beust.jcommander.JCommander;
import java.io.*;
import java.util.ArrayList;

public class ParticleEvaluation {
    public static void main(String[] args) throws Exception {
        ArgParsing a = new ArgParsing();
        CommandEval eval = a. new CommandEval();
        CommandTrain train =a. new CommandTrain();
        JCommander jc = JCommander.newBuilder()
            .addObject(a)
            .addCommand("train", train)
            .addCommand("eval", eval)
            .build();
        jc.parse(args);

        String parsedCommand = jc.getParsedCommand();
        if(parsedCommand == null)
        {
            jc.usage();
        }
        else if(parsedCommand.equals("eval")) {
            
            String inputFilePath = eval.evalSet;
            String resFilePath = eval.pathResults;

            
            Dataset npdata = new Dataset(inputFilePath);
            npdata.show();
            
            XyData filteredData = npdata.getFiltered(eval.sector, eval.charge);
            System.out.println("Evaluating");
            System.out.println("EvalSet: "+inputFilePath+", ResultsPath: " + resFilePath+", Filter C: " + eval.charge +", S: "+ eval.sector +", ModelPath: " + eval.pathLoad);
            Model m = new Model(3);
            // m.train(filteredData, filteredData);
            m.load("savedModel.ml");
            float[][] res = m.predict(filteredData);

            File output = new File(resFilePath);
            BufferedWriter bw = new BufferedWriter(new FileWriter(output));

            ArrayList<String> in = npdata.getInput();
            for(int i = 0; i < res.length; i++) {
                bw.write(in.get(i));
                for(int j = 0; j < res[i].length; j++) {
                    bw.write(""+res[i][j] + " ");
                }
                bw.write("\n");
            }

            bw.close();
        }
        else if(parsedCommand.equals("train")) {
            String trainFilePath = train.trainingSet;
            String testFilePath = train.testSet;
            String modelSavePath = train.pathSave;

            
            Dataset trainNpData = new Dataset(trainFilePath);
            Dataset testNpData = new Dataset(testFilePath);
            // npdata.show();
            
            XyData trainFilteredData = trainNpData.getFiltered(train.sectorTrain, train.chargeTrain);
            XyData testFilteredData = testNpData.getFiltered(train.sectorEval, train.chargeEval);
            System.out.println("Traing");
            System.out.println("TraingSet: "+trainFilePath+", TestingSet: " + testFilePath+", Filter C_train: " + train.chargeTrain +", S_train: "+ train.sectorTrain + ", C_test: " + train.chargeEval +", S_test: "+ train.sectorEval+ ", SaveModelPath: " + train.pathSave);
            Model m = new Model(3);
            m.train(trainFilteredData, testFilteredData);
            m.save(modelSavePath);
        }
        else {
            jc.usage();
        }
    }
}
