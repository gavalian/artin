package np;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.*;


@Parameters(commandDescription = "XGBoost CLI")
public class ArgParsing {

    @Parameter(names="--help", help = true)
    public boolean help;
    
    @Parameters(commandDescription = "Train a XGBoost model")
    public class CommandTrain {

        @Parameter(names="--t", description = "Path to training set", required =  true)
        public String trainingSet;

        @Parameter(names = "--e", description = "Path to evaluation set", required =  true)
        public String testSet;

        @Parameter(names = "--m", description = "Path to save trained model", required =  true)
        public String pathSave;

        @Parameter(names = "--ct", description = "Charge to filter training set (0 for no filtering)")
        public float chargeTrain = 1.0f;

        @Parameter(names = "--st", description = "Sector to filter training set (-1 for no filtering)")
        public float sectorTrain = 2.0f;

        @Parameter(names = "--ce", description = "Charge to filter testing set (0 for no filtering)")
        public float chargeEval = 1.0f;

        @Parameter(names = "--se", description = "Sector to filter testing set (-1 for no filtering)")
        public float sectorEval = 2.0f;
    }

    @Parameters(commandDescription = "Use a XGBoost model")
    public class CommandEval {

        @Parameter(names = "--e", description = "Path to evaluation set", required =  true)
        public String evalSet;

        @Parameter(names = "--m", description = "Path to trained model", required =  true)
        public String pathLoad;

        @Parameter(names = "--o", description = "Path to save results", required =  true)
        public String pathResults;

        @Parameter(names = "--c", description = "Charge to filter eval set (0 for no filtering)")
        public float charge = 1.0f;

        @Parameter(names = "--s", description = "Sector to filter eval set (-1 for no filtering)")
        public float sector = -1.0f;
    }
}
