package org.helpers;
public class TestTrainSet {
    
    public double[] trainX, trainY, testX, testY;

    public TestTrainSet(double[] trainX, double[] trainY, double[] testX, double[] testY) {
        this.trainX = trainX;
        this.trainY = trainY;
        this.testX = testX;
        this.testY = testY;
    }
}
