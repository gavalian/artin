package org;
// package main.java;
import java.io.File;
import java.util.ArrayList;
import org.extratrees.*;
import org.extratrees.data.*;
import org.helpers.*;

import java.io.*;

public class App {

    private static double[] sec2_pos_inputs;
    private static double[] sec2_pos_outputs;
    private static double[] sec2_neg_inputs;
    private static double[] sec2_neg_outputs;
    private static double[] all_pos_inputs;
    private static double[] all_neg_inputs;
    private static double[] all_pos_outputs;
    private static double[] all_neg_outputs;
    private static ExtraTrees pos_model;
    private static ExtraTrees neg_model;

    public static void main(String[] args) throws Exception {
        String filepath = "../data/a_extract_regression_data_1n1p_tb.txt";

        // Reads file and creates arrays for positive/negative charge for all and only sector 2 features 
        read(filepath);

        TestTrainSet set_sec2_pos = create_train_test_set(sec2_pos_inputs, sec2_pos_outputs, 0.9);
        Matrix features_sec2_train_pos = new Matrix(set_sec2_pos.trainX, set_sec2_pos.trainX.length/6, 6);
        Matrix features_sec2_test_pos = new Matrix(set_sec2_pos.testX, set_sec2_pos.testX.length/6, 6);
        
        TestTrainSet set_sec2_neg = create_train_test_set(sec2_neg_inputs, sec2_neg_outputs, 0.9);
        Matrix features_sec2_train_neg = new Matrix(set_sec2_neg.trainX, set_sec2_neg.trainX.length/6, 6);
        Matrix features_sec2_test_neg = new Matrix(set_sec2_neg.testX, set_sec2_neg.testX.length/6, 6);


        Matrix features_all_pos = new Matrix(all_pos_inputs, all_pos_inputs.length/6, 6);
        Matrix features_all_neg = new Matrix(all_neg_inputs, all_neg_inputs.length/6, 6);

        pos_model = new ExtraTrees(features_sec2_train_pos, set_sec2_pos.trainY);
        // Train tress here
        pos_model.learnTrees(5, 4, 200);
        double[] yhat_pos_val = pos_model.getValues(features_sec2_test_pos);
        System.out.println("Positive model MAE: " + mae( set_sec2_pos.testY, yhat_pos_val));
        
        double[] yhat_pos_all = pos_model.getValues(features_all_pos);
        
        neg_model = new ExtraTrees(features_sec2_train_neg, set_sec2_neg.trainY);
        // Train tress here
        neg_model.learnTrees(5, 4, 200);
        double[] yhat_neg_val = neg_model.getValues(features_sec2_test_neg);
        System.out.println("Negative model MAE: " + mae( set_sec2_neg.testY, yhat_neg_val));

        double[] yhat_neg_all = neg_model.getValues(features_all_neg);


        BufferedReader br=new BufferedReader(new FileReader(new File(filepath)));

        File output = new File(filepath+".out");
        BufferedWriter bw = new BufferedWriter(new FileWriter(output));
        
        
        for(int i = 0; i < yhat_pos_all.length; i++)
        {
            bw.write(br.readLine()+":"+ all_neg_outputs[i]+","+yhat_neg_all[i]+"\n");
            bw.write(br.readLine()+":"+ all_pos_outputs[i]+","+yhat_pos_all[i]+"\n");
        }
        
        bw.close();
    }

    public static void read(String filepath) {
        ArrayList<Double> features_pos = new ArrayList<Double>();
        ArrayList<Double> features_neg = new ArrayList<Double>();
        ArrayList<Double> labels_pos = new ArrayList<Double>();
        ArrayList<Double> labels_neg = new ArrayList<Double>();
        ArrayList<Double> features_pos_sec2 = new ArrayList<Double>();
        ArrayList<Double> features_neg_sec2 = new ArrayList<Double>();
        ArrayList<Double> labels_pos_sec2 = new ArrayList<Double>();
        ArrayList<Double> labels_neg_sec2 = new ArrayList<Double>();


        try {
            File file = new File(filepath);
            
            BufferedReader br=new BufferedReader(new FileReader(file));
            // System.out.println("File contents");
            String line;
            while((line = br.readLine()) != null) {
                // System.out.println(line);
                String [] res = line.split(":");
                String [] first_part = res[0].strip().split("[\\s]+");
                int sector = Integer.parseInt(first_part[0]);
                int charge = Integer.parseInt(first_part[1]);
                String [] outputs = res[1].strip().split("[\\s]+");
                String [] res2 = line.split("==>");
                String [] outs = res2[1].strip().split("[\\s]+");
                // System.out.println(outs[0]);
                // System.out.println("Outputs size: "+outputs.length);
                // for(String v: outputs) {
                //     System.out.println(v);
                // }
                // System.out.println("Outs size: "+outs.length);
                if(charge == 1) {
                    if(sector == 2) {
                        labels_pos_sec2.add(Double.parseDouble(outputs[0]));
                        features_pos_sec2.add(Double.parseDouble(outs[0]));
                        features_pos_sec2.add(Double.parseDouble(outs[1]));
                        features_pos_sec2.add(Double.parseDouble(outs[2]));
                        features_pos_sec2.add(Double.parseDouble(outs[3]));
                        features_pos_sec2.add(Double.parseDouble(outs[4]));
                        features_pos_sec2.add(Double.parseDouble(outs[5]));
                    }
                    labels_pos.add(Double.parseDouble(outputs[0]));
                    features_pos.add(Double.parseDouble(outs[0]));
                    features_pos.add(Double.parseDouble(outs[1]));
                    features_pos.add(Double.parseDouble(outs[2]));
                    features_pos.add(Double.parseDouble(outs[3]));
                    features_pos.add(Double.parseDouble(outs[4]));
                    features_pos.add(Double.parseDouble(outs[5]));
                }
                else {
                    if(sector == 2) {
                        labels_neg_sec2.add(Double.parseDouble(outputs[0]));
                        features_neg_sec2.add(Double.parseDouble(outs[0]));
                        features_neg_sec2.add(Double.parseDouble(outs[1]));
                        features_neg_sec2.add(Double.parseDouble(outs[2]));
                        features_neg_sec2.add(Double.parseDouble(outs[3]));
                        features_neg_sec2.add(Double.parseDouble(outs[4]));
                        features_neg_sec2.add(Double.parseDouble(outs[5]));
                    }
                    labels_neg.add(Double.parseDouble(outputs[0]));
                    features_neg.add(Double.parseDouble(outs[0]));
                    features_neg.add(Double.parseDouble(outs[1]));
                    features_neg.add(Double.parseDouble(outs[2]));
                    features_neg.add(Double.parseDouble(outs[3]));
                    features_neg.add(Double.parseDouble(outs[4]));
                    features_neg.add(Double.parseDouble(outs[5]));
                }
            }
            br.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        all_pos_inputs = new double[features_pos.size()];
        all_neg_inputs = new double[features_neg.size()];
        all_pos_outputs = new double[labels_pos.size()];
        all_neg_outputs = new double[labels_neg.size()];

        sec2_pos_inputs = new double[features_pos_sec2.size()];
        sec2_pos_outputs = new double[labels_pos_sec2.size()];
        sec2_neg_inputs = new double[features_neg_sec2.size()];
        sec2_neg_outputs = new double[labels_neg_sec2.size()];

        for(int i = 0; i < all_pos_inputs.length; i++) {
            all_pos_inputs[i] = features_pos.get(i);
        }

        for(int i = 0; i < all_neg_inputs.length; i++) {
            all_neg_inputs[i] = features_neg.get(i);
        }

        for(int i = 0; i < all_pos_outputs.length; i++) {
            all_pos_outputs[i] = labels_pos.get(i);
        }

        for(int i = 0; i < all_neg_outputs.length; i++) {
            all_neg_outputs[i] = labels_neg.get(i);
        }

        for(int i = 0; i < sec2_pos_inputs.length; i++) {
            sec2_pos_inputs[i] = features_pos_sec2.get(i);
        }

        for(int i = 0; i < sec2_neg_inputs.length; i++) {
            sec2_neg_inputs[i] = features_neg_sec2.get(i);
        }

        for(int i = 0; i < sec2_pos_outputs.length; i++) {
            sec2_pos_outputs[i] = labels_pos_sec2.get(i);
        }

        for(int i = 0; i < sec2_neg_outputs.length; i++) {
            sec2_neg_outputs[i] = labels_neg_sec2.get(i);
        }
    }

    public static TestTrainSet create_train_test_set(double[] input, double [] output, double ratio) {

        int trainLength = (int) ((input.length / 6) * ratio) ;
        int testLength = (input.length / 6) - trainLength;
        double[] trainX = new double[6 * trainLength];
        double[] trainY = new double[(input.length / 6) - testLength];
        
        double[] testX = new double[6 * testLength];
        double[] testY = new double[testLength];

        for(int i = 0; i < trainLength * 6; i+= 6) {
            for(int j = 0; j < 6; j++) {
                trainX[i + j] = input[i + j];
            }
        }

        for(int i = 0; i < trainLength; i++) {
            // for(int j = 0; j < 6; j++) {
                trainY[i] = output[i];
            // }
        }

        for(int i = trainLength * 6; i < trainLength * 6 + testLength * 6; i+= 6) {
            for(int j = 0; j < 6; j++) {
                testX[i - trainLength * 6 + j] = input[i + j];
            }
        }

        for(int i = trainLength; i < trainLength + testLength; i++) {
            // for(int j = 0; j < 6; j++) {
                testY[i - trainLength] = output[i];
            // }
        }

        return new TestTrainSet(trainX, trainY, testX, testY);
    }

    public static double mae(double[] y, double[] pred) throws Exception {

        if(y.length != pred.length) {
            throw new Exception("Unequal lengths");
        }

        double error = 0;

        for(int i= 0; i < y.length; i++) {
            error += Math.abs(y[i]-pred[i]);
        }

        return error/ y.length;
    }



}
