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
    
    public static ExtraTrees train(double[] trainX, double[] trainY) {
        
        System.out.println("Training on " + trainX.length / 6 +" rows");
        Pair data_train = prepare_nlabel_dataset(trainX, trainY, 3);
        Matrix features_sec2_train_pos = new Matrix(data_train.input, data_train.input.length/6, 6);
        
        ExtraTrees model = new ExtraTrees(features_sec2_train_pos, trainY, data_train.tasks);
        // // Train trees here
        model.learnTrees(5, 4, 200);
        
        return model;
    }
    
    public static double[] test(ExtraTrees mode, double[] testX, double[] testY) {

        System.out.println("Testing on " + testX.length / 6 +" rows");
        Pair data_test = prepare_nlabel_dataset(testX, testY, 3);
        Matrix features_sec2_test = new Matrix(data_test.input, data_test.input.length/6, 6);

        double[] yhat_val = pos_model.getValuesMT(features_sec2_test, data_test.tasks);

        return yhat_val;
    }

    public static void main(String[] args) throws Exception {
        String filepath = "../data/a_extract_regression_data_1n1p_tb.txt";
        
        // Reads file and creates arrays for positive/negative charge for all and only sector 2 features 
        read(filepath);
                
        TestTrainSet set_sec2_pos = create_train_test_set(sec2_pos_inputs, sec2_pos_outputs, 0.9);
        pos_model = train(set_sec2_pos.trainX, set_sec2_pos.trainY);
        double[] yhat_pos_val = test(pos_model, set_sec2_pos.testX, set_sec2_pos.testY);
        System.out.println("Test Positive model MAE: " + mae( set_sec2_pos.testY, yhat_pos_val));

        TestTrainSet set_sec2_neg = create_train_test_set(sec2_neg_inputs, sec2_neg_outputs, 0.9);
        neg_model = train(set_sec2_neg.trainX, set_sec2_neg.trainY);
        double[] yhat_neg_val = test(neg_model, set_sec2_neg.testX, set_sec2_neg.testY);
        System.out.println("Test Negative model MAE: " + mae( set_sec2_neg.testY, yhat_neg_val));
        
        double[] yhat_pos_all = test(pos_model, all_pos_inputs, all_pos_outputs);
        System.out.println("Eval Positive model MAE: " + mae( all_pos_outputs, yhat_pos_all));
        
        double[] yhat_neg_all = test(neg_model, all_neg_inputs, all_neg_outputs);
        System.out.println("Eval Negative model MAE: " + mae( all_neg_outputs, yhat_neg_all));
        BufferedReader br=new BufferedReader(new FileReader(new File(filepath)));

        File output = new File(filepath+".out");
        BufferedWriter bw = new BufferedWriter(new FileWriter(output));
        
        
        for(int i = 0; i < yhat_pos_all.length/3; i++)
        {
            bw.write(br.readLine());
            for(int j = 0; j < 3; j++) {
                bw.write(" "+yhat_neg_all[i*3 + j]);
            }
            bw.write("\n");

            bw.write(br.readLine());
            for(int j = 0; j < 3; j++) {
                bw.write(" "+yhat_pos_all[i*3 + j]);
            }
            bw.write("\n");
        }
        
        bw.close();
    }

    public static Pair prepare_nlabel_dataset(double[] input, double[] output, int n_labels) {

        int num_rows = output.length / n_labels;
        int num_features_per_row = input.length / num_rows;
        double [] adjusted_input = new double[input.length * n_labels];
        int tasks[] = new int[output.length];
        for(int i = 0; i < num_rows; i++) {
            for(int j = 0; j < n_labels; j++) {
                for(int k = 0; k < num_features_per_row; k++) {
                    adjusted_input[i*num_features_per_row*n_labels + j*num_features_per_row + k ] = input[i*num_features_per_row + k];
                }
                tasks[i*n_labels + j] = j;
            }
        }

        return new Pair(adjusted_input, tasks);
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
                String [] res = line.split(":");
                String [] first_part = res[0].strip().split("[\\s]+");
                int sector = Integer.parseInt(first_part[0]);
                int charge = Integer.parseInt(first_part[1]);
                String [] outputs = res[1].strip().split("[\\s]+");
                String [] res2 = line.split("==>");
                String [] outs = res2[1].strip().split("[\\s]+");
                if(charge == 1) {
                    if(sector == 2) {
                        java.util.Collections.addAll(labels_pos_sec2, dstring_to_darray(outputs, 3));
                        java.util.Collections.addAll(features_pos_sec2, dstring_to_darray(outs, 6));
                    }
                    java.util.Collections.addAll(labels_pos, dstring_to_darray(outputs, 3));
                    java.util.Collections.addAll(features_pos, dstring_to_darray(outs, 6));
                }
                else {
                    if(sector == 2) {
                        java.util.Collections.addAll(labels_neg_sec2, dstring_to_darray(outputs, 3));
                        java.util.Collections.addAll(features_neg_sec2, dstring_to_darray(outs, 6));
                    }
                    java.util.Collections.addAll(labels_neg, dstring_to_darray(outputs, 3));
                    java.util.Collections.addAll(features_neg, dstring_to_darray(outs, 6));
                }
            }
            br.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        all_pos_inputs = alist_to_a(features_pos);
        all_neg_inputs = alist_to_a(features_neg);
        all_pos_outputs = alist_to_a(labels_pos);
        all_neg_outputs = alist_to_a(labels_neg);

        sec2_pos_inputs = alist_to_a(features_pos_sec2);
        sec2_pos_outputs = alist_to_a(labels_pos_sec2);
        sec2_neg_inputs = alist_to_a(features_neg_sec2);
        sec2_neg_outputs = alist_to_a(labels_neg_sec2);
    }

    public static TestTrainSet create_train_test_set(double[] input, double [] output, double ratio) {

        int trainLength = (int) ((input.length / 6) * ratio) ;
        int testLength = (input.length / 6) - trainLength;
        double[] trainX = new double[6 * trainLength];
        double[] trainY = new double[((input.length / 6) - testLength) * 3];
        
        double[] testX = new double[6 * testLength];
        double[] testY = new double[3 * testLength];

        for(int i = 0; i < trainLength * 6; i+= 6) {
            for(int j = 0; j < 6; j++) {
                trainX[i + j] = input[i + j];
            }
        }

        for(int i = 0; i < trainLength * 3; i+=3) {
            for(int j = 0; j < 3; j++) {
                trainY[i + j] = output[i + j];
            }
        }

        for(int i = trainLength * 6; i < trainLength * 6 + testLength * 6; i+= 6) {
            for(int j = 0; j < 6; j++) {
                testX[i - trainLength * 6 + j] = input[i + j];
            }
        }

        for(int i = trainLength * 3; i < trainLength * 3 + testLength * 3; i+=3) {
            for(int j = 0; j < 3; j++) {
                testY[i - trainLength * 3 + j] = output[i + j];
            }
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

    public static double[] alist_to_a(ArrayList<Double> arraylist) {
        double [] array = new double[arraylist.size()];

        for(int i = 0; i < array.length; i++) {
            array[i] = arraylist.get(i);
        }

        return array;
    }

    public static Double[] dstring_to_darray(String[] sa, int num_values) {
        Double[] res = new Double[num_values];
        for(int i = 0; i < num_values; i++) {
            res[i] = Double.parseDouble(sa[i]);
        }

        return res;
    }

    


}
