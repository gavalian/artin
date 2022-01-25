/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.crtcjlab.denoise2d;

import j4np.utils.io.OptionParser;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author gavalian
 */
public class RunDenoise2d {
    
    public int nThreads = 1;
    public String filename = "";
    public int chunkSize = 2000;
    
    public void process(){
        
        List<Thread>  threads = new ArrayList<>();
        
        for(int i = 0; i < nThreads; i++){
            Denoise2dProcess process = new Denoise2dProcess(i+1,filename);
            process.setChunkSize(chunkSize);
            Thread th = new Thread(process);
            threads.add(th);
        }
        
        for(Thread th : threads){ th.start();}
        
        
        boolean keepRun = true;
        while(keepRun==true){
            int count = 0;
            
            try {
                Thread.sleep(2000);
            } catch (InterruptedException ex) {
                Logger.getLogger(RunDenoise2d.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            for(int i = 0; i < threads.size(); i++){
                if(threads.get(i).isAlive()==true) count++;
            }
            if(count==0) keepRun = false;
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss"); 
                       LocalDateTime now = LocalDateTime.now();  
            System.out.printf(">>>> [%s] threads running #%7d/%7d\n", dtf.format(now), count,threads.size());            
        }
        System.out.println("\n>>> exiting.... ");

            
    }
        
    public static void main(String[] args){
        OptionParser parser = new OptionParser();
        parser.addOption("-chunk", "20000", "chunk size to process");
        parser.addOption("-t", "1", "number of threads");
        
        parser.parse(args);
        

        List<String> inputs = parser.getInputList();
        if(inputs.size()<1) {
            System.out.println("\n\nERROR : please provide input file\n");
            parser.printUsage();
            System.exit(0);
        }
        
        RunDenoise2d exec = new RunDenoise2d();
        
        exec.filename = inputs.get(0);
        exec.chunkSize = parser.getOption("-chunk").intValue();
        exec.nThreads = parser.getOption("-t").intValue();
        
        exec.process();
    }
}
