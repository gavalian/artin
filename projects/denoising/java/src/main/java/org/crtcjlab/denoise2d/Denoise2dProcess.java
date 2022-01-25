/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.crtcjlab.denoise2d;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author gavalian
 */
public class Denoise2dProcess implements Runnable {

    private int         processid = 0;
    private String        jarFile = "denoise2d-1.0-jar-with-dependencies.jar";
    private String memorySettings = "-Xmx2048m -Xms1024m";
    private String       filename = "";
    public  int         chunkSize = 1000;
    public  int         chunk     = 0;
    public Denoise2dProcess(int pid, String fn){
        this.processid = pid; chunk = pid;
        this.filename  = fn;
    }
    
    public void setChunkSize(int cz){ chunkSize = cz;}
    public void setChunk(int c){ chunk = c;}
    
    public String getCommandString(){
        StringBuilder str = new StringBuilder();
        str.append(String.format("java -cp %s org.crtcjlab.denoise2d.clas12.Clas12Denoiser ",jarFile));
        str.append(memorySettings);
        str.append(String.format(" -o %s.dn.%d.hipo ", filename,processid));
        int skip    = chunkSize*(chunk-1);
        int nevents = chunkSize;
        str.append(String.format(" -skip %d -n %d ",skip,nevents));
        str.append(filename);
        return str.toString();
    }
    
    @Override
    public void run() {
        System.out.printf(">>>> starting thread # %5d\n",processid);
        String command = getCommandString();
        System.out.printf(">>>> command line # %d : %s\n",processid,command);
        try
        {
             BufferedReader is;  // reader for output of process
             String line;
            // Command to create an external process            
            // Running the above command
            Runtime run  = Runtime.getRuntime();
            Process proc = run.exec(command);
            is = new BufferedReader(new InputStreamReader(proc.getInputStream()));

            while ((line = is.readLine()) != null){
                System.out.println(line);
            }
            
            proc.waitFor();
        }  
        catch (IOException e)
        {
            e.printStackTrace();
        } catch (InterruptedException ex) {
            Logger.getLogger(Denoise2dProcess.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.printf(">>>> finished thread # %5d\n",processid);
    }
    
}
