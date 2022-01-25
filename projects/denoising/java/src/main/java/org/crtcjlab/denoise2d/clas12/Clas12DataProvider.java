/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.crtcjlab.denoise2d.clas12;

import j4np.data.base.DataFrame;
import j4np.data.base.DataStream;
import j4np.data.base.DataWorker;
import j4np.hipo5.data.Bank;
import j4np.hipo5.data.Event;
import j4np.hipo5.data.Schema;
import j4np.hipo5.io.HipoReader;
import j4np.hipo5.io.HipoWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.crtcjlab.denoise2d.models.DenoisingAutoEncoder;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author gavalian
 */
public class Clas12DataProvider extends DataWorker<HipoReader,Event> {
   
    private Schema schema = null;
    private DenoiserModelPool denoiserPool = new DenoiserModelPool(8);
    DenoisingAutoEncoder model = null;
    
    public Clas12DataProvider(){
        model = new DenoisingAutoEncoder();
        try {
            model.loadKerasModel("models/cnn_autoenc_config.json", "models/cnn_autoenc_weights.h5");
        } catch (UnsupportedKerasConfigurationException ex) {
            Logger.getLogger(Clas12Denoiser.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Clas12Denoiser.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InvalidKerasConfigurationException ex) {
            Logger.getLogger(Clas12Denoiser.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    public List<INDArray>  getDataFeatures(Bank bank){
        int x = 36;
        int y = 112;
        //long then = System.currentTimeMillis();
        List<INDArray>  list = new ArrayList<>();
        for(int i = 0; i < 6; i++){
            list.add(Nd4j.zeros(1,x,y,1));
        }
        int nrows = bank.getRows();
        for(int i = 0; i < nrows; i++){
            int    sector = bank.getInt("sector", i);
            int     layer = bank.getInt("layer", i);
            int component = bank.getInt("component", i);            
            list.get(sector-1).putScalar(new int[]{0,layer-1,component-1,0}, 1.0);
        }
        //long now = System.currentTimeMillis();
        
        return list;
    }

    @Override
    public boolean init(HipoReader r) {
        schema = r.getSchemaFactory().getSchema("DC::tdc"); return true;
    }
    
    protected List<Clas12Denoiser.TDC> getList(Bank bank, int sector, INDArray array){
        List<Clas12Denoiser.TDC> list = new ArrayList<>();
        
        int nrows = bank.getRows();
        for(int i = 0; i < nrows; i++){
            int sec = bank.getInt("sector", i);
            if(sec==sector){
                int layer = bank.getInt("layer", i);
                int  wire = bank.getInt("component",i);
                if(array.getDouble(new int[]{0,layer-1,wire-1,0})>0.5){
                    Clas12Denoiser.TDC entry = new Clas12Denoiser.TDC();
                    entry.sector = (byte) sector;
                    entry.layer = (byte) layer;
                    entry.component = (short) wire;
                    entry.order = bank.getByte("order", i);
                    entry.tdc = bank.getInt("TDC", i);
                    list.add(entry);
                }
            }
        }
        return list;
    }

    public Bank reduce(Bank dc, List<INDArray> prediction){

        List<Clas12Denoiser.TDC>  entries = new ArrayList<>();    
        for(int i = 0; i < 6; i++){
            List<Clas12Denoiser.TDC> entry = this.getList(dc, i+1, prediction.get(i));
            entries.addAll(entry);
        }        
        int nrows = entries.size();
        Bank dcnuevo = new Bank(dc.getSchema(),nrows);
        for(int row = 0; row < nrows; row++){
            Clas12Denoiser.TDC entry = entries.get(row);
            dcnuevo.putByte("sector", row, entry.sector);
            dcnuevo.putByte("layer", row, entry.layer);
            dcnuevo.putShort("component", row, entry.component);
            dcnuevo.putByte("order", row, entry.order);
            dcnuevo.putInt("TDC", row, entry.tdc);                        
        }
       
        return dcnuevo;
    }
    
    @Override
    public void execute(Event event) {
        
        //DenoisingAutoEncoder modelp = this.denoiserPool.borrowObject();
        Bank dc = new Bank(schema);
        event.read(dc);
        
        List<INDArray>  features = this.getDataFeatures(dc);         
        List<INDArray>    output = model.predict(features, 2, 0, 0.05, false);
        
        //this.denoiserPool.returnObject(model);
        
        Bank dcnuevo = this.reduce(dc, output);
        event.write(dcnuevo);
    }
    
    
    public static void main(String[] args){
        
        String  filename = "/Users/gavalian/Work/DataSpace/evio/clas_003852.evio.981.hipo";
        int   nFrameSize = 128;
        int     nThreads = 1;
        
        DataStream<HipoReader,HipoWriter,Event> stream = new DataStream<>();
        
        HipoReader r = new HipoReader(filename);
        HipoWriter w = new HipoWriter();
        w.getSchemaFactory().copy(r.getSchemaFactory());
        w.open("output.denoised.h5");
        
        DataFrame<Event>  frame = new DataFrame<>();
        
        for(int i = 0; i < nFrameSize; i++) frame.addEvent(new Event());
        
        Clas12DataProvider consumer = new Clas12DataProvider();
        
        stream.threads(nThreads);
        stream.frame(frame).source(r).consumer(consumer);
        stream.withOutput(w);
        
        stream.show();
        stream.run();
        
        stream.show();
    }
}
