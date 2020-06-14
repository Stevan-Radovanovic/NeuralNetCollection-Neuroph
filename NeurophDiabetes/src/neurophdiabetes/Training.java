/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neurophdiabetes;

import org.neuroph.core.NeuralNetwork;

/**
 *
 * @author jelica
 */
public class Training {
    public NeuralNetwork neuralNet;
    public double accuracy;
    public double msne;

    public Training() {
    }

    public Training(NeuralNetwork neuralNet,double accuracy,double msne) {
        this.neuralNet = neuralNet;
        this.accuracy = accuracy;
        this.msne = msne;
    }



    
    
    
}
