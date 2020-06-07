/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neurophbreastcancer;

import org.neuroph.core.NeuralNetwork;

/**
 *
 * @author StevanMcfc
 */
public class Training {
    
   NeuralNetwork neuralNet;
    double error;

    public Training(NeuralNetwork neuralNet, double error) {
        this.neuralNet = neuralNet;
        this.error = error;
    }
    
}
