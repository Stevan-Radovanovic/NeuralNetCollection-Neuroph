/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neurophwines;

import org.neuroph.core.NeuralNetwork;


public class Training {
    
    public double accuracy;
    public double msne;
    public NeuralNetwork neuralNetwork;
    
    public Training() {
        
    }
    
    public Training(NeuralNetwork neuralNetwork, double msne, double accuracy) {
        this.accuracy = accuracy;
        this.msne = msne;
        this.neuralNetwork = neuralNetwork;
    };
    
}
