package neurophglass;

import java.lang.reflect.Constructor;
import org.neuroph.core.NeuralNetwork;

public class Training {
    
    public double accuracy;
    public NeuralNetwork neuralNet;
    public double msne;
    
    public Training() {
        
    }
    
    public Training( NeuralNetwork neuralNet, double msne, double accuracy) {
        this.accuracy = accuracy;
        this.neuralNet = neuralNet;
        this.msne = msne;
    }
    
}
