package neurophglass;

import java.lang.reflect.Constructor;
import org.neuroph.core.NeuralNetwork;

public class Training {
    
    public double accuracy;
    public NeuralNetwork neuralNet;
    
    public Training() {
        
    }
    
    public Training(double accuracy, NeuralNetwork neuralNet) {
        this.accuracy = accuracy;
        this.neuralNet = neuralNet;
    }
    
}
