
package neurophround2;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.MaxNormalizer;


public class NeurophRound2Wines {
    
    int input = 13;
    int output = 3;
    ArrayList<Training> trainings = new ArrayList<Training>();
    int hiddenNeurons = 22;
    double[] learningRates = {0.2,0.4,0.6};
    
    public void run() {
        
        DataSet data = DataSet.createFromFile("wines.csv", input, output, ",");
        MaxNormalizer max = new MaxNormalizer(data);
        max.normalize(data);
        data.shuffle();
        
        DataSet[] yo = data.split(0.7,0.3);
        DataSet train = yo[0];
        DataSet test = yo[1];
        
        for(double lr: learningRates) {
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hiddenNeurons,output);
            neuralNet.setLearningRule(new BackPropagation());
            BackPropagation bp = neuralNet.getLearningRule();
            bp.setLearningRate(lr);
            bp.setMaxError(0.02);
        }
        
    }
    
}
