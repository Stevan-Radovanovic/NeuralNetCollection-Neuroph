
package neurophwines;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;

public class NeurophWines {

    ArrayList<Training> trainings = new ArrayList<Training>();
    double[] learningRates = {0.2,0.4,0.6};
    String[] labels = new String[]{"s1","s2","s3"};
    int hiddenNeurons = 20;
    int input=13;
    int output=3;
    
    public static void main(String[] args) {
        new NeurophWines().run();
    }
    
    public void run () {
        
        System.out.println("Creating. Normalizing. Shuffling. Splitting");
        DataSet data = DataSet.createFromFile("wines.csv", input, output, ",");
        MaxNormalizer max = new MaxNormalizer(data);
        max.normalize(data);
        data.shuffle();
        DataSet yo[] = data.split(0.65,0.35);
        DataSet train = yo[0];
        DataSet test = yo[1];
        
        int numberOfIterations=0;
        int numberOfTrainings = 0;
        
        for(double lr: learningRates) {
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hiddenNeurons,output);
            neuralNet.setLearningRule(new MomentumBackpropagation());
            MomentumBackpropagation mbp = (MomentumBackpropagation) neuralNet.getLearningRule();
            
            mbp.setMomentum(0.6);
            mbp.setLearningRate(lr);
            mbp.setMaxError(0.007);
            mbp.setMaxIterations(500);
            
            neuralNet.learn(train);
            
            double accuracy = calculateAccuracy(neuralNet,test);
            double msne = calculateMsne(neuralNet,test);
            
            numberOfIterations+=mbp.getCurrentIteration();
            numberOfTrainings++;
            
            trainings.add(new Training(neuralNet,msne,accuracy));
        }
        
        System.out.println("Average iterations: " + (double)numberOfIterations/numberOfTrainings);
        serializeMaxAcc();
        
    }

    public void serializeMaxAcc() {
        Training max = trainings.get(0);
        for(Training t : trainings) {
            if(max.accuracy<t.accuracy) {
                max=t;
            }
        }
        System.out.println("Serializing...");
        //max.neuralNetwork.save("maxacc.nnet");
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
           
        ConfusionMatrix cm = new ConfusionMatrix(labels);
        double sum = 0;
        
        for(DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            int actual = getMaxIndexArray(row.getDesiredOutput());
            int predicted = getMaxIndexArray(neuralNet.getOutput());
            
            cm.incrementElement(actual, predicted);
        }
        
        for(int i=0;i<output;i++) {
            int tp = cm.getTruePositive(i);
            int tn = cm.getTrueNegative(i);
            int fp = cm.getFalsePositive(i);
            int fn = cm.getFalseNegative(i);
            sum += (double)(tp+tn)/(tp+tn+fn+fp);
        }
        
        double acc = (double)sum/output;
        System.out.println("Accuracy: " + acc);
        return acc;
        
    }

    private double calculateMsne(MultiLayerPerceptron neuralNet, DataSet test) {
        double sum = 0;
        
        for(DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double[] actual = row.getDesiredOutput();
            double[] predicted = neuralNet.getOutput();
            
            for(int i=0;i<actual.length;i++) {
                sum+=Math.pow(actual[i]-predicted[i],2);
            }
        }
        
             double msne = (double)sum/(2*test.size());
             System.out.println("Msne: " + msne);
             return msne;
    }

    private int getMaxIndexArray(double[] array) {

        int position = 0;
        for(int i=0;i<array.length;i++) {
            if(array[i]>array[position]) position=i;
        }
        
        return position;
        
    }
    
}