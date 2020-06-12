
package neurophround2;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
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
        
        System.out.println("Starting......");
        DataSet data = DataSet.createFromFile("wines.csv", input, output, ",");
        MaxNormalizer max = new MaxNormalizer(data);
        max.normalize(data);
        data.shuffle();
        
        DataSet[] yo = data.split(0.7,0.3);
        DataSet train = yo[0];
        DataSet test = yo[1];
        
        for(double lr: learningRates) {
            System.out.println("Training....");
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hiddenNeurons,output);
            neuralNet.setLearningRule(new BackPropagation());
            BackPropagation bp = neuralNet.getLearningRule();
            bp.setLearningRate(lr);
            bp.setMaxError(0.02);
            
            neuralNet.learn(train);
            
            double acc = calculateAccuracy(neuralNet,test);
            calculateAccuracyNeuroph(neuralNet, test);
            
            Training t = new Training();
            t.neuralNet = neuralNet;
            t.accuracy = acc;
            
            System.out.println("Number of iterations for learning rate " + lr + " is " + bp.getCurrentIteration());
        }
        
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        
        int tp,tn,fp,fn;
        ConfusionMatrix cm = new ConfusionMatrix(new String[]{"s1","s2","s3"});
        double sum=0;
        
        for(DataSetRow row : test) {
            
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            int actual = getMax(row.getDesiredOutput());
            int predicted = getMax(neuralNet.getOutput()); 
         
            cm.incrementElement(actual, predicted);
        }
        
        
        for(int i=0;i<output;i++) {
            tp = cm.getTruePositive(i);
            tn = cm.getTrueNegative(i);
            fp = cm.getFalsePositive(i);
            fn = cm.getFalseNegative(i);
            double accuracy = (double)(tp+tn)/(tp+tn+fp+fn);
            sum+=accuracy;
        }
     
        System.out.println(cm);
        
        double acc = (double)sum/output;
        System.out.println("Accuracy: " + acc);
        return acc;
    }

    private void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.MultiClass(new String[]{"s1","s2","s3"}));
        eval.evaluate(neuralNet, test);
        
        ConfusionMatrix cm = eval.getEvaluator(ClassifierEvaluator.MultiClass.class).getResult();
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        System.out.println(cm);
        
        System.out.println("Accuracy Neuroph: " + average.accuracy);
    }

    private int getMax(double[] desiredOutput) {
            int max = 0;
            for(int i=0;i<desiredOutput.length;i++) {
                    if(desiredOutput[max]<desiredOutput[i]) {
                        max=i;
                    }
                }
            return max;
    }
    
}
