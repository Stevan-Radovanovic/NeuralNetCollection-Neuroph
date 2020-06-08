package neurophglass;

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


public class NeurophGlass {

    int input = 9;
    int output = 7;
    ArrayList<Training> array = new ArrayList<Training>();
    double[] lr = {0.2,0.4,0.6};
    int[] hn = {10,20,30};
    
    public static void main(String[] args) {
        new NeurophGlass().run();
    }
    
    public void run() {
        
        DataSet data = DataSet.createFromFile("glass.csv", input, output, ",");
        MaxNormalizer max = new MaxNormalizer(data);
        max.normalize(data);
        data.shuffle();
        
        DataSet[] split = data.split(0.65,0.35);
        DataSet train = split[0];
        DataSet test = split[1];
        
        int iterations=0;
        int trainings=0;
        
        for(double l: lr) {
            for(int h: hn) {
                System.out.println("Learning rate: " + l + ". Hidden Neurons: " + h);
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,h,output);
                
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation bp = (MomentumBackpropagation) neuralNet.getLearningRule();
                
                bp.setMomentum(0.6);
                bp.setMaxError(0.07);
                bp.setLearningRate(l);

                neuralNet.learn(train);
                
                double accuracy = calculateAccuracy(neuralNet,test);
                calculateAccuracyNeuroph(neuralNet, test);
                
                trainings++;
                iterations+=bp.getCurrentIteration();
                
                array.add(new Training(accuracy,neuralNet));
            }
        }
        
        System.out.println("Iteration average: " + (double)iterations/trainings);
        serialize();
    }
    
    public void serialize() {
        
        Training max = array.get(0);
        for(Training t : array) {
            if(max.accuracy<t.accuracy) {
                max = t;
            }
        }
        System.out.println("Saving file");
        max.neuralNet.save("file.nnet");
        
    }
    
    public double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        
        //ovo je kod ako imamo samo jednu izlaznu
        /*
        int tp=0,fp=0,fn=0,tn=0;
        
        for(DataSetRow row : test) {
            
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double actual = row.getDesiredOutput()[0];
            double predicted = neuralNet.getOutput()[0];
         
             if (actual == 1.0 && predicted > 0.5) {
                tp++;
            }
            if (actual == 0.0 && predicted <= 0.5) {
                tn++;
            }
            if (actual == 1.0 && predicted <= 0.5) {
                fn++;
            }
            if (actual == 0.0 && predicted > 0.5) {
                fp++;
            }
            
        }
        double accuracy = (double)(tp+tn)/(tp+tn+fn+fp);
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    */
        String[] classLabels = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"};
        ConfusionMatrix mk = new ConfusionMatrix(classLabels);

        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            int actual = getMaxInArray(row.getDesiredOutput());
            int predicted = getMaxInArray(neuralNet.getOutput());

            mk.incrementElement(actual, predicted);
        }
        
        System.out.println(mk.toString());

        double accuracy = 0;

        for (int i = 0; i < output; i++) {
            int tp = mk.getTruePositive(i);
            int tn = mk.getTrueNegative(i);
            int fp = mk.getFalsePositive(i);
            int fn = mk.getFalseNegative(i);

            accuracy += (double) (tp + tn) / (tp + tn + fp + fn);
        }

        double averageAcc = (double) accuracy / output;

        System.out.println("Accuracy: " + averageAcc);

        return averageAcc;
    }
    
    public void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        String[] classLabels = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        evaluation.evaluate(neuralNet, test);

        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
        ConfusionMatrix cm = evaluator.getResult();
        System.out.println(cm.toString());

        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);

        System.out.println("Accuracy neuroph: " + average.accuracy);
    }
    
     private int getMaxInArray(double[] array) {
        int position = 0;

        for (int i = 1; i < array.length; i++) {
            if (array[position] < array[i]) {
                position = i;
            }
        }

        return position;
    }
    
}
