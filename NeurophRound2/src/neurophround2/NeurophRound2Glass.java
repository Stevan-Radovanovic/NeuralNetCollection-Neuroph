/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neurophround2;

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

/**
 *
 * @author StevanMcfc
 */
public class NeurophRound2Glass {
    
    public static void main(String[] args) {
        new NeurophRound2BreastCancer().run();
    }
    
    //Serilization skipped
    
    int input=9;
    int output=7;
    String[] classLabels = {"s1","s2","s3","s4","s5","s6","s7"};
    ArrayList<Training> trainings = new ArrayList<Training>();
    int[] hidden = {10,20,30};
    double[] learningRates = {0.2,0.4,0.6};
    
    public void run() {
        
        DataSet data = DataSet.createFromFile("glass.csv", input, output, ",");
        MaxNormalizer norm = new MaxNormalizer(data);
        norm.normalize(data);
        data.shuffle();
        
        DataSet[] array = data.split(0.65,0.35);
        DataSet train = array[0];
        DataSet test = array[1];
        
        int numberOfIterations=0;
        int numberOfTrainings=0;
        
        for(double lr: learningRates) {
            for(int hn: hidden) {
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hn,output);
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation mbp = (MomentumBackpropagation) neuralNet.getLearningRule();
                
                mbp.setMomentum(0.6);
                mbp.setLearningRate(lr);
                mbp.setMaxError(0.09);
                mbp.setMaxIterations(200);
                
                neuralNet.learn(train); //!
                
                calculateAccuracy(neuralNet,test);
                calculateAccuracyNeuroph(neuralNet,test);
                
                calculateMsne(neuralNet,test);
                calculateMsneNeuroph(neuralNet,test);
                
                numberOfTrainings++;
                numberOfIterations += mbp.getCurrentIteration();

            }
        }
        
        System.out.println("Iteration average: " + (double)numberOfIterations/numberOfTrainings);
    }

    private void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        eval.evaluate(neuralNet, test);
        
        ConfusionMatrix cm = eval.getEvaluator(ClassifierEvaluator.MultiClass.class).getResult();
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        
        System.out.println("Accuracy Neuroph: " + average.accuracy);
        
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        
         int tp,tn,fp,fn;
         double sumAccuracy=0;
         ConfusionMatrix cm = new ConfusionMatrix(classLabels);
        for (DataSetRow row : test) {
                
            
            
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(neuralNet.getOutput());
            
            cm.incrementElement(actual, predicted);
        }
        
         for(int i=0;i<output;i++) {
            tp=cm.getTruePositive(i);
            tn=cm.getTrueNegative(i);
            fp = cm.getFalsePositive(i);
            fn = cm.getFalseNegative(i);
            
            sumAccuracy += (double)(tp+tn)/(tp+tn+fp+fn);
        }
         
        double accuracy = sumAccuracy/output; 
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
                
        
    }

    private double calculateMsne(MultiLayerPerceptron neuralNet, DataSet test) {  
        
        double sumError=0,msne;
        
        for(DataSetRow row: test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double[] actual = row.getDesiredOutput();
            double[] predicted = neuralNet.getOutput();
            
            for(int i=0;i<actual.length;i++) {
                sumError+=Math.pow(predicted[i]-actual[i],2);
            }
           
            
        }
        
        msne = (double)sumError/(2*test.size());
        System.out.println("Msne: " + msne);
        return msne;
    }

    private void calculateMsneNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        eval.evaluate(neuralNet, test);
        
        System.out.println("Msne Neuroph: " + eval.getMeanSquareError());
    }
    
    private int getMaxIndex(double[] array) {
        int maxIndex = 0;
        double max = array[0];
        for(int i=0;i<array.length;i++) {
            if(max<array[i]) {
                maxIndex = i;
                max = array[i];
            }
        }
        return maxIndex;
    }
    
}
