/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neurophsweden;

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
public class NeurophSweden {

    int input = 1;
    int output = 1;
    double[] learningRates = {0.2,0.4,0.6};
    int[] hiddenNeurons = {30,20,10};
    ArrayList<Training> trainings = new ArrayList<Training>();
    
    
    public static void main(String[] args) {
        new NeurophSweden().run();
    }
    
    public void run() {
                
        System.out.println("Creating data set...");
        DataSet data = DataSet.createFromFile("sweden.txt", input, output, "\t");
        System.out.println("Normalizing and shuffling data...");
        MaxNormalizer norm = new MaxNormalizer(data);
        norm.normalize(data);
        data.shuffle();
    
        System.out.println("Spliting data set...");
        DataSet[] yo = data.split(0.7,0.3);
        DataSet train = yo[0];
        DataSet test = yo[1];
        
        int noOfTrainings=0;
        int noOfIterations=0;
        
        for(double lr : learningRates) {
            for(int hn : hiddenNeurons) {
                
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hn,output);                
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation mbp = (MomentumBackpropagation) neuralNet.getLearningRule();
                mbp.setMaxError(0.07);
                mbp.setLearningRate(lr);
                mbp.setMomentum(0.6);
                
                neuralNet.learn(train);
                
                noOfTrainings++;
                noOfIterations+=mbp.getCurrentIteration();
                
                calculateAccNeuroph(neuralNet,test);
                double accuracy = calculateAcc(neuralNet,test);
              
                calculateAccMsne(neuralNet,test);
                double msne = calculateMsne(neuralNet,test);
                
                Training t = new Training();
                t.neuralNet = neuralNet;
                t.accuracy = accuracy;
                t.msne = msne;
                
                trainings.add(t);
            } 
        }
        
         serializeMaxMsne();
         System.out.println("Number of average iterations: " + (double)noOfIterations/noOfTrainings);
        
    }

    private void serializeMaxMsne() {
    }

    private void calculateAccNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, test);
        
        ConfusionMatrix cm = evaluation.getEvaluator(ClassifierEvaluator.Binary.class).getResult();
        System.out.println(cm.toString());
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        
        System.out.println("Accuracy neuroph: " + average.accuracy);

    }

    private double calculateAcc(MultiLayerPerceptron neuralNet, DataSet test) {
        
        int tp=0,tn=0,fp=0,fn=0;
        
        for(DataSetRow row : test) {
            
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double actual = row.getDesiredOutput()[0];
            double predicted = neuralNet.getOutput()[0];
            
            if(actual>0.5 && predicted>0.5) tp++;
            if(actual<=0.5 && predicted<=0.5) tn++;
            if(actual>0.5 && predicted<=0.5) fn++;
            if(actual<=0.5 && predicted>0.5) fp++;
        }
        
        double acc = (double)(tp+tn)/(tp+tn+fp+fn);
        System.out.println("Accuracy: " + acc);
        return acc;
    }

    private void calculateAccMsne(MultiLayerPerceptron neuralNet, DataSet test) {
    }

    private double calculateMsne(MultiLayerPerceptron neuralNet, DataSet test) {
        return 0;
    }
    
}
