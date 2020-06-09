
package neurophwines;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;

public class NeurophWines {

    ArrayList<Training> trainings = new ArrayList<Training>();
    double[] learningRates = {0.2,0.4,0.6};
    int[] hiddenNeurons = {20};
    int input=13;
    int output=3;
    
    public static void main(String[] args) {
        new NeurophWines().run();
    }
    
    public void run () {
        
        DataSet data = DataSet.createFromFile("wines.csv",input,output,",");
        MaxNormalizer norm = new MaxNormalizer(data);
        norm.normalize(data);
        data.shuffle();
        
        int numOfTrainings=0, numOfIterations=0;
        
        DataSet[] yo = data.split(0.7,0.3);
        DataSet train = yo[0];
        DataSet test = yo[1];
        
        for(double lr: learningRates) {
            for(int hn: hiddenNeurons) {
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hn,output);
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation mbp = (MomentumBackpropagation) neuralNet.getLearningRule();
                mbp.setLearningRate(lr);
                mbp.setMaxError(0.02);
                mbp.setMomentum(0.6);
                
                neuralNet.learn(train);
                
                numOfTrainings++;
                numOfIterations+=mbp.getCurrentIteration();
                
                double accuracy = 0;
                double msne = 0;

                
                if(output==1) {
                   accuracy = calculateAccuracyOneOutput(neuralNet, test);
                   msne = calculateMsneOneOutput(neuralNet, test);
                } else {
                   accuracy =  calculateAccuracyMultiOutput(neuralNet, test);
                   msne = calculateMsneMultiOutput(neuralNet, test);
                }
              
                trainings.add(new Training(neuralNet,msne,accuracy));
            }
        }
        
        serializeMaxAcc();
        System.out.println("Iterations average: " + (double)numOfIterations/numOfTrainings);
        
    }
    
    public void serializeMaxAcc() {
        Training max = trainings.get(0);
        for(Training t : trainings) {
            if(max.accuracy<t.accuracy) {
                max=t;
            }
        }
        System.out.println("Serializing...");
        max.neuralNetwork.save("maxacc.nnet");
    }
    
    public double calculateAccuracyOneOutput(MultiLayerPerceptron neuralNet, DataSet test) {
       
        int tp=0,tn=0,fp=0,fn=0;
        
        for(DataSetRow row: test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double actual = row.getDesiredOutput()[0];
            double predicted = neuralNet.getOutput()[0];
            
            if(actual==1 && predicted>0.5) tp++;
            if(actual==0 && predicted<=0.5) tn++;
            if(actual==1 && predicted<=0.5) fn++;
            if(actual==0 && predicted>0.5)  fp++;
        }
        
        double accuracy = (double) (tp+tn)/(tp+tn+fp+fn);
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
        
    }
    
    public double calculateAccuracyMultiOutput(MultiLayerPerceptron neuralNet, DataSet test) {
        int tp=0,tn=0,fp=0,fn=0;
        
        String[] classLabels = {"s1","s2","s3"};
        ConfusionMatrix cm = new ConfusionMatrix(classLabels);
        
        for(DataSetRow row: test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            int actual = getMaxIndexArray(row.getDesiredOutput());
            int predicted = getMaxIndexArray(neuralNet.getOutput());
           
            cm.incrementElement(actual, predicted);
        }
        
        System.out.println(cm.toString());
        
        double accuracy=0;
        
        for(int i=0;i<output;i++) {
            tp = cm.getTruePositive(i);
            tn = cm.getTrueNegative(i);
            fp = cm.getFalsePositive(i);
            fn = cm.getFalseNegative(i);
            
            accuracy += (double)(tp+tn)/(tp+tn+fp+fn);
        }
        
        System.out.println("Accuracy: " + accuracy/output);
        return (double) accuracy/output;
    }
    
    public double calculateMsneOneOutput(MultiLayerPerceptron neuralNet, DataSet test) {
        
       double sumError=0,msne=0;
       
       for(DataSetRow row : test) {
           neuralNet.setInput(row.getInput());
           neuralNet.calculate();
           
           double actual = row.getDesiredOutput()[0];
           double predicted = neuralNet.getOutput()[0];
           
           sumError+=Math.pow(predicted-actual, 2);
       }
       
       msne = (double)sumError/(2*test.size());
       System.out.println("MSNE: " + msne);
       return msne; 
    }
    
    public double calculateMsneMultiOutput(MultiLayerPerceptron neuralNet, DataSet test) {
               
       double sumError=0,msne=0;
       
       for(DataSetRow row : test) {
           neuralNet.setInput(row.getInput());
           neuralNet.calculate();
           
           double[] actual = row.getDesiredOutput();
           double[] predicted = neuralNet.getOutput();
           
           for(int i=0;i<actual.length;i++) {
            sumError+=Math.pow(predicted[i]-actual[i], 2);
           }
       }
       
       msne = (double)sumError/(2*test.size());
       System.out.println("MSNE: " + msne);
       return msne;
    }
    
    public int getMaxIndexArray(double[] array) {
        int max = 0;
        for(int i=0;i<array.length;i++) {
            if(array[max]<array[i]) {
                max=i;
            }
        }
        
        return max;
    }
    
}
