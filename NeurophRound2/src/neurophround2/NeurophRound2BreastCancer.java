
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

public class NeurophRound2BreastCancer {

    
    public static void main(String[] args) {
        new NeurophRound2BreastCancer().run();
    }
    
    //Serilization skipped
    
    int input=30;
    int output=1;
    ArrayList<Training> trainings = new ArrayList<Training>();
    int[] hidden = {10,20,30};
    double[] learningRates = {0.2,0.4,0.6};
    
    public void run() {
        
        DataSet data = DataSet.createFromFile("breast_cancer_data.csv", input, output, ",");
        MaxNormalizer norm = new MaxNormalizer(data);
        norm.normalize(data);
        data.shuffle();
        
        DataSet[] array = data.split(0.7,0.3);
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
                mbp.setMaxError(0.02);
                
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
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(neuralNet, test);
        
        ConfusionMatrix cm = eval.getEvaluator(ClassifierEvaluator.Binary.class).getResult();
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        
        System.out.println("Accuracy Neuroph: " + average.accuracy);
        
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        
         int tp = 0, tn = 0, fp = 0, fn = 0;
         
        for (DataSetRow row : test) {

            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            double actual = row.getDesiredOutput()[0];
            double predicted = neuralNet.getOutput()[0];

            if (actual == 1.0 && predicted > 0.5) tp++;
            if (actual == 0.0 && predicted <=0.5) tn++;
            if (actual == 1.0 && predicted <= 0.5) fn++;
            if (actual == 0.0 && predicted > 0.5) fp++;   
        }
      
            double accuracy = (double)(tp+tn)/(tp+tn+fp+fn);
            System.out.println("Accuracy: " + accuracy);
            return accuracy;
    }

    private double calculateMsne(MultiLayerPerceptron neuralNet, DataSet test) {
        
        double sumError=0,msne=0;
        
        for (DataSetRow row : test) {

            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            double actual = row.getDesiredOutput()[0];
            double predicted = neuralNet.getOutput()[0];

            sumError+=Math.pow(predicted-actual,2);           
        }
        
        msne = (double) sumError / (2*test.size());
        System.out.println("Msne: " + msne);
        return msne;
        
    }

    private void calculateMsneNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(neuralNet, test);
        
        System.out.println("Msne Neuroph: " + eval.getMeanSquareError());
        
    }
}
