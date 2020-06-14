package neurophdiabetes;

import neurophdiabetes.Training;
import java.util.ArrayList;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;

public class NeurophDiabetes implements LearningEventListener {

    int input = 8;
    int output = 1;
    double[] learningRates = {0.2, 0.4, 0.6};
    ArrayList<Training> trainings = new ArrayList<>();

    public static void main(String[] args) {
        new NeurophDiabetes().run(); //ovo uvek u main
    }

    private void run() {

        System.out.println("Creating DataSet...");
        DataSet dataSet = DataSet.createFromFile("diabetes_data.csv", input, output, ",");
        System.out.println("Normalizing...");
        MaxNormalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        System.out.println("Shuffling...");
        dataSet.shuffle();

        System.out.println("Spliting...");
        DataSet[] dataSets = dataSet.createTrainingAndTestSubsets(0.70, 0.30);
        DataSet train = dataSets[0];
        DataSet test = dataSets[1];
        
        //Inicijalizacija varijabli koje nam trebaju za prosecan broj iteracija
        int numberOfItterations = 0;

        for(double lr : learningRates) {
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,20,10,output);
            neuralNet.setLearningRule(new BackPropagation());
            BackPropagation bp = (BackPropagation) neuralNet.getLearningRule();
            
            bp.setLearningRate(lr);
            bp.setMaxError(0.07);
            bp.addListener(this);
            
            neuralNet.learn(train);
            
            System.out.println("Number of iterations for lr: " + lr + " is  " + bp.getCurrentIteration());
            
            double msne = calculateMsne(neuralNet,test);
            calculateMsneNeuroph(neuralNet,test);
            double accuracy = calculateAccuracy(neuralNet,test);
            calculateAccuracyNeuroph(neuralNet,test);
            trainings.add(new Training(neuralNet,accuracy,msne));
        }

        saveMaxAccuracyNet();
    }

    private void saveMaxAccuracyNet() {
        Training max = trainings.get(0);
        for (int i = 1; i < trainings.size(); i++) {
            if (max.accuracy < trainings.get(i).accuracy) {
                max = trainings.get(i);
            }
        }

        System.out.println("Saving net with max accuracy...");
        //max.neuralNet.save("net.nnet");
    }

    private double calculateMsne(MultiLayerPerceptron neuralNet, DataSet test) {
        double sumError = 0;
        
        for(DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double predicted = neuralNet.getOutput()[0];
            double actual = row.getDesiredOutput()[0];
            
            sumError += Math.pow(actual-predicted,2);
        }
        
        double msne = (double)sumError/(2*test.size());
        System.out.println("Msne: " + msne);
        return msne;
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        double total = 0,good=0;
        
        for(DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double predicted = neuralNet.getOutput()[0];
            double actual = row.getDesiredOutput()[0];
            
           if((actual==1 && predicted>0.5) || (actual==0 && predicted<=0.5)) good++;
           total++;
        }
        
        double accuracy = (double)good/total;
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    }

    private void calculateMsneNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(neuralNet, test);
        
        System.out.println("Msne Neuroph: " + eval.getMeanSquareError());
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

    @Override
    public void handleLearningEvent(LearningEvent event) {
        BackPropagation bp = (BackPropagation) event.getSource();
        
        if(bp.isStopped()) {
            System.out.println("Training stopped at iteration " + bp.getCurrentIteration() + " with " + bp.getTotalNetworkError() + " error");
        }
    }

}

