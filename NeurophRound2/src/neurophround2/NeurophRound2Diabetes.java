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

public class NeurophRound2Diabetes {
    
    int input=8;
    int output=1;
    ArrayList<Training> trainings = new ArrayList<Training>();
    double[] learningRates = {0.2,0.4,0.6};
    
    public void run() {
        
        System.out.println("Starting.......");
        DataSet data = DataSet.createFromFile("diabetes_data.csv", input, output, ",");
        MaxNormalizer max = new MaxNormalizer(data);
        max.normalize(data);
        data.shuffle();
        
        DataSet[] yo = data.split(0.7,0.3);
        DataSet train = yo[0];
        DataSet test = yo[1];
        
        for(double lr: learningRates) {
            
            System.out.println("Learning rate: " + lr);
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,20,10,output);
            neuralNet.setLearningRule(new BackPropagation());
            BackPropagation bp = neuralNet.getLearningRule();
            bp.setMaxError(0.07);
            
            neuralNet.learn(train);
        
            double accuracy = calculateAccuracy(neuralNet,test);
            calculateAccuracyNeuroph(neuralNet,test);
            Training t = new Training();
            t.neuralNet = neuralNet;
            t.accuracy = accuracy;
            trainings.add(t);
            
            System.out.println("Number of iterations for " + lr + " is " + bp.getCurrentIteration());
        }
        
        serializeMaxAcc();
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
            
            int accurate=0,total=0;
            
            for(DataSetRow row : test) {
                
                neuralNet.setInput(row.getInput());
                neuralNet.calculate();
                
                double actual = row.getDesiredOutput()[0];
                double predicted = neuralNet.getOutput()[0];
                
                if((actual==1 && predicted>0.5) || (actual==0 && predicted<=0.5)) accurate++;
                total++;
                        
            }
        
            double acc = (double)accurate/total;
            System.out.println("Accuracy: " + acc);
            return acc;
    }

    private void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(neuralNet,test);
        
        ConfusionMatrix cm = eval.getEvaluator(ClassifierEvaluator.Binary.class).getResult();
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        System.out.println("Accuracy Neuroph: " + average.accuracy);
    }

    private void serializeMaxAcc() {
        Training max = trainings.get(0);
        for(Training t: trainings) {
            if(max.accuracy<t.accuracy) max=t;
        }
        
        System.out.println("Saving file...");
        //max.neuralNet.save("file.nnet");
    }
}
