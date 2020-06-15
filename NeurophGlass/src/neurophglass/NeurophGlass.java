package neurophglass;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;


public class NeurophGlass implements LearningEventListener {

    int input = 9;
    int output = 7;
    ArrayList<Training> trainings = new ArrayList<Training>();
    double[] learningRates = {0.2,0.4,0.6};
    int[] hiddenNeurons = {10,20,30};
    String[] labels = new String[]{"s1","s2","s3","s4","s5","s6","s7"};
    
    public static void main(String[] args) {
        new NeurophGlass().run();
    }
    
    public void run() {
        
        System.out.println("Creating from file...");
        DataSet data = DataSet.createFromFile("glass.csv", input, output, ",");
        System.out.println("Normalizing...");
        MaxNormalizer max = new MaxNormalizer(data);
        max.normalize(data);
        System.out.println("Shuffling...");
        data.shuffle();
        
        System.out.println("Splitting...");
        DataSet[] split = data.split(0.65,0.35);
        DataSet train = split[0];
        DataSet test = split[1];
        
        int numberOfIterations=0;
        int numberOfTrainings=0;
        
        for(double lr: learningRates) {
            for(int hn: hiddenNeurons) {
                System.out.println("Learning rate: " + lr + ". Hidden Neurons: " + hn);
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hn,output);
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation mbp = (MomentumBackpropagation) neuralNet.getLearningRule();
                mbp.setMomentum(0.6);
                mbp.setMaxIterations(100); //Not given in the task
                mbp.addListener(this);
                
                neuralNet.learn(train);
                
                double acc = calculateAccuracy(neuralNet,test);
                calculateAccuracyNeuroph(neuralNet,test);
                double msne = calculateMsne(neuralNet,test);
                calculateMsneNeuroph(neuralNet,test);
                trainings.add(new Training(neuralNet,msne,acc));
                
                numberOfIterations+=mbp.getCurrentIteration();
                numberOfTrainings++;              
            }
        }
        
        System.out.println("Average Iterations: " + (double)numberOfIterations/numberOfTrainings);
        serialize();
    }
    
    public double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        
        double sumAcc=0;
        ConfusionMatrix cm = new ConfusionMatrix(labels);
        
        for(DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(neuralNet.getOutput());
            
            cm.incrementElement(actual, predicted);
        }
        
        for(int i=0;i<output;i++) {
            int tp = cm.getTruePositive(i);
            int tn = cm.getTrueNegative(i);
            int fp = cm.getFalsePositive(i);
            int fn = cm.getFalseNegative(i);
            sumAcc+= (double)(tp+tn)/(tp+tn+fn+fp);
        }
        
        double acc = (double) sumAcc/output;
        System.out.println("Accuracy: " + acc);
        return acc;
    }
    
    public void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.MultiClass(labels));
        eval.evaluate(neuralNet, test);
        
        ConfusionMatrix cm = eval.getEvaluator(ClassifierEvaluator.MultiClass.class).getResult();
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        
        System.out.println("Accuracy: " + average.accuracy);  
    }
    
    private double calculateMsne(MultiLayerPerceptron neuralNet, DataSet test) {
        double sum=0;
        
        for(DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double[] actual = row.getDesiredOutput();
            double[] predicted = neuralNet.getOutput();
            
            for(int i=0;i<actual.length;i++) {
                sum+=Math.pow(actual[i]-predicted[i], 2);
            }
        }
                     
        double msne = (double) sum/(2*test.size());
        System.out.println("Msne: " + msne);
        return msne;
    }

    private void calculateMsneNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.MultiClass(labels));
        eval.evaluate(neuralNet, test);
        
        System.out.println("Msne Neuroph: " + eval.getMeanSquareError());
    }
    
    private void serialize() {
        
        Training max = trainings.get(0);
        for(Training t: trainings) {
            if(max.accuracy<t.accuracy) max=t;
        }
        
        System.out.println("Saving file...");
        //max.neuralNet.save("file.nnet");
        
    }
    
    private int getMaxIndex(double[] array) {
        
        int max = 0;
        
        for(int i=0;i<array.length;i++) {
            if(array[max]<array[i]) max=i;
        }
        
        return max;
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {

        MomentumBackpropagation mbp = (MomentumBackpropagation)event.getSource();
        if(mbp.isStopped()) {
            System.out.println("Training stopped: TotalNet: " + mbp.getTotalNetworkError() + " Iterations: " + mbp.getCurrentIteration() );
        }
    }
    
}
