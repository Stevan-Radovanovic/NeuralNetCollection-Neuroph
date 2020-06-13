
package neurophbreastcancer;

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

public class NeurophBreastCancer implements LearningEventListener {

    //This part is always the same
    //Look through the exam paper for info
    ArrayList<Training> trainings = new ArrayList<Training>();
    int input = 30;
    int output = 1;
    double[] learningRates = {0.2, 0.4, 0.6};
    int[] hiddenNeurons = {10, 20, 30};
    
    public static void main(String[] args) {
        //Always the same
        new NeurophBreastCancer().run();
    }
    
    public void run() {
        
        System.out.println("Creating file...");
        DataSet data = DataSet.createFromFile("breast_cancer_data.csv", this.input, this.output, ",");
        MaxNormalizer normalizer = new MaxNormalizer(data);
        System.out.println("Normalizing...");
        normalizer.normalize(data);
        System.out.println("Shuffling...");
        data.shuffle();
        
         System.out.println("Splitting...");
        DataSet[] trainAndTest = data.split(0.65,0.35);
        DataSet train = trainAndTest[0];
        DataSet test = trainAndTest[1];
        
        int numberOfTrainings = 0;
        int numberOfIterations = 0;
        
        for (double lr : learningRates) {
            for (int hn : hiddenNeurons) {
               System.out.println("Starting Training for lr: " + lr + " with " + hn + " hidden neurons.");
               MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input,hn,output);
               neuralNet.setLearningRule(new MomentumBackpropagation());
               MomentumBackpropagation mbp = (MomentumBackpropagation) neuralNet.getLearningRule();
               mbp.setMomentum(0.7);
               mbp.setLearningRate(lr);
               mbp.addListener(this);
               
               neuralNet.learn(train);
               
               double msne = calculateMeanSquaredError(neuralNet,test);
               calculateMeanSquaredErrorNeuroph(neuralNet, test);
               double acc =  calculateAccuracy(neuralNet,test);
               calculateAccuracyNeuroph(neuralNet, test);

               numberOfIterations+=mbp.getCurrentIteration();
               numberOfTrainings++;
              
               trainings.add(new Training(neuralNet,msne,acc));
            }
        }

        System.out.println("Average iteration number: " + (double) numberOfIterations / numberOfTrainings);
        saveNetWithMinError();
        System.out.println("All trainings finished");
        
    }
    
  
    private double calculateMeanSquaredError(MultiLayerPerceptron neuralNet, DataSet test) {
        
        double msne=0;
    
        for(DataSetRow row: test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double actual = row.getDesiredOutput()[0];
            double predicted = neuralNet.getOutput()[0];
            
            msne += Math.pow(actual-predicted, 2);
        }
        
        msne = (double)msne/(2*test.size());
        
        System.out.println("Msne: " + msne);
        return msne;
    
    }

    private void calculateMeanSquaredErrorNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {

        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(neuralNet, test);
       
        System.out.println("Msne Neuroph: " + eval.getMeanSquareError());
        
    }

    private void saveNetWithMinError() {
        Training min = trainings.get(0);
        for (int i = 1; i < trainings.size(); i++) {
            if (min.error > trainings.get(i).error) {
                min = trainings.get(i);
            }
        }

        System.out.println("Saving file...");
        //min.neuralNet.save("neuralNet.nnet");
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {

        int total=0,good=0;
    
        for(DataSetRow row: test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double actual = row.getDesiredOutput()[0];
            double predicted = neuralNet.getOutput()[0];
            
            if((actual==1 && predicted>0.5) || (actual==0 && predicted<=0.5)) good++;
            total++;
        }
        
        double acc  = (double)good/total;
        
        System.out.println("Accuracy: " + acc);
        return acc;

    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation mbp = (MomentumBackpropagation) event.getSource();
        
        
        if(mbp.isStopped()) {
            System.out.println("Training completed");
            System.out.println("Last iteration: " + mbp.getCurrentIteration());
            System.out.println("Total Network Error: " + mbp.getTotalNetworkError());
        } else {
            /*
            System.out.println("Iteration number: " + mbp.getCurrentIteration() + 
                    ". Current max error: " + mbp.getTotalNetworkError());
            */
        }
    }

    private void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation eval = new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(neuralNet, test);
        
        ConfusionMatrix confusion = eval.getEvaluator(ClassifierEvaluator.Binary.class).getResult();
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusion);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        System.out.println("Accuracy Neuroph: " + average.accuracy);    }
}
