
package neurophbreastcancer;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;

public class NeurophBreastCancer {

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
        
        DataSet data;
        data = DataSet.createFromFile("breast_cancer_data.csv", this.input, this.output, ",");
        MaxNormalizer normalizer = new MaxNormalizer(data);
        normalizer.normalize(data);
        data.shuffle();
        
        DataSet[] trainAndTest = data.createTrainingAndTestSubsets(0.65, 0.35);
        DataSet train = trainAndTest[0];
        DataSet test = trainAndTest[1];
        
        int numberOfTrainings = 0;
        int numberOfIterations = 0;
        
        for (double lr : learningRates) {
            for (int hn : hiddenNeurons) {
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(input, hn, output);
                MomentumBackpropagation bp = (MomentumBackpropagation) neuralNet.getLearningRule();
                bp.setMomentum(0.7);
                bp.setLearningRate(lr);
                bp.setMaxError(0.07);

                neuralNet.learn(train);

                calculateMeanSquaredError(neuralNet, test);
                calculateMeanSquaredErrorNeuroph(neuralNet, test);

                numberOfTrainings++;
                numberOfIterations += bp.getCurrentIteration();

            }
        }

         saveNetWithMinError();
        System.out.println("Srednja vrednost broja iteracija: " + (double) numberOfIterations / numberOfTrainings);
        
    }
    
  
    private void calculateMeanSquaredError(MultiLayerPerceptron neuralNet, DataSet test) {
        double sumError = 0, meanSquaredError;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            double predicted = neuralNet.getOutput()[0];
            double actual = row.getDesiredOutput()[0];

            double error = Math.pow((predicted - actual), 2);
            sumError += error;
        }

        meanSquaredError = (double) sumError / (2 * test.size());
        System.out.println("Srednja kvadratna greska treninga: " + meanSquaredError);

        Training t = new Training(neuralNet, meanSquaredError);
        trainings.add(t);
    }

    private void calculateMeanSquaredErrorNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, test);

        System.out.println("Srednja kvadratna greska treninga - njihova metoda: " + evaluation.getMeanSquareError());
    }

    private void saveNetWithMinError() {
        Training min = trainings.get(0);
        for (int i = 1; i < trainings.size(); i++) {
            if (min.error > trainings.get(i).error) {
                min = trainings.get(i);
            }
        }

        min.neuralNet.save("neuralNet.nnet");
    }
}
