
/*
Na samom pocetku obavezno ucitati sve neophodne jar fajlove
*/

package neurophdiabetes;

import neurophdiabetes.Training;
import java.util.ArrayList;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;

public class NeurophDiabetes {

    int inputCount = 8;
    int outputCount = 1;
    double[] learningRates = {0.2, 0.4, 0.6};
    int[] hiddenNeurons = {20,10};
    ArrayList<Training> trainings = new ArrayList<>();

    public static void main(String[] args) {
        new NeurophDiabetes().run();
    }

    private void run() {

        DataSet dataSet = DataSet.createFromFile("diabetes_data.csv", inputCount, outputCount, ",");
        MaxNormalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        dataSet.shuffle();


        DataSet[] dataSets = dataSet.createTrainingAndTestSubsets(0.70, 0.30);
        DataSet train = dataSets[0];
        DataSet test = dataSets[1];
        
        int numberOfTrainings = 0;
        int numberOfItterations = 0;

        for (double lr : learningRates) {
            for(int hn : hiddenNeurons) {
                System.out.println("Training neural network for learning rate " + lr);
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hn, outputCount);
                MomentumBackpropagation bp = (MomentumBackpropagation) neuralNet.getLearningRule();
                bp.setMaxError(0.07);
                bp.setLearningRate(lr);

                neuralNet.learn(train);

                numberOfTrainings++;
                numberOfItterations+=bp.getCurrentIteration();
                double accuracy = evaluateAccuracy(neuralNet, test);
                evaluateAccuracyNeuroph(neuralNet, test);

                Training t = new Training(neuralNet, accuracy);
                trainings.add(t);
            }
        }

        System.out.println((double)numberOfItterations/numberOfTrainings);
        saveMaxAccuracyNet();
    }

    private void evaluateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, test);

        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.Binary.class);
        ConfusionMatrix cm = evaluator.getResult();
        System.out.println("Matrica konfuzije Neuroph: ");
        System.out.println(cm.toString());

        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);

        System.out.println("Accuracy neuroph: " + average.accuracy);

    }

    private double evaluateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        int tp = 0, tn = 0, fp = 0, fn = 0;

        for (DataSetRow row : test) {
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

        double accuracy = (double) (tp + tn) / (tp + tn + fp + fn);

        System.out.println("Accuracy: " + accuracy);

        return accuracy;
    }

    private void saveMaxAccuracyNet() {
        Training max = trainings.get(0);
        for (int i = 1; i < trainings.size(); i++) {
            if (max.accuracy < trainings.get(i).accuracy) {
                max = trainings.get(i);
            }
        }

        System.out.println("Saving net with max accuracy");
        max.neuralNet.save("net.nnet");
    }

}

