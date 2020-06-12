
/*
Na samom pocetku obavezno ucitati sve neophodne jar fajlove
*/

/*
U ovom zadatku skriveni neuroni treba da se ubace automatski, a ne preko petlje jedan po jedan, tu je greska
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

    int inputCount = 8; //Pise u tekstu zadatka, broj ulaznih varijabli
    int outputCount = 1; //pise u tekstu zadatka, broj izlaznih varijabli
    double[] learningRates = {0.2, 0.4, 0.6}; //pise u tekstu
    int[] hiddenNeurons = {20,10}; //pise u tekstu
    ArrayList<Training> trainings = new ArrayList<>();

    public static void main(String[] args) {
        new NeurophDiabetes().run(); //ovo uvek u main
    }

    private void run() {

        //Ovaj deo uvek isti, osim ako eventualno ne traze na kraju da se promesaju redovi, onda bez shuffle
        DataSet dataSet = DataSet.createFromFile("diabetes_data.csv", inputCount, outputCount, ",");
        MaxNormalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        dataSet.shuffle();

        //Podela na train  i test, odnos pise u tekstu
        DataSet[] dataSets = dataSet.createTrainingAndTestSubsets(0.70, 0.30);
        DataSet train = dataSets[0];
        DataSet test = dataSets[1];
        
        //Inicijalizacija varijabli koje nam trebaju za prosecan broj iteracija
        int numberOfTrainings = 0;
        int numberOfItterations = 0;

        //dve for petlje, spoljasnja za sve learningRates, unutrasnja za sve skrivene neurone
        for (double lr : learningRates) {
            for(int hn : hiddenNeurons) {
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hn, outputCount);
                MomentumBackpropagation bp = (MomentumBackpropagation) neuralNet.getLearningRule();
                bp.setMaxError(0.07); //ovo uvek ovako
                bp.setLearningRate(lr);

                neuralNet.learn(train);

                numberOfTrainings++;
                numberOfItterations+=bp.getCurrentIteration();
                
                //Ovo pozivamo i pravimo ako nam je receno da sami pravimo funkcije
                double accuracy = evaluateAccuracy(neuralNet, test);
                
                //Ovo pozivamo ako nam nije receno da sami pravimo funkcije
                evaluateAccuracyNeuroph(neuralNet, test);

                Training t = new Training(neuralNet, accuracy);
                trainings.add(t);
            }
        }

        //Stampamo prosecan broj iteracija
        System.out.println((double)numberOfItterations/numberOfTrainings);
       
        //Funkcija za serijalizaciju u fajl
        saveMaxAccuracyNet();
    }

    //Neuroph racuna sam
    private void evaluateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        
        //Ovde sve napamet
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

    //Rucno pravljenje funkcije
    private double evaluateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        int tp = 0, tn = 0, fp = 0, fn = 0;

        //Za svaki red u test data setu
        for (DataSetRow row : test) {
            //Ova dva reda napamet
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            //Stvarna i predicted vrednost
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

        //Sve sto smo tacno predvideli, bilo pozitivno, bilo negativno
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

