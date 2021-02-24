import java.util.*; 
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class LinearClassifierRunner {

    public static void main(String[] args) {
        int dimNum = 819;
        int numPasses = 4;
        int numIterations = 2;
        String trainFile = "pa3train.txt";
        String testFile = "pa3test.txt";

        LinearClassifier lc = new LinearClassifier(dimNum, trainFile, testFile);

        // part 1

        lc.useClasses1And2();
        //print("Part 1:");
        //lc.runPerceptronPasses(numPasses);


        // part 2

        print("Part 2:");
        double[] wT = lc.runGradientDescent(numIterations);
        //lc.runMethodTests();

        double error = lc.getLogisticRegressionTrainingError(wT);
        print("after " + numIterations + " runs of gradient descent, we get the following training error: " + error);

        



        print("todo: finish LinearClassifierRunner main");

    }

    private static void print(String text) {
        System.out.println(text);
    }

    


}